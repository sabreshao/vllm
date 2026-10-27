# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1250 (wave32) port of ``kernel.py``: the same block scores over the same
e4m3_rn_satfinite(e2m1 x scale) keys.

gfx1250's fp8 WMMA does not accumulate exactly, so keys and index-q are widened
from e4m3 to bf16 (exact) and scored with v_wmma_f32_16x16x32_bf16, four K steps
per 16-token tile. Lane (col, h) holds token 16 t + col, dims 64 h .. 64 h + 63
in the key's dim order (gfx950's lanes (col, 2h) and (col, 2h + 1)); a chunk
holds up to 32 requests, one per lane. A page keeps its NVFP4 codes in registers
and dequantizes one tile at a time, so every column group sharing the page is
scored without holding the whole page in bf16.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops import flydsl_gfx1250 as g
from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    as_f32,
    bits,
    dequant8,
    global_i32_ptr,
    key_order4,
    maxf_nn,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_index_decode.kernel import (
    BLK,
    MAX_ROWS,
    MT,
    NW,
    PAGE_BYTES,
    PAGE_CODES,
    UMAX,
    D,
    block_sentinels,
    padded_rows,
)

WAVE = g.WAVE
MAX_REQS = WAVE  # requests per chunk, one per lane
NKS = D // 32  # bf16 K steps: a lane half supplies 16 of its 64 dims per step

__all__ = [
    "BLK",
    "D",
    "MAX_ROWS",
    "NW",
    "UMAX",
    "chunk_max",
    "compile_score_decode",
    "padded_rows",
]


def chunk_max(rows: int) -> int:
    """Requests per chunk: one per lane, and 256 Q rows (32 KB) in LDS."""
    return min(MAX_REQS, 256 // rows)


def compile_score_decode(rows: int, heads: int, buffered: bool = False):
    """Build the scorer (see ``kernel.compile_score_decode``)."""
    assert 1 <= rows <= MAX_ROWS and heads in (1, 2, 4) and rows % heads == 0
    H = heads
    QL = rows // H
    R = padded_rows(rows)  # a request's columns; column j = query row j // H
    RPC = max(1, 16 // R)  # requests per 16-column group
    CPR = max(1, R // 16)  # 16-column groups per request
    GMASK = (1 << RPC) - 1
    QROWS = chunk_max(R) * R
    QROW_BYTES = D
    NTH = WAVE * NW
    name = f"m3_nvfp4_index_score_decode_gfx1250_r{rows}h{H}" + (
        "_buf" if buffered else ""
    )
    SB_STRIDE = UMAX + 1  # padded against LDS bank conflicts

    @fx.struct
    class Shared:
        q: fx.Array[fx.Uint8, QROWS * QROW_BYTES, 16]
        lut: fx.Array[fx.Int32, 512, 16]
        seq: fx.Array[fx.Int32, MAX_REQS]
        # The unit's scores as f32 bits, plus one dummy slot per lane.
        sbuf: fx.Array[fx.Int32, (QROWS * SB_STRIDE + WAVE) if buffered else 16, 16]

    @flyc.kernel(name=name, known_block_size=[NTH, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [num_reqs * QL, H, 128] e4m3
        arg_kv: fx.Int64,  # [num_blocks, 128, 72] uint8
        arg_lut: fx.Int64,  # nvfp4_e4m3_lut [256] int64
        arg_score: fx.Int64,  # [H, num_reqs * QL, S] fp32
        arg_bt: fx.Int64,  # [num_reqs, >= max_blocks] int32
        arg_seq: fx.Int64,  # [num_reqs] int32
        i64_page_stride: fx.Int64,
        i32_q_st: fx.Int32,  # q strides (bytes)
        i32_q_sh: fx.Int32,
        i64_q_bytes: fx.Int64,
        i32_s_sh: fx.Int32,  # score strides (elements)
        i32_s_st: fx.Int32,
        i64_s_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_chunk: fx.Int32,  # requests per chunk
        i32_nch: fx.Int32,
        i32_splits: fx.Int32,  # workgroups per chunk
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_per_wave: fx.Int32,  # 1: each wave walks its own positions
        i32_unit: fx.Int32,  # consecutive positions per unit, 0: none
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % WAVE
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // WAVE))
        col, h = lane % 16, lane // 16
        ch = pid % i32_nch
        split = pid // i32_nch
        req0 = ch * i32_chunk
        left = i32_num_reqs - req0
        nr = (left < i32_chunk).select(left, i32_chunk)

        # Lane r holds request req0 + r.
        r_ok = lane < nr
        seq_r = fx.Int32(global_i32_ptr(arg_seq)[r_ok.select(req0 + lane, fx.Int32(0))])
        seq_r = r_ok.select(seq_r, fx.Int32(0))
        nblk_r = (seq_r + (BLK - 1)) // BLK
        first_row = seq_r - (QL - 1)
        # Every row sees blocks before rbf_r whole: init / local ones there only
        # need their sentinel.
        rbf_r = ((first_row < 0).select(fx.Int32(0), first_row) + (BLK - 1)) // BLK
        ll = nblk_r - i32_local
        loc_last_r = (ll < 0).select(fx.Int32(0), ll)
        fm_r = seq_r - QL  # tokens past this are masked for some row

        btr = buffer_ops.create_buffer_resource_from_addr(
            arg_bt, num_records_bytes=i64_bt_bytes
        )
        sr = buffer_ops.create_buffer_resource_from_addr(
            arg_score, num_records_bytes=i64_s_bytes
        )
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=i64_q_bytes
        )

        q16 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.q.ptr),
                fx.make_layout(QROWS * QROW_BYTES // 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        atom16 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        seq_l = smem.seq.ptr
        scb = smem.sbuf.ptr
        per_wave = i32_per_wave != fx.Int32(0)
        if const_expr(buffered):
            direct = per_wave | (i32_unit == fx.Int32(0))
            to_lds = (i32_per_wave == fx.Int32(0)) & (i32_unit != fx.Int32(0))
            usz = (i32_unit == fx.Int32(0)).select(fx.Int32(1), i32_unit)
        else:
            direct = i32_unit >= fx.Int32(0)
            to_lds = i32_unit < fx.Int32(0)
            usz = fx.Int32(1)

        def prologue():
            """Q rows, the e4m3 table and the request lengths -> LDS.

            Row i = request i // R, column j = i % R (query row j // H, head
            j % H); 16 B column c8 sits at c8 ^ (i & 7), in the key's dim order.
            """
            for k in range_constexpr(QROWS * 8 // NTH):
                u = tx + k * NTH
                i = u // 8
                c8 = u % 8
                rr = i // R
                j = i % R
                # Missing requests' and padding rows stay unwritten; their
                # scores are never stored.
                if (rr < nr) & (j // H < QL):
                    tq = (req0 + rr) * QL + j // H
                    off = (tq * i32_q_st + (j % H) * i32_q_sh + c8 * 16) // 4
                    d = Vec(
                        buffer_ops.buffer_load(
                            qr, off, vec_width=4, dtype=fx.Int32, mask=rr < nr
                        )
                    )
                    d = [fx.Int32(d[e]) for e in range_constexpr(4)]
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r.store(key_order4(d))
                    fx.copy(atom16, r, fx.slice(q16, (None, i * 8 + (c8 ^ (i & 7)))))
            lut_g = global_i32_ptr(arg_lut)
            lut_l = smem.lut.ptr
            for e in range_constexpr(512 // NTH):
                lut_l[tx + e * NTH] = fx.Int32(lut_g[tx + e * NTH])
            seq_l[lane] = seq_r

        prologue()
        fx.gpu.barrier()

        lut8 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.lut.ptr), fx.make_layout(512, 1)
            ),
            fx.make_layout(2, 1),
        )
        lut_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)

        def lut_load(sbyte):
            r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
            fx.copy(lut_atom, fx.slice(lut8, (None, sbyte)), r)
            v = Vec(r.load())
            return fx.Int32(v[0]), fx.Int32(v[1])

        def q_frag(cg):
            """B operands of column group cg: per K step ks, dims 64 h + 16 ks ..
            + 15 of Q row cg * 16 + col (16 B column 4 h + ks), widened to bf16."""
            i = cg * 16 + col
            steps = []
            for ks in range_constexpr(NKS):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(
                    atom16, fx.slice(q16, (None, i * 8 + ((4 * h + ks) ^ (i & 7)))), r
                )
                d = Vec(r.load())
                lo = g.cvt_fp8x8_bf16(d[0], d[1])
                hi = g.cvt_fp8x8_bf16(d[2], d[3])
                steps.append(
                    Vec.from_elements(
                        [lo[e] for e in range_constexpr(4)]
                        + [hi[e] for e in range_constexpr(4)],
                        fx.Int32,
                    )
                )
            return steps

        code_off = (col * 64 + h * 32) // 4  # dwords
        scale_off = col * 2 + h  # dword: groups 4 h .. 4 h + 3 of token col

        def page_rsrc(page):
            return buffer_ops.create_buffer_resource_from_addr(
                arg_kv + fx.Int64(page) * i64_page_stride, num_records_bytes=PAGE_BYTES
            )

        def load_tile(kr, ok, t):
            """Tile t of a page: this lane's 32 B of codes (token 16 t + col) and
            its four group scales; no memory access when not ``ok``."""
            c = []
            for hh in range_constexpr(2):
                v = Vec(
                    buffer_ops.buffer_load(
                        kr,
                        code_off + 4 * hh,
                        vec_width=4,
                        dtype=fx.Int32,
                        mask=ok,
                        soffset_bytes=t * 1024,
                        cache_modifier=g.TH_NT,
                    )
                )
                c += [fx.Int32(v[e]) for e in range_constexpr(4)]
            sc = fx.Int32(
                buffer_ops.buffer_load(
                    kr,
                    scale_off,
                    vec_width=1,
                    dtype=fx.Int32,
                    mask=ok,
                    soffset_bytes=PAGE_CODES + t * 128,
                    cache_modifier=g.TH_NT,
                )
            )
            return c + [sc]

        def dequant_tile(tile):
            """A operands of a tile per K step: code dwords 2 ks, 2 ks + 1 (dims
            64 h + 16 ks ..) under group scale byte ks, widened to bf16."""
            sc = tile[8]
            steps = []
            for ks in range_constexpr(NKS):
                tab = lut_load((sc >> (8 * ks)) & 0xFF)
                f = []
                for e in range_constexpr(2):
                    lo, hi = dequant8(tile[2 * ks + e], *tab)
                    bf = g.cvt_fp8x8_bf16(lo, hi)
                    f += [bf[i] for i in range_constexpr(4)]
                steps.append(Vec.from_elements(f, fx.Int32))
            return steps

        zero8 = Vec.filled(8, 0.0, fx.Float32)
        neg_inf = fx.Float32(float("-inf"))

        def frag_max(tiles, qfs, masked, lims):
            """Per column group: the max of its column's scores over the page."""
            runs = [neg_inf] * len(qfs)
            for t in range_constexpr(MT):
                kf = dequant_tile(tiles[t])
                for gi in range_constexpr(len(qfs)):
                    acc = zero8
                    for ks in range_constexpr(NKS):
                        acc = g.wmma_16x16x32(kf[ks], qfs[gi][ks], acc, False)
                    run = runs[gi]
                    for i in range_constexpr(8):
                        x = acc[i]
                        if const_expr(masked):
                            x = (lims[gi] >= t * 16 + i).select(x, neg_inf)
                        run = maxf_nn(run, x)
                    runs[gi] = run
            return [g.xlane2(r, maxf_nn) for r in runs]

        def row_value(mx, seq, q_off, b):
            return block_sentinels(mx, seq - QL + q_off + 1, b, i32_init, i32_local)

        def put(i_row, jpos, v, cond, addr):
            """Score v of Q row i_row at unit position jpos, to memory or to the
            LDS buffer; lanes with nothing to buffer write a dummy slot."""
            if const_expr(not buffered):
                buffer_ops.buffer_store(v, sr, addr, mask=cond)
            else:
                buffer_ops.buffer_store(v, sr, addr, mask=cond & direct)
                idx = (cond & to_lds).select(
                    i_row * SB_STRIDE + jpos, QROWS * SB_STRIDE + lane
                )
                (scb + idx).store(bits(v))  # `scb[idx] = v` would rebind scb

        def flush(ub):
            """Store the unit at block ub: 4 consecutive blocks per thread and
            pass."""
            NCH = UMAX // 4
            k = fx.Int32(0)
            while k < fx.Int32((QROWS * NCH + NTH - 1) // NTH):
                item = tx + k * NTH
                i = item // NCH
                c = item % NCH
                rr = i // R
                jc = i % R
                qoff = jc // H
                iv = i < QROWS
                nb_c = (fx.Int32(seq_l[iv.select(rr, fx.Int32(0))]) + (BLK - 1)) // BLK
                b0c = ub + c * 4
                ok = iv & (rr < nr) & (qoff < QL) & (c * 4 < i32_unit)
                addr = (jc % H) * i32_s_sh + ((req0 + rr) * QL + qoff) * i32_s_st + b0c
                for e in range_constexpr(4):
                    v = as_f32(
                        fx.Int32(scb[iv.select(i, fx.Int32(0)) * SB_STRIDE + c * 4 + e])
                    )
                    buffer_ops.buffer_store(
                        v,
                        sr,
                        addr + e,
                        mask=ok & (b0c + e < nb_c) & (c * 4 + e < i32_unit),
                    )
                k = k + 1

        def score_cgs(tiles, hm, masked_any, cgs, b, jpos):
            """Scores of column groups cgs, masked where a request is at its last
            blocks, with sentinels."""
            info = []
            for cg in cgs:
                i = cg * 16 + col
                crr = i // R
                st_j = i % R
                info.append((cg, crr, st_j // H, st_j % H))
            qfs = [q_frag(cg) for cg, _, _, _ in info]
            seqs = [fx.Int32(seq_l[crr]) for _, crr, _, _ in info]
            mxs = [neg_inf for _ in info]
            if masked_any:
                lims = [
                    seq_c - QL + qoff - b * BLK - h * 8
                    for seq_c, (_, _, qoff, _) in zip(seqs, info)
                ]
                mxs = frag_max(tiles, qfs, True, lims)
            else:
                mxs = frag_max(tiles, qfs, False, None)
            for mx, seq_c, (cg, crr, qoff, head) in zip(mxs, seqs, info):
                hit = ((hm >> crr) & fx.Int32(1)) != fx.Int32(0)
                ov = row_value(mx, seq_c, qoff, b)
                otq = (req0 + crr) * QL + qoff
                put(
                    cg * 16 + col,
                    jpos,
                    ov,
                    hit & (h == 0) & (qoff < QL),
                    head * i32_s_sh + otq * i32_s_st + b,
                )

        def score_groups(tiles, hm, mmask, b, jpos):
            """Every column group holding the page (requests hm)."""
            left_m = hm
            while left_m != fx.Int32(0):
                lead_req = fx.Int32(fx.math.cttz(left_m))
                if const_expr(CPR == 1):
                    cg = lead_req // RPC
                    gbits = fx.Int32(GMASK) << (cg * RPC)
                    left_m = left_m & ~gbits
                    score_cgs(tiles, hm, (mmask & gbits) != fx.Int32(0), [cg], b, jpos)
                else:
                    gbits = fx.Int32(1) << lead_req
                    left_m = left_m & ~gbits
                    masked_any = (mmask & gbits) != fx.Int32(0)
                    score_cgs(
                        tiles,
                        hm,
                        masked_any,
                        [lead_req * CPR + gq for gq in range(CPR)],
                        b,
                        jpos,
                    )

        def ballot(c):
            return fx.Int32(fx.rocdl.ballot(T.i32, c))

        # The waves share positions in units of i32_unit blocks; page k of
        # position `it` goes to wave (k + it) % NW. In per-wave mode (fewer
        # requests than waves) wave w walks its own units instead.
        u_step = per_wave.select(i32_splits * NW, i32_splits)
        ub = per_wave.select(split * NW + wave, split) * usz
        it = fx.Int32(0)
        while ballot(ub < nblk_r) != fx.Int32(0):
            b = ub
            jpos = fx.Int32(0)
            while (jpos < usz) & (ballot(b < nblk_r) != fx.Int32(0)):
                in_r = b < nblk_r
                skip = (b < rbf_r) & ((b < i32_init) | (b >= loc_last_r))
                need_r = in_r & ~skip
                bid = fx.Int32(
                    buffer_ops.buffer_load(
                        btr,
                        need_r.select((req0 + lane) * i32_bt_stride + b, fx.Int32(0)),
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                )
                bid = need_r.select(bid, fx.Int32(-1))
                # Sentinel-only blocks, stored by wave 0.
                if (ballot(in_r & skip) != fx.Int32(0)) & ((wave == 0) | per_wave):
                    # A runtime loop: unrolled, all store addresses stay live.
                    jr = fx.Int32(0)
                    while jr < fx.Int32(QL * H):
                        sv = row_value(neg_inf, seq_r, jr // H, b)
                        stq = (req0 + lane) * QL + jr // H
                        put(
                            lane * R + jr,
                            jpos,
                            sv,
                            in_r & skip,
                            (jr % H) * i32_s_sh + stq * i32_s_st + b,
                        )
                        jr = jr + 1
                pend = ballot(need_r)
                mmask = ballot(need_r & (b * BLK + (BLK - 1) > fm_r))
                # This wave's pages, as the mask of their first lanes.
                mine = fx.Int32(0)
                k = it
                while pend != fx.Int32(0):
                    lead = fx.Int32(fx.math.cttz(pend))
                    page = fx.Int32(fx.rocdl.readlane(T.i32, bid, lead))
                    pend = pend & ~ballot(bid == page)
                    mine = mine | (((k % NW) == wave) | per_wave).select(
                        fx.Int32(1) << lead, fx.Int32(0)
                    )
                    k = k + 1
                ok0 = mine != fx.Int32(0)
                l0 = fx.Int32(fx.math.cttz(mine))
                kr0 = page_rsrc(
                    fx.Int32(fx.rocdl.readlane(T.i32, bid, ok0.select(l0, fx.Int32(0))))
                )
                tiles = [load_tile(kr0, ok0, t) for t in range_constexpr(MT)]
                while mine != fx.Int32(0):
                    lead = fx.Int32(fx.math.cttz(mine))
                    page = fx.Int32(fx.rocdl.readlane(T.i32, bid, lead))
                    hm = ballot(bid == page)
                    mine = mine & ~(fx.Int32(1) << lead)
                    okn = mine != fx.Int32(0)
                    ln = fx.Int32(fx.math.cttz(mine))
                    krn = page_rsrc(
                        fx.Int32(
                            fx.rocdl.readlane(T.i32, bid, okn.select(ln, fx.Int32(0)))
                        )
                    )
                    # The next page's codes load while this page is scored.
                    nxt = [load_tile(krn, okn, t) for t in range_constexpr(MT)]
                    fx.rocdl.sched_barrier(0)
                    score_groups(tiles, hm, mmask, b, jpos)
                    tiles = nxt
                b = b + 1
                jpos = jpos + 1
                it = it + 1
            if const_expr(buffered):  # noqa: SIM102 (the outer if is static)
                if to_lds:
                    fx.gpu.barrier()
                    flush(ub)
                    fx.gpu.barrier()
            ub = ub + u_step * usz

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_lut: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        i64_page_stride: fx.Int64,
        i32_q_st: fx.Int32,
        i32_q_sh: fx.Int32,
        i64_q_bytes: fx.Int64,
        i32_s_sh: fx.Int32,
        i32_s_st: fx.Int32,
        i64_s_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_chunk: fx.Int32,
        i32_nch: fx.Int32,
        i32_splits: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_per_wave: fx.Int32,
        i32_unit: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_kv,
            arg_lut,
            arg_score,
            arg_bt,
            arg_seq,
            i64_page_stride,
            i32_q_st,
            i32_q_sh,
            i64_q_bytes,
            i32_s_sh,
            i32_s_st,
            i64_s_bytes,
            i32_bt_stride,
            i64_bt_bytes,
            i32_num_reqs,
            i32_chunk,
            i32_nch,
            i32_splits,
            i32_init,
            i32_local,
            i32_per_wave,
            i32_unit,
        ).launch(
            grid=(fx.Int64(i32_nch * i32_splits), 1, 1),
            block=(NTH, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch
