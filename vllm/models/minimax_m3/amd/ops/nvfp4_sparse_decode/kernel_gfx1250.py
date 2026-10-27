# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1250 (wave32, WMMA) port of ``kernel.py``: the same NVFP4 pages and
numerics. K and V dequantize to e4m3_rn_satfinite(e2m1 x block scale), widened
to bf16; scores and softmax are fp32, P is bf16.

Wave32 lane (n, h) takes over the data of gfx950 lanes (n, 2h) and (n, 2h + 1).
S^T tile st row r is still key 32 w + 8 (r // 4) + 4 st + r % 4, so a lane's 16
accumulators are keys 32 w + 16 h + j, which are its P^T and V^T K slots; its
QK^T K slots are dims 64 h + 8 ks .. and 64 h + 32 + 8 ks ...
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops import flydsl_gfx1250 as g
from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    PRELOAD_HINTS,
    as_f32,
    bits,
    exp2,
    global_i32_ptr,
    maxf_nn,
    perm,
    pk_bf16,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_prefill.kernel import (
    BLK,
    DATA_B,
    HG,
    KPW,
    NDT,
    NKS,
    NW,
    SCALE_B,
    D,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_sparse_decode.kernel import (
    MAG4,
    SIGN4,
    _global_atomic_add,
)

WAVE = g.WAVE
# NVFP4 dwords a lane loads per page: K 2 rows x 8, 2 K-scale dwords, V 2 octets
# x 8, 4 V-scale dwords.
RAW = 16 + 2 + 16 + 4
R_KS, R_V, R_VS = 16, 18, 34
PART_F = NDT * 8 + 4  # fp32 per (split, row, lane) partial: O, max, sum, pad


def _nnan():
    return _ir.Attribute.parse("#arith.fastmath<nnan>")


def kv_lane_loader(arg_cache, page_bytes, kvh, hkv, w, n, h):
    """The loader of this lane's RAW NVFP4 dwords of a page (lane (n, h))."""
    side_b = hkv * (DATA_B + SCALE_B)
    k_data = kvh * DATA_B
    k_scl = hkv * DATA_B + kvh * SCALE_B
    v_data = side_b + kvh * DATA_B
    v_scl = side_b + hkv * DATA_B + kvh * SCALE_B
    k_rows = [w * KPW + 8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]
    k_load = [(k_data + r * (D // 2) + h * 32) // 4 for r in k_rows]
    k_scale_load = (k_scl + ((w * 4 + 2 * h) * 16 + n) * 4) // 4
    v_load = (v_data + ((w * KPW + 16 * h) // 8 * 16 + n) * 32) // 4
    v_scale_load = (v_scl + (n // 2) * BLK + w * KPW + 16 * h) // 4

    def load(page):
        pr = buffer_ops.create_buffer_resource_from_addr(
            arg_cache + fx.Int64(page) * fx.Int64(page_bytes),
            num_records_bytes=page_bytes,
        )

        def b128(off):
            v = Vec(buffer_ops.buffer_load(pr, off, vec_width=4, dtype=fx.Int32))
            return [v[i] for i in range_constexpr(4)]

        raw = []
        for st in range_constexpr(2):
            raw += b128(k_load[st]) + b128(k_load[st] + 4)
        for gg in range_constexpr(2):
            raw.append(
                fx.Int32(
                    buffer_ops.buffer_load(
                        pr, k_scale_load + gg * 16, vec_width=1, dtype=fx.Int32
                    )
                )
            )
        for o in range_constexpr(2):  # token octets 4 w + 2 h + o, 512 B apart
            raw += b128(v_load + o * 128) + b128(v_load + o * 128 + 4)
        raw += b128(v_scale_load)
        return raw

    return load


def _k_bf16x8(codes, tab_lo, tab_hi):
    """Eight K codes under one block scale -> four dwords of bf16 pairs."""
    even = perm(tab_hi, tab_lo, codes & MAG4) ^ ((codes << 4) & SIGN4)
    odd = perm(tab_hi, tab_lo, (codes >> 4) & MAG4) ^ (codes & SIGN4)
    lo = perm(odd, even, 0x05010400)
    hi = perm(odd, even, 0x07030602)
    bf = g.cvt_fp8x8_bf16(lo, hi)
    return [bf[i] for i in range_constexpr(4)]


def _v_bf16x8(codes, scales):
    """One V dword (eight keys of one dim) and its 8 fp16 block scales -> four
    dwords of bf16 pairs. The fp16 products are exact; they are clamped to
    +-448 and rounded once to e4m3."""
    v = g.cvt_fp4x8_f16(codes) * scales
    big = Vec.filled(8, 448.0, fx.Float16)
    small = Vec.filled(8, -448.0, fx.Float16)
    v = _arith.MinNumFOp(
        fx.as_ir_value(v), fx.as_ir_value(big), fastmath=_nnan()
    ).result
    v = _arith.MaxNumFOp(v, fx.as_ir_value(small), fastmath=_nnan()).result
    f8 = g.f16x8_fp8(Vec(v))
    bf = g.cvt_fp8x8_bf16(f8[0], f8[1])
    return [bf[i] for i in range_constexpr(4)]


def compile_nvfp4_sparse_decode(splits: int = 1, topk: int = 16, xcds: int = 0):
    """Build the decode kernel (see ``kernel.compile_nvfp4_sparse_decode``)."""
    assert 1 <= topk <= WAVE and splits >= 1
    SPLITS, TOPK, XCDS = splits, topk, xcds
    # Split partials are device-coherent unless all splits of a row share an XCD.
    PART_SCOPE = 0 if XCDS else g.SCOPE_DEV
    name = f"m3_nvfp4_sparse_decode_gfx1250_k{TOPK}_s{SPLITS}" + (
        f"_x{XCDS}" if XCDS else ""
    )

    @fx.struct
    class Shared:
        o: fx.Array[fx.Int32, NW * NDT * WAVE * 8, 16]
        ml: fx.Array[fx.Float32, NW * 2 * WAVE]
        lut: fx.Array[fx.Int32, 512]
        ticket: fx.Array[fx.Int32, 4]

    @flyc.kernel(name=name, known_block_size=[WAVE * NW, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_ql: fx.Int32,  # rows per request
        i32_topk_hs: fx.Int32,  # topk stride over KV heads
        i32_bt_s: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        arg_seq: fx.Int64,  # [reqs] int32
        arg_topk: fx.Int64,  # [Hkv, >= rows, TOPK] int32 blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_lut: fx.Int64,  # [256] int64 K magnitude tables
        arg_q: fx.Int64,  # [rows, H, 128] bf16
        arg_cache: fx.Int64,  # [pages, 2 * Hkv, 128, 72] uint8
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        arg_out: fx.Int64,  # [rows, H, 128] bf16
        arg_part: fx.Int64,  # [grid rows * Hkv, S, 32, PART_F] fp32
        arg_cnt: fx.Int64,  # [grid rows * Hkv] int32 tickets, zero
        i64_part_bytes: fx.Int64,
        i32_page_byte: fx.Int32,
        f32_c: fx.Float32,  # sm_scale * log2(e)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % WAVE
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // WAVE))
        n, h = lane % 16, lane // 16
        topk = global_i32_ptr(arg_topk)
        bt = global_i32_ptr(arg_bt)
        seq_p = global_i32_ptr(arg_seq)
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        c = f32_c * fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
        v_scale = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
        neg_inf = fx.Float32(float("-inf"))
        lut_g = global_i32_ptr(arg_lut)
        for e in range_constexpr(512 // (WAVE * NW)):
            smem.lut.ptr[tx + e * WAVE * NW] = fx.Int32(lut_g[tx + e * WAVE * NW])

        if const_expr(XCDS > 0):
            xcd = bx % XCDS
            slot = bx // XCDS
            row = ((slot // i32_ql) * XCDS + xcd) * i32_ql + slot % i32_ql
        else:
            row = bx
        row_ok = row < i32_rows
        rowc = row_ok.select(row, fx.Int32(0))
        req = rowc // i32_ql
        pos = fx.Int32(seq_p[req]) - i32_ql + rowc % i32_ql
        # Padding rows (seq_len 0 or past i32_rows) see nothing.
        pos = (row_ok & (pos >= 0)).select(pos, fx.Int32(-1))
        pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, pos))
        req = fx.Int32(fx.rocdl.readfirstlane(T.i32, req))
        self_blk = (pos >= 0).select(pos // BLK, fx.Int32(-1))

        # Lane = top-k slot.
        slot_ok = lane < TOPK
        blk = fx.Int32(
            topk[kvh * i32_topk_hs + rowc * TOPK + slot_ok.select(lane, fx.Int32(0))]
        )
        ok = slot_ok & (blk >= 0) & (blk <= self_blk)
        page = fx.Int32(bt[req * i32_bt_s + ok.select(blk, fx.Int32(0))])
        page = ok.select(page, fx.Int32(-1))
        blk = ok.select(blk, fx.Int32(-1))
        vmask = fx.Int32(fx.rocdl.ballot(T.i32, ok))
        n_v = fx.Int32(fx.math.ctpop(vmask))
        if const_expr(SPLITS > 1):
            below = (fx.Int32(1) << lane) - fx.Int32(1)
            rank = fx.Int32(fx.math.ctpop(vmask & below))
            chunk = (n_v + (SPLITS - 1)) // SPLITS
            lo = fx.block_idx.z * chunk
            vmask = fx.Int32(
                fx.rocdl.ballot(T.i32, ok & (rank >= lo) & (rank < lo + chunk))
            )
            n_v = fx.Int32(fx.math.ctpop(vmask))

        lim = pos % BLK - w * KPW
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )
        # Dims 64 h .. 64 h + 63 of head n: qd[j] holds dims 64 h + 8 j ...
        q_row = (rowc * i32_H + kvh * HG + n) * (D // 2) + h * 32
        qd = [
            Vec(buffer_ops.buffer_load(qr, q_row + j * 4, vec_width=4, dtype=fx.Int32))
            for j in range_constexpr(8)
        ]
        qf = [
            Vec.from_elements(
                [qd[ks][i] for i in range_constexpr(4)]
                + [qd[4 + ks][i] for i in range_constexpr(4)],
                fx.Int32,
            )
            for ks in range_constexpr(NKS)
        ]

        load_block = kv_lane_loader(arg_cache, i32_page_byte, kvh, i32_hkv, w, n, h)

        def dequant_k(raw):
            kf = []
            for st in range_constexpr(2):
                # tabs[half][grp]: groups 4 h + 2 half + grp of row st.
                tabs = []
                for half in range_constexpr(2):
                    th = []
                    for grp in range_constexpr(2):
                        sb = raw[R_KS + half].shrui(fx.Int32(8 * (2 * st + grp)))
                        sb = sb & fx.Int32(0xFF)
                        th.append(
                            (
                                fx.Int32(smem.lut.ptr[sb * 2]),
                                fx.Int32(smem.lut.ptr[sb * 2 + 1]),
                            )
                        )
                    tabs.append(th)
                kf.append(
                    [
                        Vec.from_elements(
                            _k_bf16x8(raw[st * 8 + ks], *tabs[0][ks // 2])
                            + _k_bf16x8(raw[st * 8 + 4 + ks], *tabs[1][ks // 2]),
                            fx.Int32,
                        )
                        for ks in range_constexpr(NKS)
                    ]
                )
            return kf

        def dequant_v(raw):
            sc = [
                g.cvt_fp8x8_f16(raw[R_VS + 2 * o], raw[R_VS + 2 * o + 1])
                for o in range_constexpr(2)
            ]
            return [
                Vec.from_elements(
                    _v_bf16x8(raw[R_V + dt], sc[0])
                    + _v_bf16x8(raw[R_V + 8 + dt], sc[1]),
                    fx.Int32,
                )
                for dt in range_constexpr(NDT)
            ]

        fx.gpu.barrier()  # LUT in LDS

        zero8 = Vec.filled(8, 0.0, fx.Float32)
        # Score index st * 8 + i is key 16 h + 8 (i // 4) + 4 st + i % 4.
        key_off = [
            16 * h + 8 * (i // 4) + 4 * st + i % 4
            for st in range_constexpr(2)
            for i in range_constexpr(8)
        ]
        # P^T K slot j is key 16 h + j.
        p_src = [(j % 8) // 4 * 8 + 4 * (j // 8) + j % 4 for j in range_constexpr(16)]
        first = (n_v > 0).select(fx.Int32(fx.math.cttz(vmask)), fx.Int32(0))

        def slot_at(mask):
            """The lowest slot left in ``mask``, the first one when empty."""
            return (mask != fx.Int32(0)).select(fx.Int32(fx.math.cttz(mask)), first)

        def page_of(slot):
            pg = fx.Int32(fx.rocdl.readlane(T.i32, page, slot))
            return (pg >= 0).select(pg, fx.Int32(0))

        def attend(kf, vf, cur_blk, live, m_t, l_t, o_t):
            """Fold one block into (max, sum, O); a dead block is fully masked."""
            a = []
            for st in range_constexpr(2):
                acc = zero8
                for ks in range_constexpr(NKS):
                    acc = g.wmma_16x16x32(kf[st][ks], qf[ks], acc, False)
                a += [acc[i] for i in range_constexpr(8)]
            diag = cur_blk == self_blk
            dead = live == fx.Int32(0)
            a = [
                (dead | (diag & (key_off[i] > lim))).select(neg_inf, a[i])
                for i in range_constexpr(16)
            ]
            mx = a[0]
            for i in range_constexpr(1, 16):
                mx = maxf_nn(mx, a[i])
            # The running max is kept in scaled units (c > 0).
            m_new = maxf_nn(m_t, g.xlane2(mx, maxf_nn) * c)
            m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
            alpha = exp2(m_t - m_use)
            p = [exp2(fx.math.fma(x, c, -m_use)) for x in a]
            ps = p[0]
            for i in range_constexpr(1, 16):
                ps = ps + p[i]
            pf = Vec.from_elements(
                [
                    pk_bf16(p[p_src[2 * j]], p[p_src[2 * j + 1]])
                    for j in range_constexpr(8)
                ],
                fx.Int32,
            )
            av = Vec.filled(8, alpha, fx.Float32)
            o_t = [
                g.wmma_16x16x32(vf[dt], pf, o_t[dt] * av, False)
                for dt in range_constexpr(NDT)
            ]
            return m_new, l_t * alpha + ps, o_t

        # Two blocks per iteration, the next pair's loads in flight; an odd last
        # block is masked.
        init = [vmask, neg_inf, fx.Float32(0.0)]
        init += [zero8 for _ in range_constexpr(NDT)]
        init += load_block(page_of(slot_at(vmask)))
        n_st = 3 + NDT
        results = init
        for u, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index((n_v + 1) // 2),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            rem = fx.Int32(st_[0])
            m_t = fx.Float32(st_[1])
            l_t = fx.Float32(st_[2])
            o_t = [Vec(st_[3 + dt]) for dt in range_constexpr(NDT)]
            raw_a = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            blk_a = fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem)))
            rem_b = rem & (rem - fx.Int32(1))
            raw_b = load_block(page_of(slot_at(rem_b)))
            fx.rocdl.sched_barrier(0)
            kf_a = dequant_k(raw_a)
            vf_a = dequant_v(raw_a)
            rem_n = rem_b & (rem_b - fx.Int32(1))
            fx.rocdl.sched_barrier(0)
            raw_n = load_block(page_of(slot_at(rem_n)))
            fx.rocdl.sched_barrier(0)
            res = attend(kf_a, vf_a, blk_a, fx.Int32(1), m_t, l_t, o_t)
            m_t, l_t, o_t = res[0], res[1], res[2]
            blk_b = fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem_b)))
            live_b = (rem_b != fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
            kf_b = dequant_k(raw_b)
            vf_b = dequant_v(raw_b)
            res = attend(kf_b, vf_b, blk_b, live_b, m_t, l_t, o_t)
            m_t, l_t, o_t = res[0], res[1], res[2]
            state = [rem_n, m_t, l_t] + o_t + raw_n
            results = yield [fx.as_ir_value(x) for x in state]

        def store_out(acc, inv):
            """O^T tile dt element i of lane (n, h) is dim 64 h + 8 i + dt."""
            outr = buffer_ops.create_buffer_resource_from_addr(
                arg_out,
                num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2),
            )
            o_row = (row * i32_H + kvh * HG + n) * (D // 2) + h * 32
            for i in range_constexpr(8):
                val = Vec.from_elements(
                    [
                        pk_bf16(acc[2 * j][i] * inv, acc[2 * j + 1][i] * inv)
                        for j in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                buffer_ops.buffer_store(val, outr, o_row + i * 4, mask=row < i32_rows)

        # Combine the waves' partials in wave 0.
        o_lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(NW * NDT * WAVE * 8, 1),
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

        def o_chunk(ww, dt, half):
            return fx.slice(o_lds, (None, ((ww * NDT + dt) * 2 + half) * WAVE + lane))

        ml = smem.ml.ptr
        ml[(w * 2) * WAVE + lane] = fx.Float32(results[1])
        ml[(w * 2 + 1) * WAVE + lane] = fx.Float32(results[2])
        for dt in range_constexpr(NDT):
            o8 = Vec(results[3 + dt]).bitcast(fx.Int32)
            for half in range_constexpr(2):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                r.store(
                    Vec.from_elements(
                        [o8[4 * half + i] for i in range_constexpr(4)], fx.Int32
                    )
                )
                fx.copy(lds_atom, r, o_chunk(w, dt, half))
        fx.gpu.barrier()
        if w == 0:
            ms = [fx.Float32(ml[(ww * 2) * WAVE + lane]) for ww in range_constexpr(NW)]
            big = ms[0]
            for ww in range_constexpr(1, NW):
                big = maxf_nn(big, ms[ww])
            big_raw = big
            big = (big == neg_inf).select(fx.Float32(0.0), big)
            wt = [fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)]
            l_tot = fx.Float32(0.0)
            for ww in range_constexpr(NW):
                l_tot = l_tot + fx.Float32(ml[(ww * 2 + 1) * WAVE + lane]) * wt[ww]
            l_tot = g.xlane2(l_tot, lambda x, y: x + y)
            acc = [zero8 for _ in range_constexpr(NDT)]
            for ww in range_constexpr(NW):
                wv = Vec.filled(8, wt[ww], fx.Float32)
                for dt in range_constexpr(NDT):
                    parts = []
                    for half in range_constexpr(2):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        fx.copy(lds_atom, o_chunk(ww, dt, half), r)
                        v4 = Vec(r.load())
                        parts += [v4[i] for i in range_constexpr(4)]
                    o8 = Vec.from_elements(parts, fx.Int32).bitcast(fx.Float32)
                    acc[dt] = acc[dt] + o8 * wv
            if const_expr(SPLITS > 1):
                tid = row * i32_hkv + kvh
                pr = buffer_ops.create_buffer_resource_from_addr(
                    arg_part, num_records_bytes=i64_part_bytes
                )
                row_f = (tid * SPLITS + fx.block_idx.z) * WAVE + lane
                for dt in range_constexpr(NDT):
                    a8 = acc[dt].bitcast(fx.Int32)
                    for half in range_constexpr(2):
                        buffer_ops.buffer_store(
                            Vec.from_elements(
                                [a8[4 * half + i] for i in range_constexpr(4)],
                                fx.Int32,
                            ),
                            pr,
                            row_f * PART_F + dt * 8 + half * 4,
                            cache_modifier=PART_SCOPE,
                        )
                buffer_ops.buffer_store(
                    Vec.from_elements([bits(big_raw), bits(l_tot)], fx.Int32),
                    pr,
                    row_f * PART_F + NDT * 8,
                    cache_modifier=PART_SCOPE,
                )
                # The partial is stored before the ticket is taken; gfx1250
                # counts stores apart from loads.
                g.wait_store()
                if lane == 0:
                    smem.ticket.ptr[0] = _global_atomic_add(
                        arg_cnt + fx.Int64(tid) * 4, 1
                    )
                g.wait_ds()
                if (
                    fx.Int32(
                        fx.rocdl.readfirstlane(T.i32, fx.Int32(smem.ticket.ptr[0]))
                    )
                    == SPLITS - 1
                ):
                    rows_f = [
                        (tid * SPLITS + sp) * WAVE + lane
                        for sp in range_constexpr(SPLITS)
                    ]
                    mls = [
                        Vec(
                            buffer_ops.buffer_load(
                                pr,
                                rf * PART_F + NDT * 8,
                                vec_width=2,
                                dtype=fx.Int32,
                                cache_modifier=PART_SCOPE,
                            )
                        )
                        for rf in rows_f
                    ]
                    bigs = [as_f32(x[0]) for x in mls]
                    top = bigs[0]
                    for sp in range_constexpr(1, SPLITS):
                        top = maxf_nn(top, bigs[sp])
                    top = (top == neg_inf).select(fx.Float32(0.0), top)
                    ws = [fx.Float32(fx.math.exp2(b - top)) for b in bigs]
                    l_all = fx.Float32(0.0)
                    for sp in range_constexpr(SPLITS):
                        l_all = l_all + as_f32(mls[sp][1]) * ws[sp]
                    inv_all = (l_all > fx.Float32(0.0)).select(
                        v_scale / l_all, fx.Float32(0.0)
                    )
                    acc_all = [zero8 for _ in range_constexpr(NDT)]
                    for sp in range_constexpr(SPLITS):
                        wv = Vec.filled(8, ws[sp], fx.Float32)
                        for dt in range_constexpr(NDT):
                            parts = []
                            for half in range_constexpr(2):
                                v4 = Vec(
                                    buffer_ops.buffer_load(
                                        pr,
                                        rows_f[sp] * PART_F + dt * 8 + half * 4,
                                        vec_width=4,
                                        dtype=fx.Int32,
                                        cache_modifier=PART_SCOPE,
                                    )
                                )
                                parts += [v4[i] for i in range_constexpr(4)]
                            part = Vec.from_elements(parts, fx.Int32).bitcast(
                                fx.Float32
                            )
                            acc_all[dt] = acc_all[dt] + part * wv
                    store_out(acc_all, inv_all)
                    if lane == 0:
                        global_i32_ptr(arg_cnt + fx.Int64(tid) * 4)[0] = fx.Int32(0)
            else:
                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                store_out(acc, inv)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_lut: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_rows: fx.Int32,
        i32_ql: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        i32_grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_rows,
            i32_ql,
            i32_topk_hs,
            i32_bt_s,
            i32_H,
            i32_hkv,
            arg_seq,
            arg_topk,
            arg_bt,
            arg_lut,
            arg_q,
            arg_cache,
            arg_ks,
            arg_vs,
            arg_out,
            arg_part,
            arg_cnt,
            i64_part_bytes,
            i32_page_b,
            f32_c,
        ).launch(
            grid=(i32_grid_x, i32_hkv, SPLITS),
            block=(WAVE * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
