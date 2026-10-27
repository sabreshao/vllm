# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1250 (wave32) port of ``kernel.py``: the same pages and index-q, bit for
bit. A wave holds one token (gfx950's half-wave): [q heads | k heads | index-k |
index-q heads | v heads x tp] waves per tp tokens, 4 dims of the token per lane.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops import flydsl_gfx1250 as g
from vllm.models.minimax_m3.amd.ops.flydsl_utils import PRELOAD_HINTS, as_f32, pk_bf16
from vllm.models.minimax_m3.amd.ops.norm_rope_nvfp4_insert.kernel import (
    BLK,
    F448,
    F448N,
    NW,
    RCP6,
    D,
    _bf16x4,
    _gptr,
    _k_scale_off,
    _maxnum,
    _minnum,
    _norm_rope_out4,
    _xmax,
)

WAVE = g.WAVE


def _quant4(xs, gs, inv_gs):
    """E2M1 codes of this lane's 4 values (16 bits) and the e4m3 scale byte of
    their 16-value group (lanes l ^ 0..3)."""
    a = [fx.math.absf(x) for x in xs]
    m = _maxnum(_maxnum(a[0], a[1]), _maxnum(a[2], a[3]))
    m = _xmax(_xmax(m, 1), 2)
    sf = _minnum((m * as_f32(RCP6)) * inv_gs, as_f32(F448))
    byte = g.e4m3_byte(sf)
    sv = g.e4m3_value(byte)
    quant = (sv != 0.0).select(fx.Float32(1.0) / (gs * sv), fx.Float32(0.0))
    zero = fx.Float32(0.0)
    codes = g.e2m1x8([x * quant for x in xs] + [zero] * 4)
    return codes & fx.Int32(0xFFFF), byte


def _v_row(state, row, r, n_run, p0, vs, inv_vs):
    """Adds row ``r`` of a V run at octet positions p0.. to ``state``: new
    nibbles and scale bytes, with their masks."""
    words, written = state[0:4], state[4]
    sc, sc_mask = state[5:7], state[7:9]
    nib, sf = _quant4(_bf16x4(row), vs, inv_vs)
    in_run = n_run > r
    p = p0 + r
    sh = p * 4
    words = [
        wd | in_run.select(((nib >> (4 * i)) & 15) << sh, fx.Int32(0))
        for i, wd in enumerate(words)
    ]
    written = written | in_run.select(fx.Int32(15) << sh, fx.Int32(0))
    bsh = (p % 4) * 8
    for k in range_constexpr(2):
        here = in_run & ((p // 4) == k)
        sc[k] = sc[k] | here.select(sf << bsh, fx.Int32(0))
        sc_mask[k] = sc_mask[k] | here.select(fx.Int32(255) << bsh, fx.Int32(0))
    return [*words, written, *sc, *sc_mask]


def compile_norm_rope_insert(nq: int, nkv: int, niq: int, rotary_dim: int, tp: int):
    """Build the kernel; ``tp`` tokens per norm / RoPE wave, one per V wave."""
    assert rotary_dim in (8, 16, 32, 64, 128)
    NR = nq + nkv + 1 + niq  # norm + RoPE waves
    S = NR + nkv * tp  # waves per tp tokens
    ROW = (nq + 2 * nkv + niq + 1) * D
    HALF = rotary_dim // 2
    SIDE = nkv * BLK * 72  # bytes of one side of a main page
    name = (
        f"m3_norm_rope_nvfp4_insert_gfx1250_q{nq}_kv{nkv}_iq{niq}_r{rotary_dim}_t{tp}"
    )

    @flyc.kernel(name=name, known_block_size=[WAVE * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        arg_qkv: fx.Int64,  # [n, ROW] bf16
        arg_pos: fx.Int64,  # positions [n] i64
        arg_slot: fx.Int64,  # main slots [n] i64, -1 skips
        arg_islot: fx.Int64,  # index slots [n] i64, -1 skips
        arg_cs: fx.Int64,  # cos_sin_cache [i32_cs_rows, rotary_dim] bf16
        arg_qw: fx.Int64,  # norm weights [128] bf16: q, k, index-q, index-k
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_q: fx.Int64,  # q_out [n, nq * 128] bf16
        arg_iq: fx.Int64,  # index_q_out [n, niq * 128] e4m3, row stride i32_iq_st
        arg_kv: fx.Int64,  # [blocks, 2 * nkv, 128, 72] u8
        arg_idx: fx.Int64,  # [blocks, 128, 72] u8
        arg_ks: fx.Int64,  # fp32 global scales
        arg_vs: fx.Int64,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_iq_st: fx.Int32,
        i32_eps: fx.Int32,  # fp32 bits
    ):
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        hl = tx % WAVE
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // WAVE))
        wid = bx * NW + wave
        s = wid % S
        hi16 = (hl & 16) != 0
        dim0 = hl * 4
        toks = [(wid // S) * tp + t for t in range_constexpr(tp)]
        lives = [tk < i32_n for tk in toks]

        qkv = buffer_ops.create_buffer_resource_from_addr(
            arg_qkv, num_records_bytes=fx.Int64(i32_n) * (ROW * 2)
        )
        sl_r = buffer_ops.create_buffer_resource_from_addr(
            arg_slot, num_records_bytes=fx.Int64(i32_n) * 8
        )

        if s < NR:
            is_q = s < nq
            is_k = (s >= nq) & (s < nq + nkv)
            is_ik = s == nq + nkv
            is_kq = is_k | is_ik
            # This wave's head column in the qkv row.
            col = (s <= nq + nkv).select(
                is_ik.select(fx.Int32(nq + 2 * nkv + niq), s),
                s - (nq + nkv + 1) + (nq + 2 * nkv),
            )
            w_addr = is_q.select(
                arg_qw, is_k.select(arg_kw, is_ik.select(arg_ikw, arg_iqw))
            )
            wr = buffer_ops.create_buffer_resource_from_addr(
                w_addr, num_records_bytes=D * 2
            )
            pos_r = buffer_ops.create_buffer_resource_from_addr(
                arg_pos, num_records_bytes=fx.Int64(i32_n) * 8
            )
            isl_r = buffer_ops.create_buffer_resource_from_addr(
                arg_islot, num_records_bytes=fx.Int64(i32_n) * 8
            )
            cs_r = buffer_ops.create_buffer_resource_from_addr(
                arg_cs, num_records_bytes=fx.Int64(i32_cs_rows) * (rotary_dim * 2)
            )
            xws, poss, slots = [], [], []
            for t in range_constexpr(tp):
                xws.append(
                    Vec(
                        buffer_ops.buffer_load(
                            qkv,
                            toks[t] * (ROW // 2) + col * (D // 2) + hl * 2,
                            vec_width=2,
                            dtype=fx.Int32,
                            mask=lives[t],
                        )
                    )
                )
                poss.append(
                    fx.Int32(
                        buffer_ops.buffer_load(
                            pos_r,
                            toks[t] * 2,
                            vec_width=1,
                            dtype=fx.Int32,
                            mask=lives[t],
                        )
                    )
                )
                kslot = fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r,
                        toks[t] * 2,
                        vec_width=1,
                        dtype=fx.Int32,
                        mask=lives[t] & is_k,
                    )
                )
                islot = fx.Int32(
                    buffer_ops.buffer_load(
                        isl_r,
                        toks[t] * 2,
                        vec_width=1,
                        dtype=fx.Int32,
                        mask=lives[t] & is_ik,
                    )
                )
                slots.append(is_k.select(kslot, islot))
            w = _bf16x4(
                Vec(buffer_ops.buffer_load(wr, hl * 2, vec_width=2, dtype=fx.Int32))
            )
            first = dim0 < HALF
            in_rope = dim0 < rotary_dim
            ib = first.select(dim0, dim0 - HALF)
            css = []
            for t in range_constexpr(tp):
                cs_off = (poss[t] * rotary_dim + ib) // 2
                css.append(
                    tuple(
                        _bf16x4(
                            Vec(
                                buffer_ops.buffer_load(
                                    cs_r,
                                    cs_off + half,
                                    vec_width=2,
                                    dtype=fx.Int32,
                                    mask=lives[t] & in_rope,
                                )
                            )
                        )
                        for half in (0, HALF // 2)
                    )
                )
            eps = i32_eps.bitcast(fx.Float32)
            qvs = []
            for t in range_constexpr(tp):
                out = _norm_rope_out4(
                    xws[t], css[t], w, eps, hi16, first, in_rope, HALF // 4
                )
                lo, hi = pk_bf16(out[0], out[1]), pk_bf16(out[2], out[3])
                qvs.append(Vec.from_elements([lo, hi], fx.Int32))
            if is_q:
                q_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_q, num_records_bytes=fx.Int64(i32_n) * (nq * D * 2)
                )
                for t in range_constexpr(tp):
                    buffer_ops.buffer_store(
                        qvs[t],
                        q_r,
                        toks[t] * (nq * D // 2) + s * (D // 2) + hl * 2,
                        mask=lives[t],
                    )
            if s > nq + nkv:
                iq_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_iq, num_records_bytes=fx.Int64(i32_n) * fx.Int64(i32_iq_st)
                )
                iq_col = (s - (nq + nkv + 1)) * D
                for t in range_constexpr(tp):
                    f = [
                        _minnum(_maxnum(x, as_f32(F448N)), as_f32(F448))
                        for x in _bf16x4(qvs[t])
                    ]
                    word = g.f32x2_fp8(f[2], f[3], g.f32x2_fp8(f[0], f[1]), hi=True)
                    buffer_ops.buffer_store(
                        word,
                        iq_r,
                        toks[t] * i32_iq_st + iq_col + dim0,
                        mask=lives[t],
                        offset_is_bytes=True,
                    )
            if is_kq:
                gs = is_k.select(
                    fx.Float32(_gptr(arg_ks, T.f32, 4)[0]), fx.Float32(1.0)
                )
                inv_gs = fx.Float32(1.0) / gs
                quants = [
                    _quant4(_bf16x4(qvs[t]), gs, inv_gs) for t in range_constexpr(tp)
                ]
                h = s - nq
                grp = hl // 4
                for t in range_constexpr(tp):
                    slot = slots[t]
                    tp_ = slot & (BLK - 1)
                    page = is_k.select(
                        arg_kv + fx.Int64(slot >> 7) * i64_kv_st,
                        arg_idx + fx.Int64(slot >> 7) * i64_idx_st,
                    )
                    d_off = is_k.select(h * (BLK * 64), fx.Int32(0)) + tp_ * 64 + hl * 2
                    s_off = is_k.select(
                        nkv * BLK * 64 + h * (BLK * 8) + _k_scale_off(tp_, grp),
                        BLK * 64 + tp_ * 8 + grp,
                    )
                    d_ptr = _gptr(page + fx.Int64(d_off), T.i16, 2)
                    s_ptr = _gptr(page + fx.Int64(s_off), T.i8, 1)
                    packed = quants[t][0].to(fx.Int16)
                    sf8 = quants[t][1].to(fx.Int8)
                    if lives[t] & (slot >= 0):
                        d_ptr[0] = packed
                        if hl % 4 == 0:
                            s_ptr[0] = sf8
        else:
            vh = (s - NR) // tp
            tok = (wid // S) * tp + (s - NR) % tp
            live = tok < i32_n
            vcol = (nq + nkv + vh) * (D // 2)  # V head column (dwords)
            vs = fx.Float32(_gptr(arg_vs, T.f32, 4)[0])
            near_t = tok - 7 + hl
            has_near = (hl < 16) & (near_t >= 0) & (near_t < i32_n)
            near = has_near.select(
                fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r, near_t * 2, vec_width=1, dtype=fx.Int32, mask=has_near
                    )
                ),
                fx.Int32(-1),
            )
            slot = fx.Int32(
                buffer_ops.buffer_load(
                    sl_r, tok * 2, vec_width=1, dtype=fx.Int32, mask=live
                )
            )
            has_prev = live & (tok > 0)
            prev = has_prev.select(
                fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r, tok * 2 - 2, vec_width=1, dtype=fx.Int32, mask=has_prev
                    )
                ),
                fx.Int32(-1),
            )
            # The first token of a run in one 8-token octet writes the run and
            # keeps the octet's other nibbles and scales. Rows it can cover load
            # before its slot is known.
            rows = [
                Vec(
                    buffer_ops.buffer_load(
                        qkv,
                        (tok + r) * (ROW // 2) + vcol + hl * 2,
                        vec_width=2,
                        dtype=fx.Int32,
                        mask=tok + r < i32_n,
                    )
                )
                for r in range_constexpr(8)
            ]
            base = slot - slot % 8
            in_oct = (near >= base) & (near < base + 8)
            votes = fx.Int32(fx.rocdl.ballot(T.i32, in_oct)) & 0xFFFF
            n_run = fx.Int32(fx.math.ctpop(votes))
            run_first = (prev < 0) | ((prev >> 3) != (slot >> 3))
            tp_ = slot & (BLK - 1)
            v_side = arg_kv + fx.Int64(slot >> 7) * i64_kv_st + SIDE
            v_data = v_side + fx.Int64(vh * (BLK * 64) + (tp_ // 8) * (D * 4))
            v_sf = v_side + fx.Int64(nkv * BLK * 64 + vh * (BLK * 8) + tp_ - tp_ % 8)
            dp = _gptr(v_data + fx.Int64(dim0 * 4), T.i32, 4)
            sp = _gptr(v_sf + fx.Int64((hl // 4) * BLK), T.i32, 4)
            go = live & (slot >= 0) & run_first
            inv_vs = fx.Float32(1.0) / vs
            # Runs of <= 4 tokens (decode) take the short tile.
            for run_rows, take in (
                (rows[:4], go & (n_run <= 4)),
                (rows, go & (n_run > 4)),
            ):
                if take:
                    old = [fx.Int32(dp[i]) for i in range_constexpr(4)]
                    old_sc = [fx.Int32(sp[i]) for i in range_constexpr(2)]
                    st = [fx.Int32(0)] * 9
                    for r in range_constexpr(len(run_rows)):
                        st = _v_row(st, run_rows[r], r, n_run, tp_ % 8, vs, inv_vs)
                    keep = ~st[4]
                    for i in range_constexpr(4):
                        dp[i] = st[i] | (old[i] & keep)
                    sc0 = st[5] | (old_sc[0] & ~st[7])
                    sc1 = st[6] | (old_sc[1] & ~st[8])
                    if hl % 4 == 0:
                        sp[0] = sc0
                        sp[1] = sc1

    @flyc.jit
    def launch(
        arg_qkv: fx.Int64,
        arg_q: fx.Int64,
        arg_iq: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_cs: fx.Int64,
        arg_pos: fx.Int64,
        arg_slot: fx.Int64,
        arg_islot: fx.Int64,
        arg_kv: fx.Int64,
        arg_idx: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_n: fx.Int32,
        i32_iq_st: fx.Int32,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_cs_rows: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        waves = (i32_n + (tp - 1)) // tp * S
        kernel(
            i32_n,
            i32_cs_rows,
            arg_qkv,
            arg_pos,
            arg_slot,
            arg_islot,
            arg_cs,
            arg_qw,
            arg_kw,
            arg_iqw,
            arg_ikw,
            arg_q,
            arg_iq,
            arg_kv,
            arg_idx,
            arg_ks,
            arg_vs,
            i64_kv_st,
            i64_idx_st,
            i32_iq_st,
            i32_eps,
        ).launch(
            grid=(fx.Int64((waves + (NW - 1)) // NW), 1, 1),
            block=(WAVE * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
