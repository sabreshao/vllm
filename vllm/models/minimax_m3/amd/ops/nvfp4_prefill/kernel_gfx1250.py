# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1250 (wave32, WMMA) port of ``kernel.py``: NVFP4 sparse prefill with the
same numerics (K and V the exact e2m1 x e4m3 products in fp16; Q and P fp16;
fp32 scores, softmax and accumulation; bf16 output).

A workgroup takes 2 query tokens x the 16 query heads of one KV head, so the
union of their selected pages is one (token, slot) per lane of a wave32, and the
per-token O state stays at gfx950's 128 VGPRs under 8-value WMMA accumulators.
Lane (n, h) holds the data of gfx950's lanes (n, 2h) and (n, 2h + 1), as in the
decode port.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops import flydsl_gfx1250 as g
from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    exp2,
    global_i32_ptr,
    maxf_nn,
    pk_bf16,
    pk_f16,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_prefill.kernel import (
    BLK,
    HG,
    KPW,
    NDT,
    NKS,
    NW,
    Q_ROW,
    TOPK,
    D,
    _bf16_to_f16,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_sparse_decode.kernel_gfx1250 import (
    R_KS,
    R_V,
    R_VS,
    RAW,
    kv_lane_loader,
)

WAVE = g.WAVE
TOK_PER_WG = 2
MAXU = TOK_PER_WG * TOPK  # union capacity: one lane per (token, slot)
assert MAXU == WAVE


def compile_nvfp4_prefill_fp16(rounds: int = 2):
    """Build the prefill kernel (see ``kernel.compile_nvfp4_prefill_fp16``)."""
    assert rounds in (1, 2)
    tpr = TOK_PER_WG // rounds  # tokens per combine round
    wpe = rounds  # waves per SIMD
    name = f"m3_nvfp4_sparse_prefill_gfx1250_fp16matched_tq{TOK_PER_WG}_ksplit"
    if rounds == 1:
        name += "_r1"
    NTH = WAVE * NW

    @fx.struct
    class Shared:
        # Q^T during the page loop, then the partial O per (wave, token, d-tile,
        # half, lane) in 16 B tiles; (max, sum) per (wave, token, lane).
        o: fx.Array[fx.Int32, NW * tpr * NDT * WAVE * 8, 16]
        ml: fx.Array[fx.Float32, NW * TOK_PER_WG * 2 * WAVE]

    @flyc.kernel(name=name, known_block_size=[NTH, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [T, H, 128] bf16
        arg_cache: fx.Int64,  # [pages, 2 * Hkv, 128, 72] uint8
        arg_topk: fx.Int64,  # [Hkv, T, 16] int32 logical blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_req: fx.Int64,  # [T] int32
        arg_pos: fx.Int64,  # [T] int32
        arg_out: fx.Int64,  # [T, H, 128] bf16
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,  # topk stride over KV heads
        i32_bt_s: fx.Int32,
        i32_page_byte: fx.Int32,
        f32_c: fx.Float32,  # sm_scale * log2(e)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bidx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % WAVE
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // WAVE))
        n, h = lane % 16, lane // 16
        topk = global_i32_ptr(arg_topk)
        bt = global_i32_ptr(arg_bt)
        req_p = global_i32_ptr(arg_req)
        pos_p = global_i32_ptr(arg_pos)
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        c = f32_c * fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
        v_scale = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
        neg_inf = fx.Float32(float("-inf"))
        topk_base = kvh * i32_topk_hs

        lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(NW * tpr * NDT * WAVE * 8, 1),
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2)
        )
        if const_expr(rounds == 2):
            # Q^T in LDS as fp16; thread (token, head, quarter) stages 64 B.
            tq, hq, qq = tx // 64, (tx // 4) % HG, tx % 4
            tok_q = bidx * TOK_PER_WG + tq
            tok_q = (tok_q < i32_T).select(tok_q, fx.Int32(0))
            q_src = (tok_q * i32_H + kvh * HG + hq) * (D // 2) + qq * 16
            for i in range_constexpr(4):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                q4 = buffer_ops.buffer_load(
                    qr, q_src + 4 * i, vec_width=4, dtype=fx.Int32
                )
                r.store(_bf16_to_f16(q4))
                q_tile = (tq * HG + hq) * Q_ROW + qq * 4 + i
                fx.copy(lds_atom, r, fx.slice(lds, (None, q_tile)))

        def q_frags(t):
            """Token t's Q^T, one 8-dword B operand per K step: dims 64 h + 8 ks
            .. and 64 h + 32 + 8 ks .. (16 B tiles 8 h + ks and 8 h + 4 + ks)."""
            if const_expr(rounds == 1):
                return q_regs[t]
            out = []
            for ks in range_constexpr(NKS):
                parts = []
                for j in (8 * h + ks, 8 * h + 4 + ks):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(
                        lds_atom, fx.slice(lds, (None, (t * HG + n) * Q_ROW + j)), r
                    )
                    v4 = Vec(r.load())
                    parts += [v4[e] for e in range_constexpr(4)]
                out.append(Vec.from_elements(parts, fx.Int32))
            return out

        # Per token: the page of its own block and its causal limit.
        diag, lim, q_regs = [], [], []
        for t in range_constexpr(TOK_PER_WG):
            tok = bidx * TOK_PER_WG + t
            ok = tok < i32_T
            tc = ok.select(tok, fx.Int32(0))
            pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(pos_p[tc])))
            req = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(req_p[tc])))
            diag.append(
                fx.Int32(
                    fx.rocdl.readfirstlane(
                        T.i32, fx.Int32(bt[req * i32_bt_s + pos // BLK])
                    )
                )
            )
            lim.append(pos % BLK - w * KPW)
            if const_expr(rounds == 1):
                q_row = (tc * i32_H + kvh * HG + n) * (D // 2) + h * 32
                qd = [
                    _bf16_to_f16(
                        buffer_ops.buffer_load(
                            qr, q_row + j * 4, vec_width=4, dtype=fx.Int32
                        )
                    )
                    for j in range_constexpr(8)
                ]
                q_regs.append(
                    [
                        Vec.from_elements(
                            [qd[ks][e] for e in range_constexpr(4)]
                            + [qd[4 + ks][e] for e in range_constexpr(4)],
                            fx.Int32,
                        )
                        for ks in range_constexpr(NKS)
                    ]
                )

        load_page = kv_lane_loader(arg_cache, i32_page_byte, kvh, i32_hkv, w, n, h)

        # Lane (token lane // 16, slot lane % 16): the page it selected, -1 none.
        lane_tok = bidx * TOK_PER_WG + lane // TOPK
        lane_tok_ok = lane_tok < i32_T
        lane_tok_clamped = lane_tok_ok.select(lane_tok, fx.Int32(0))
        blk = fx.Int32(topk[topk_base + lane_tok_clamped * TOPK + lane % TOPK])
        lane_blk_ok = lane_tok_ok & (blk >= 0)
        lane_page = fx.Int32(
            bt[
                fx.Int32(req_p[lane_tok_clamped]) * i32_bt_s
                + lane_blk_ok.select(blk, fx.Int32(0))
            ]
        )
        lane_page = lane_blk_ok.select(lane_page, fx.Int32(-1))

        def ballot(cond):
            return fx.Int32(fx.rocdl.ballot(T.i32, cond))

        # The union's first page is the lowest valid lane's: load it before the
        # dedup runs.
        valid = ballot(lane_page >= 0)
        first = (valid != fx.Int32(0)).select(
            fx.Int32(fx.math.cttz(valid)), fx.Int32(0)
        )

        def page_at(mask):
            """Page of the lowest lane left in ``mask``, the first page when empty."""
            idx = (mask != fx.Int32(0)).select(fx.Int32(fx.math.cttz(mask)), first)
            pg = fx.Int32(fx.rocdl.readlane(T.i32, lane_page, idx))
            return (pg >= 0).select(pg, fx.Int32(0))

        raw0 = load_page(page_at(valid))
        fx.rocdl.sched_barrier(0)
        # Union: the first lane of each distinct page.
        n_dup = fx.Int32(0)
        for j in range_constexpr(MAXU - 1):
            pj = fx.Int32(fx.rocdl.readlane(T.i32, lane_page, j))
            n_dup = n_dup + ((lane_page == pj) & (lane > j)).select(
                fx.Int32(1), fx.Int32(0)
            )
        umask = ballot((lane_page >= 0) & (n_dup == 0))
        n_u = fx.Int32(fx.math.ctpop(umask))

        def f16x8_dwords(v8):
            b = Vec(v8).bitcast(fx.Int32)
            return [b[e] for e in range_constexpr(4)]

        def dequant(raw):
            """K frags kf[st][ks] and V^T frags vf[dt] of a page in fp16 (exact
            e2m1 x e4m3 products)."""
            kf = []
            for st in range_constexpr(2):
                # Groups 4 h + 2 half + grp of row st.
                ksc = [g.cvt_fp8x2_f16(raw[R_KS + half], st == 1) for half in (0, 1)]
                row = []
                for ks in range_constexpr(NKS):
                    words = []
                    for half in range_constexpr(2):
                        s8 = Vec.filled(8, ksc[half][ks // 2], fx.Float16)
                        k8 = g.cvt_fp4x8_f16(raw[st * 8 + 4 * half + ks]) * s8
                        words += f16x8_dwords(k8)
                    row.append(Vec.from_elements(words, fx.Int32))
                kf.append(row)
            vsc = [
                g.cvt_fp8x8_f16(raw[R_VS + 2 * o], raw[R_VS + 2 * o + 1])
                for o in range_constexpr(2)
            ]
            # Keys 16 h + j: octet o's dword holds keys 8 o .. 8 o + 7 of one dim.
            vf = [
                Vec.from_elements(
                    f16x8_dwords(g.cvt_fp4x8_f16(raw[R_V + dt]) * vsc[0])
                    + f16x8_dwords(g.cvt_fp4x8_f16(raw[R_V + 8 + dt]) * vsc[1]),
                    fx.Int32,
                )
                for dt in range_constexpr(NDT)
            ]
            return kf, vf

        zero8 = Vec.filled(8, 0.0, fx.Float32)
        key_off = [
            16 * h + 8 * (i // 4) + 4 * st + i % 4
            for st in range_constexpr(2)
            for i in range_constexpr(8)
        ]
        p_src = [(j % 8) // 4 * 8 + 4 * (j // 8) + j % 4 for j in range_constexpr(16)]
        if const_expr(rounds == 2):
            fx.gpu.barrier()  # Q^T staged
        # Iteration u dequantizes page u and loads page u + 1.
        init = [umask]
        init += [neg_inf for _ in range_constexpr(TOK_PER_WG)] + [
            fx.Float32(0.0) for _ in range_constexpr(TOK_PER_WG)
        ]
        init += [zero8 for _ in range_constexpr(TOK_PER_WG * NDT)]
        init += raw0
        n_st = 1 + 2 * TOK_PER_WG + TOK_PER_WG * NDT
        results = init
        for u, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index(n_u),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            rem = fx.Int32(st_[0])
            m = [fx.Float32(st_[1 + t]) for t in range_constexpr(TOK_PER_WG)]
            l_run = [
                fx.Float32(st_[1 + TOK_PER_WG + t]) for t in range_constexpr(TOK_PER_WG)
            ]
            o = [
                [
                    Vec(st_[1 + 2 * TOK_PER_WG + t * NDT + dt])
                    for dt in range_constexpr(NDT)
                ]
                for t in range_constexpr(TOK_PER_WG)
            ]
            raw = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            page = page_at(rem)
            rem = rem & (rem - fx.Int32(1))
            raw_next = load_page(page_at(rem))
            kf, vf = dequant(raw)
            hits = ballot(lane_page == page)
            for t in range_constexpr(TOK_PER_WG):
                m_t, l_t, o_t = m[t], l_run[t], o[t]
                if ((hits >> (TOPK * t)) & fx.Int32(0xFFFF)) != fx.Int32(0):
                    qf = q_frags(t)
                    a = []
                    for st in range_constexpr(2):
                        acc = zero8
                        for ks in range_constexpr(NKS):
                            acc = g.wmma_16x16x32(kf[st][ks], qf[ks], acc, True)
                        a += [acc[i] for i in range_constexpr(8)]
                    if page == diag[t]:
                        a = [
                            (key_off[i] <= lim[t]).select(a[i], neg_inf)
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
                    l_t = l_t * alpha + ps
                    pf = Vec.from_elements(
                        [
                            pk_f16(p[p_src[2 * j]], p[p_src[2 * j + 1]])
                            for j in range_constexpr(8)
                        ],
                        fx.Int32,
                    )
                    # alpha is exactly 1 unless some row's max grew.
                    if ballot(m_new != m_t) != fx.Int32(0):
                        av = Vec.filled(8, alpha, fx.Float32)
                        o_t = [x * av for x in o_t]
                    o_t = [
                        g.wmma_16x16x32(vf[dt], pf, o_t[dt], True)
                        for dt in range_constexpr(NDT)
                    ]
                    m_t = m_new
                m[t], l_run[t], o[t] = m_t, l_t, o_t
            state = (
                [rem] + m + l_run + [x for token_out in o for x in token_out] + raw_next
            )
            results = yield [fx.as_ir_value(x) for x in state]

        def o_tile(ww, tt, dt, half):
            return fx.slice(
                lds, (None, (((ww * tpr + tt) * NDT + dt) * 2 + half) * WAVE + lane)
            )

        # Combine the waves' partials, tpr tokens per round; wave w finalizes
        # token w in round w // tpr.
        ml = smem.ml.ptr
        if const_expr(rounds == 2):
            fx.gpu.barrier()
        for t in range_constexpr(TOK_PER_WG):
            ml[((w * TOK_PER_WG + t) * 2) * WAVE + lane] = fx.Float32(results[1 + t])
            ml[((w * TOK_PER_WG + t) * 2 + 1) * WAVE + lane] = fx.Float32(
                results[1 + TOK_PER_WG + t]
            )
        for rnd in range_constexpr(rounds):
            for tt in range_constexpr(tpr):
                for dt in range_constexpr(NDT):
                    ri = 1 + 2 * TOK_PER_WG + (tpr * rnd + tt) * NDT + dt
                    o8 = Vec(results[ri]).bitcast(fx.Int32)
                    for half in range_constexpr(2):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        r.store(
                            Vec.from_elements(
                                [o8[4 * half + e] for e in range_constexpr(4)],
                                fx.Int32,
                            )
                        )
                        fx.copy(lds_atom, r, o_tile(w, tt, dt, half))
            fx.gpu.barrier()
            if (w // tpr == rnd) & (w < TOK_PER_WG):
                t = w
                ms = [
                    fx.Float32(ml[((ww * TOK_PER_WG + t) * 2) * WAVE + lane])
                    for ww in range_constexpr(NW)
                ]
                big = ms[0]
                for ww in range_constexpr(1, NW):
                    big = maxf_nn(big, ms[ww])
                big = (big == neg_inf).select(fx.Float32(0.0), big)
                wt = [
                    fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)
                ]
                l_tot = fx.Float32(0.0)
                for ww in range_constexpr(NW):
                    l_w = ml[((ww * TOK_PER_WG + t) * 2 + 1) * WAVE + lane]
                    l_tot = l_tot + fx.Float32(l_w) * wt[ww]
                l_tot = g.xlane2(l_tot, lambda a, b: a + b)
                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                acc = [zero8 for _ in range_constexpr(NDT)]
                for ww in range_constexpr(NW):
                    wv = Vec.filled(8, wt[ww], fx.Float32)
                    for dt in range_constexpr(NDT):
                        parts = []
                        for half in range_constexpr(2):
                            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                            fx.copy(lds_atom, o_tile(ww, t - tpr * rnd, dt, half), r)
                            v4 = Vec(r.load())
                            parts += [v4[e] for e in range_constexpr(4)]
                        o8 = Vec.from_elements(parts, fx.Int32).bitcast(fx.Float32)
                        acc[dt] = acc[dt] + o8 * wv
                # Lane (n, h) holds dims [64 h, 64 h + 64) of head n: element i of
                # d-tile dt is dim 64 h + 8 i + dt.
                tok = bidx * TOK_PER_WG + t
                outr = buffer_ops.create_buffer_resource_from_addr(
                    arg_out,
                    num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2),
                )
                o_row = (tok * i32_H + kvh * HG + n) * (D // 2) + h * 32
                for i in range_constexpr(8):
                    val = Vec.from_elements(
                        [
                            pk_bf16(acc[2 * j][i] * inv, acc[2 * j + 1][i] * inv)
                            for j in range_constexpr(4)
                        ],
                        fx.Int32,
                    )
                    buffer_ops.buffer_store(val, outr, o_row + i * 4, mask=tok < i32_T)
            fx.gpu.barrier()

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_req: fx.Int64,
        arg_pos: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_cache,
            arg_topk,
            arg_bt,
            arg_req,
            arg_pos,
            arg_out,
            arg_ks,
            arg_vs,
            i32_T,
            i32_H,
            i32_hkv,
            i32_topk_hs,
            i32_bt_s,
            i32_page_b,
            f32_c,
            value_attrs={
                "rocdl.waves_per_eu": wpe,
                "rocdl.flat_work_group_size": f"{NTH},{NTH}",
            },
        ).launch(
            grid=((i32_T + TOK_PER_WG - 1) // TOK_PER_WG, i32_hkv, 1),
            block=(NTH, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch
