# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1250 (wave32) NVFP4 prefill index scorer for MiniMax-M3: the block scores
of ``kernel.py`` over the same e4m3_rn_satfinite(e2m1 x scale) keys.

gfx1250 has no buffer -> LDS DMA and its fp8 WMMA does not accumulate exactly,
so a block's keys are dequantized by the workgroup (thread t: token t) through
the e4m3 table, widened to bf16 (exact) into one of two LDS slots, and scored
with v_wmma_f32_16x16x32_bf16 against Q^T held in VGPRs as bf16. Keys and Q use
the key's dim order (even dims, then odd dims per 8); a K step covers key-order
positions 32 ks .. + 31, lane half h supplying 32 ks + 16 h .. + 15. The next
block's NVFP4 codes load into registers while a block is scored.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops import flydsl_gfx1250 as g
from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    dequant8,
    global_i32_ptr,
    key_order4,
    maxf_nn,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_index_prefill.kernel import (
    BLK,
    GROUP,
    PAGE_BYTES,
    PAGE_CODES,
    D,
    page_lookup,
    score_resources,
    segment_blocks,
    tile_rows,
    with_sentinels,
)

WAVE = g.WAVE
NW = 4
NTH = WAVE * NW
assert NTH == BLK  # one thread per key token
ROWS_PER_WAVE = 64
TILE_Q = NW * ROWS_PER_WAVE
NI = ROWS_PER_WAVE // 16  # N-tiles per wave
MT = BLK // 16  # M-tiles per block
NKS = D // 32  # K steps
KROW_T = D * 2 // 16  # 16 B bf16 chunks per key row in LDS
SLOT_T = BLK * KROW_T

__all__ = ["BLK", "D", "GROUP", "TILE_Q", "compile_score_prefill"]


def compile_score_prefill(heads: int = 1):
    """Build the scorer for 1, 2 or 4 index heads (sharing one index-K head)."""
    assert heads in (1, 2, 4), heads
    H = heads
    name = f"m3_nvfp4_index_score_prefill_gfx1250_tq{TILE_Q}_h{H}"

    @fx.struct
    class Shared:
        k: fx.Array[fx.Int32, 2 * SLOT_T * 4, 16]  # bf16 key slots
        lut: fx.Array[fx.Int32, 512, 16]

    @flyc.kernel(name=name, known_block_size=[NTH, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [total_q, H, 128] e4m3
        arg_kv: fx.Int64,  # [num_blocks, 128, 72] uint8
        arg_lut: fx.Int64,  # nvfp4_e4m3_lut [256] int64
        arg_score: fx.Int64,  # [H, total_q, S] fp32
        arg_bt: fx.Int64,  # [batch, >= max_blocks] int32
        arg_cu: fx.Int64,  # cu_seqlens_q [batch + 1] int32
        arg_seq: fx.Int64,  # seq_lens [batch] int32
        i64_page_stride: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,  # query tiles per request
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_hs: fx.Int32,  # score head stride
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % WAVE
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // WAVE))
        l16, h = lane % 16, lane // 16
        live, b, seg, seq_start, seq_len, prefix, q_rows, row0 = tile_rows(
            pid, i32_qt, i32_nseg, i32_nb, arg_cu, arg_seq, H, TILE_Q
        )
        if live:
            nblk, blk0, blk1 = segment_blocks(
                prefix, row0, seq_len, seg, i32_nseg, H, TILE_Q, GROUP
            )
            if blk0 < blk1:
                qr, sr = score_resources(
                    arg_q, arg_score, i32_total_q, i32_hs, i32_S, H
                )
                page_of = page_lookup(arg_bt, b * i32_bt_stride, nblk)

                lut_g = global_i32_ptr(arg_lut)
                for e in range_constexpr(512 // NTH):
                    smem.lut.ptr[tx + e * NTH] = fx.Int32(lut_g[tx + e * NTH])
                # Every thread's dequant reads table entries other waves wrote.
                fx.gpu.barrier()

                # Rows past q_len * H read the next request's Q and are masked at
                # the store. Q^T per (N-tile, K step) as 8 dwords of bf16.
                rows = [row0 + wave * ROWS_PER_WAVE + ni * 16 + l16 for ni in range(NI)]
                qpos = [prefix + r // H for r in rows]
                qf = []
                for ni in range_constexpr(NI):
                    qrow = (seq_start * H + rows[ni]) * D
                    steps = []
                    for ks in range_constexpr(NKS):
                        # Key-order bytes 32 ks + 16 h .. + 15: the 16 B chunk
                        # 2 ks + h of the row, put in key order.
                        d = Vec(
                            buffer_ops.buffer_load(
                                qr,
                                (qrow + (2 * ks + h) * 16) // 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                        ko = key_order4([fx.Int32(d[i]) for i in range_constexpr(4)])
                        lo = g.cvt_fp8x8_bf16(ko[0], ko[1])
                        hi = g.cvt_fp8x8_bf16(ko[2], ko[3])
                        steps.append(
                            Vec.from_elements(
                                [lo[i] for i in range_constexpr(4)]
                                + [hi[i] for i in range_constexpr(4)],
                                fx.Int32,
                            )
                        )
                    qf.append(steps)

                kv_view = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.k.ptr),
                        fx.make_layout(2 * SLOT_T * 4, 1),
                    ),
                    fx.make_layout(4, 1),
                )
                lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
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

                def k_tile(slot, tok, c):
                    """16 B chunk c of key row tok in a slot (chunks swizzled)."""
                    return slot * SLOT_T + tok * KROW_T + (c ^ (tok & (KROW_T - 1)))

                def load_raw(blk):
                    """Thread tx's token of block blk: 16 code dwords, 2 scale
                    dwords."""
                    kr = buffer_ops.create_buffer_resource_from_addr(
                        arg_kv + fx.Int64(page_of(blk)) * i64_page_stride,
                        num_records_bytes=PAGE_BYTES,
                    )
                    raw = []
                    for i in range_constexpr(4):
                        v = Vec(
                            buffer_ops.buffer_load(
                                kr,
                                tx * 16 + i * 4,
                                vec_width=4,
                                dtype=fx.Int32,
                                cache_modifier=g.TH_NT,
                            )
                        )
                        raw += [fx.Int32(v[e]) for e in range_constexpr(4)]
                    sc = Vec(
                        buffer_ops.buffer_load(
                            kr,
                            (PAGE_CODES + tx * 8) // 4,
                            vec_width=2,
                            dtype=fx.Int32,
                            cache_modifier=g.TH_NT,
                        )
                    )
                    return raw + [fx.Int32(sc[0]), fx.Int32(sc[1])]

                def dequant_to_lds(raw, slot):
                    """Token tx's 128 keys -> bf16 key row in key order."""
                    for c8 in range_constexpr(8):  # 16 key-order positions each
                        sbyte = (raw[16 + c8 // 4] >> (8 * (c8 % 4))) & 0xFF
                        tab = lut_load(sbyte)
                        words = []
                        for e in range_constexpr(2):
                            lo, hi = dequant8(raw[2 * c8 + e], *tab)
                            bf = g.cvt_fp8x8_bf16(lo, hi)
                            words.append([bf[i] for i in range_constexpr(4)])
                        for e in range_constexpr(2):
                            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                            r.store(Vec.from_elements(words[e], fx.Int32))
                            fx.copy(
                                lds_atom,
                                r,
                                fx.slice(kv_view, (None, k_tile(slot, tx, 2 * c8 + e))),
                            )

                zero8 = Vec.filled(8, 0.0, fx.Float32)
                neg_inf = fx.Float32(float("-inf"))

                def score_block(slot, blk):
                    """This wave's max score per (N-tile row) over block blk's
                    causal tokens; lane (l16, h) ends with row ni * 16 + l16."""
                    vis = [qpos[ni] - (blk * BLK + h * 8) for ni in range_constexpr(NI)]
                    run = [neg_inf for _ in range(NI)]
                    for mt in range_constexpr(MT):
                        tok = mt * 16 + l16
                        accs = [zero8 for _ in range(NI)]
                        for ks in range_constexpr(NKS):
                            parts = []
                            for e in range_constexpr(2):
                                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                                fx.copy(
                                    lds_atom,
                                    fx.slice(
                                        kv_view,
                                        (None, k_tile(slot, tok, 4 * ks + 2 * h + e)),
                                    ),
                                    r,
                                )
                                v4 = Vec(r.load())
                                parts += [v4[i] for i in range_constexpr(4)]
                            a = Vec.from_elements(parts, fx.Int32)
                            for ni in range_constexpr(NI):
                                accs[ni] = g.wmma_16x16x32(
                                    a, qf[ni][ks], accs[ni], False
                                )
                        for ni in range_constexpr(NI):
                            for i in range_constexpr(8):
                                x = (vis[ni] >= mt * 16 + i).select(
                                    accs[ni][i], neg_inf
                                )
                                run[ni] = maxf_nn(run[ni], x)
                    return [g.xlane2(r, maxf_nn) for r in run]

                def store_group(g0, sc):
                    """Scores sc[j][ni] of blocks g0 .. g0 + 3 with sentinels; lane
                    half h stores N-tiles h, h + 2."""
                    full = g0 + GROUP <= blk1
                    for p in range_constexpr(NI // 2):
                        ni_a, ni_b = 2 * p, 2 * p + 1
                        row = (h == 0).select(rows[ni_a], rows[ni_b])
                        valid = row < q_rows
                        tok = row // H
                        nb_row = (prefix + tok) // BLK + 1
                        vals = [
                            with_sentinels(
                                (h == 0).select(sc[j][ni_a], sc[j][ni_b]),
                                g0 + j,
                                nb_row,
                                i32_init,
                                i32_local,
                            )
                            for j in range_constexpr(GROUP)
                        ]
                        off = (row % H) * i32_hs + (seq_start + tok) * i32_S + g0
                        if full:
                            buffer_ops.buffer_store(
                                Vec.from_elements(vals, fx.Float32), sr, off, mask=valid
                            )
                        else:
                            for j in range_constexpr(GROUP):
                                buffer_ops.buffer_store(
                                    vals[j], sr, off + j, mask=valid & (g0 + j < blk1)
                                )

                n_groups = (blk1 - blk0 + (GROUP - 1)) // GROUP
                raw0 = load_raw(blk0)
                init = [fx.as_ir_value(x) for x in raw0]
                for gi, st in range(  # type: ignore[call-overload]
                    fx.Index(0), fx.Index(n_groups), fx.Index(1), init=init
                ):
                    raw = [fx.Int32(st[i]) for i in range_constexpr(18)]
                    g0 = blk0 + fx.Int32(gi) * GROUP
                    sc = []
                    for j in range_constexpr(GROUP):
                        dequant_to_lds(raw, j % 2)
                        raw = load_raw(g0 + j + 1)
                        fx.gpu.barrier()
                        sc.append(score_block(j % 2, g0 + j))
                    store_group(g0, sc)
                    _ = yield [fx.as_ir_value(x) for x in raw]

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_lut: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_cu: fx.Int64,
        arg_seq: fx.Int64,
        i64_page_stride: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_hs: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_kv,
            arg_lut,
            arg_score,
            arg_bt,
            arg_cu,
            arg_seq,
            i64_page_stride,
            i32_bt_stride,
            i32_S,
            i32_nseg,
            i32_qt,
            i32_total_q,
            i32_nb,
            i32_init,
            i32_local,
            i32_hs,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NTH, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
