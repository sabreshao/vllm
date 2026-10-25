# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 portable Triton NVFP4 K/V cache (non-gfx950 ROCm) against the
torch reference in ``minimax_m3_nvfp4_ref``."""

import pytest
import torch

from tests.kernels.attention.minimax_m3_nvfp4_ref import (
    dequant,
    insert_nvfp4_index,
    quantize_nvfp4,
)
from vllm.models.minimax_m3.common.ops.nvfp4_triton import (
    nvfp4_index_insert,
    nvfp4_kv_insert,
    nvfp4_load_dequant,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm Triton NVFP4 path"
)

D = 128
BLOCK = 128
ROW = D // 2 + D // 16


def _rows(kind: str, n: int, device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(0)
    if kind == "random":
        x = torch.randn(n, D, generator=g, device=device) * 3
    elif kind == "outlier":
        x = torch.randn(n, D, generator=g, device=device)
        x[:, ::17] *= 300
    elif kind == "zero":
        x = torch.zeros(n, D, device=device)
        x[1::2, :16] = torch.randn(n // 2, 16, generator=g, device=device)
    else:  # values on E2M1 midpoints of a 1.0 block scale
        grid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device=device)
        x = grid.repeat(n, D // 8)
        x[:, 1::2] *= -1
    return x.to(torch.bfloat16)


def _split(cache: torch.Tensor, slots: torch.Tensor, slot_idx: int):
    rows = cache.view(-1, cache.shape[1], BLOCK, ROW)[
        slots // BLOCK, slot_idx, slots % BLOCK
    ]
    return rows[:, : D // 2], rows[:, D // 2 :]


def _codes_to_bytes(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.to(torch.uint8)
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


@pytest.mark.parametrize("kind", ["random", "outlier", "zero", "tie"])
@pytest.mark.parametrize("global_scale", [1.0, 0.0625, 3.0])
def test_insert_matches_reference(kind, global_scale):
    device = "cuda"
    n, heads, blocks = 64, 2, 4
    k = torch.stack([_rows(kind, n, device), -_rows(kind, n, device)], 1)
    v = torch.stack([_rows(kind, n, device) * 0.5, _rows(kind, n, device)], 1)
    cache = torch.zeros(blocks, 2 * heads, BLOCK, ROW, dtype=torch.uint8, device=device)
    perm = torch.randperm(blocks * BLOCK, device=device)[:n]
    slots = perm.clone()
    slots[5] = -1  # skipped token
    gs = torch.tensor([global_scale], dtype=torch.float32, device=device)
    nvfp4_kv_insert(k, v, cache, slots, gs, gs)

    keep = slots >= 0
    for h in range(heads):
        for side, src in ((0, k), (1, v)):
            codes, scales = quantize_nvfp4(src[keep, h].float(), global_scale)
            data, scale = _split(cache, slots[keep], 2 * h + side)
            torch.testing.assert_close(data, _codes_to_bytes(codes), rtol=0, atol=0)
            torch.testing.assert_close(scale, scales, rtol=0, atol=0)
    # The skipped token's slot stays untouched.
    data, scale = _split(cache, perm[5:6], 0)
    assert int(data.abs().sum()) == 0 and int(scale.abs().sum()) == 0


@triton.jit
def _dequant_rows_kernel(
    cache_ptr, out_ptr, gs, ROW_BYTES: tl.constexpr, D: tl.constexpr
):
    r = tl.program_id(0)
    d = tl.arange(0, D)
    vals = nvfp4_load_dequant(
        cache_ptr + r * ROW_BYTES + 0 * d, d, d < D, gs, HEAD_DIM=D
    )
    tl.store(out_ptr + r * D + d, vals)


def test_dequant_matches_reference():
    device = "cuda"
    n = 96
    x = _rows("outlier", n, device)
    gs = 0.25
    codes, scales = quantize_nvfp4(x.float(), gs)
    rows = torch.cat([_codes_to_bytes(codes), scales], 1).contiguous()
    out = torch.empty(n, D, dtype=torch.float32, device=device)
    _dequant_rows_kernel[(n,)](rows, out, gs, ROW_BYTES=ROW, D=D)
    ref = dequant(_codes_to_bytes(codes), scales) * gs
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def _attn_setup(device, num_kv_heads=2, group=16, blocks=4, seq_len=4 * BLOCK):
    """A one-request cache of ``seq_len`` tokens on shuffled pages, as NVFP4,
    as bf16 of the NVFP4 values, and as the unquantized bf16 K/V."""
    g = torch.Generator(device=device).manual_seed(1)
    k = torch.randn(seq_len, num_kv_heads, D, generator=g, device=device)
    v = torch.randn(seq_len, num_kv_heads, D, generator=g, device=device)
    k, v = k.to(torch.bfloat16), v.to(torch.bfloat16)
    pages = torch.randperm(blocks, device=device)
    slots = (pages[torch.arange(seq_len, device=device) // BLOCK] * BLOCK) + (
        torch.arange(seq_len, device=device) % BLOCK
    )
    k_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    v_scale = torch.tensor([2.0], dtype=torch.float32, device=device)
    nvfp4 = torch.zeros(
        blocks, 2 * num_kv_heads, BLOCK, ROW, dtype=torch.uint8, device=device
    )
    nvfp4_kv_insert(k, v, nvfp4, slots, k_scale, v_scale)

    def to_bf16_cache(kk, vv):
        cache = torch.zeros(
            blocks, num_kv_heads, BLOCK, 2 * D, dtype=torch.bfloat16, device=device
        )
        cache[slots // BLOCK, :, slots % BLOCK] = torch.cat([kk, vv], -1).to(
            torch.bfloat16
        )
        return cache

    deq = []
    for side, scale in ((0, k_scale), (1, v_scale)):
        per_head = []
        for h in range(num_kv_heads):
            data, sc = _split(nvfp4, slots, 2 * h + side)
            per_head.append(dequant(data, sc) * scale)
        deq.append(torch.stack(per_head, 1))
    block_table = pages[None].to(torch.int32)
    return dict(
        nvfp4=nvfp4,
        deq_cache=to_bf16_cache(*deq),
        ref_cache=to_bf16_cache(k, v),
        k_scale=k_scale,
        v_scale=v_scale,
        block_table=block_table,
        num_heads=num_kv_heads * group,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        blocks=blocks,
    )


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def test_sparse_decode_nvfp4():
    from vllm.models.minimax_m3.amd.ops.sparse_attn import (
        minimax_m3_sparse_attn_decode,
    )

    device = "cuda"
    s = _attn_setup(device)
    total_q = 1
    q = torch.randn(total_q, s["num_heads"], D, device=device).to(torch.bfloat16)
    topk = torch.arange(s["blocks"], device=device, dtype=torch.int32)
    topk = topk.expand(s["num_kv_heads"], total_q, -1).contiguous()
    seq_lens = torch.tensor([s["seq_len"]], dtype=torch.int32, device=device)

    def run(cache, nvfp4):
        out = torch.empty_like(q)
        minimax_m3_sparse_attn_decode(
            q, cache, topk, s["block_table"], seq_lens, s["num_kv_heads"],
            D**-0.5, out, 1,
            k_scale=s["k_scale"] if nvfp4 else None,
            v_scale=s["v_scale"] if nvfp4 else None,
            kv_nvfp4=nvfp4,
        )
        return out

    out = run(s["nvfp4"], True)
    torch.testing.assert_close(out, run(s["deq_cache"], False), rtol=0, atol=1e-2)
    assert _cos(out, run(s["ref_cache"], False)) > 0.98


def test_sparse_prefill_nvfp4():
    from vllm.models.minimax_m3.amd.ops.sparse_attn import minimax_m3_sparse_attn

    device = "cuda"
    s = _attn_setup(device)
    q_len = BLOCK  # queries in the last block: every selected block is visible
    prefix = s["seq_len"] - q_len
    q = torch.randn(q_len, s["num_heads"], D, device=device).to(torch.bfloat16)
    topk = torch.arange(s["blocks"], device=device, dtype=torch.int32)
    topk = topk.expand(s["num_kv_heads"], q_len, -1).contiguous()
    cu = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([s["seq_len"]], dtype=torch.int32, device=device)
    prefix_lens = torch.tensor([prefix], dtype=torch.int32, device=device)

    def run(cache, nvfp4):
        out = torch.empty_like(q)
        minimax_m3_sparse_attn(
            q, cache, topk, s["block_table"], cu, seq_lens, prefix_lens, q_len,
            s["num_kv_heads"], D**-0.5, out,
            k_scale=s["k_scale"] if nvfp4 else None,
            v_scale=s["v_scale"] if nvfp4 else None,
            kv_nvfp4=nvfp4,
        )
        return out

    out = run(s["nvfp4"], True)
    torch.testing.assert_close(out, run(s["deq_cache"], False), rtol=0, atol=1e-2)
    assert _cos(out, run(s["ref_cache"], False)) > 0.98


def _index_setup(device, blocks=6, seq_len=6 * BLOCK - 37):
    """Index keys of one request on shuffled pages: NVFP4 pages (checked
    against the reference insert) and bf16 pages of the dequantized keys."""
    g = torch.Generator(device=device).manual_seed(2)
    keys = (torch.randn(seq_len, D, generator=g, device=device) * 2).to(torch.bfloat16)
    pages = torch.randperm(blocks, device=device)
    pos = torch.arange(seq_len, device=device)
    slots = pages[pos // BLOCK] * BLOCK + pos % BLOCK
    nvfp4 = torch.zeros(blocks, BLOCK, ROW, dtype=torch.uint8, device=device)
    nvfp4_index_insert(keys, nvfp4, slots)
    ref = torch.zeros_like(nvfp4)
    insert_nvfp4_index(ref, keys, slots)
    codes, scales = quantize_nvfp4(keys.float(), 1.0)
    deq = dequant(_codes_to_bytes(codes), scales)
    bf16 = torch.zeros(blocks, BLOCK, D, dtype=torch.bfloat16, device=device)
    bf16[slots // BLOCK, slots % BLOCK] = deq.to(torch.bfloat16)
    return nvfp4, ref, bf16, pages[None].to(torch.int32), seq_len


def test_index_insert_matches_reference():
    nvfp4, ref, *_ = _index_setup("cuda")
    torch.testing.assert_close(nvfp4, ref, rtol=0, atol=0)


def test_index_prefill_score_nvfp4():
    from vllm.models.minimax_m3.amd.ops.index_topk import minimax_m3_index_score

    device = "cuda"
    nvfp4, _, bf16, block_table, seq_len = _index_setup(device)
    q_len = 200
    q = torch.randn(q_len, 1, D, device=device).to(torch.bfloat16)
    cu = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    prefix = torch.tensor([seq_len - q_len], dtype=torch.int32, device=device)

    def run(cache):
        return minimax_m3_index_score(
            q, cache, block_table, cu, seq_lens, prefix, q_len, seq_len, 1
        )

    out, ref = run(nvfp4), run(bf16)
    max_block = (seq_len + BLOCK - 1) // BLOCK
    out, ref = out[0, :, :max_block], ref[0, :, :max_block]
    # Only blocks starting at or before a query's position are scored; the
    # kernel leaves the rest of the row unwritten.
    q_pos = torch.arange(q_len, device=device)[:, None] + (seq_len - q_len)
    visible = torch.arange(max_block, device=device)[None, :] * BLOCK <= q_pos
    torch.testing.assert_close(out[visible], ref[visible], rtol=1e-3, atol=1e-2)


def test_index_decode_topk_nvfp4():
    from vllm.models.minimax_m3.amd.ops.index_topk import minimax_m3_index_decode

    device = "cuda"
    nvfp4, _, bf16, block_table, seq_len = _index_setup(
        device, blocks=40, seq_len=40 * BLOCK - 5
    )
    q = torch.randn(1, 1, D, device=device).to(torch.bfloat16)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    def run(cache):
        ids = minimax_m3_index_decode(
            q, cache, block_table, seq_lens, seq_len, 16, 1, 2, 1, 1, 1
        )
        return torch.sort(ids, dim=-1).values

    torch.testing.assert_close(run(nvfp4), run(bf16), rtol=0, atol=0)
