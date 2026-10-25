# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable Triton NVFP4 K/V cache for MiniMax-M3 on ROCm archs without the
gfx950 FlyDSL kernels (e.g. gfx1250).

The numerics match the gfx950 / CUDA encoders (see
``tests/kernels/attention/minimax_m3_nvfp4_ref.py``): every 16 values share an
E4M3 scale ``min(amax / 6 / global_scale, 448)`` and each value is the
round-to-nearest-even E2M1 code of ``x / (scale * global_scale)``.

The main K/V byte layout is the plain token-major head-slot one rather than
gfx950's MFMA-swizzled pages: the cache is ``[blocks, 2 * num_kv_heads, 128,
72]`` uint8, slot ``2 * head + side`` (K = 0, V = 1), and each token row is 64
bytes of E2M1 codes (the low nibble the even dim) followed by 8 E4M3 scale
bytes. The index-K cache uses the gfx950 index page ``[128 x 64 B data][128 x
8 B scales]`` with a unit global scale; storing it in NVFP4 too keeps its pages
a power-of-2 fraction of the main pages, so the hybrid allocator need not pad.
"""

import torch

from vllm.triton_utils import tl, triton

_GROUP = 16
NVFP4_GROUP = tl.constexpr(_GROUP)


@triton.jit
def _e2m1_value(code):
    """fp32 value of int32 E2M1 codes (sign bit 3, exponent 2..1, mantissa 0)."""
    e = (code >> 1) & 3
    m = (code & 1).to(tl.float32)
    mag = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where((code & 8) != 0, -mag, mag)


@triton.jit
def nvfp4_load_dequant(row_ptr, d_off, mask, global_scale, HEAD_DIM: tl.constexpr):
    """fp32 values of NVFP4 rows: ``row_ptr`` points at each element's token row
    and ``d_off`` is each element's dim (both broadcast to the tile shape)."""
    byte = tl.load(row_ptr + d_off // 2, mask=mask, other=0).to(tl.int32)
    code = (byte >> ((d_off % 2) * 4)) & 0xF
    scale_byte = tl.load(
        row_ptr + HEAD_DIM // 2 + d_off // NVFP4_GROUP, mask=mask, other=0
    )
    scale = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return _e2m1_value(code) * (scale * global_scale)


@triton.jit
def nvfp4_index_load_dequant(
    page_ptr, pos, d_off, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr
):
    """fp32 index keys of an NVFP4 index page (``pos`` / ``d_off`` broadcast)."""
    byte = tl.load(page_ptr + pos * (HEAD_DIM // 2) + d_off // 2).to(tl.int32)
    code = (byte >> ((d_off % 2) * 4)) & 0xF
    scale_byte = tl.load(
        page_ptr
        + BLOCK_SIZE * (HEAD_DIM // 2)
        + pos * (HEAD_DIM // NVFP4_GROUP)
        + d_off // NVFP4_GROUP
    )
    return _e2m1_value(code) * scale_byte.to(tl.float8e4nv, bitcast=True).to(
        tl.float32
    )


@triton.jit
def _e2m1_encode(y):
    """Round-to-nearest-even E2M1 codes of fp32 ``y``; ties go to the even code
    and magnitudes above 6 saturate."""
    a = tl.abs(y)
    # The code is the number of rounding boundaries passed; a tie stays on the
    # even code (> at boundaries above an even code, >= above an odd one).
    mag = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    sign = (y.to(tl.int32, bitcast=True) < 0).to(tl.int32)
    return mag | (sign << 3)


@triton.jit
def _nvfp4_insert_kernel(
    src_ptr,  # [num_tokens, num_heads, HEAD_DIM]
    cache_ptr,  # uint8 pages
    slot_ptr,  # [num_tokens], < 0 skips the token
    global_scale_ptr,  # fp32 global scale, unused with UNIT_SCALE
    stride_src_t,
    stride_src_h,
    stride_cache_blk,
    stride_cache_slot,  # bytes between head slots
    SIDE: tl.constexpr,  # head slot 2 * head + SIDE (0 for the index cache)
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DATA_ROW: tl.constexpr,  # bytes between token rows of data
    SCALE_BASE: tl.constexpr,  # offset of the scales in a head slot
    SCALE_ROW: tl.constexpr,  # bytes between token rows of scales
    UNIT_SCALE: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_ptr + tok).to(tl.int64)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    pos = slot % BLOCK_SIZE

    NUM_GROUPS: tl.constexpr = HEAD_DIM // NVFP4_GROUP
    x = tl.load(
        src_ptr
        + tok.to(tl.int64) * stride_src_t
        + head * stride_src_h
        + tl.arange(0, HEAD_DIM)
    ).to(tl.float32)
    groups = tl.reshape(x, (NUM_GROUPS, NVFP4_GROUP))

    # fp32 with IEEE-rounded divisions, in the reference's operation order.
    if UNIT_SCALE:
        global_scale = 1.0
    else:
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
    inv_global_scale = tl.math.div_rn(1.0, global_scale)
    amax = tl.max(tl.abs(groups), axis=1)
    scale = tl.minimum((amax * 0.16666666666666666) * inv_global_scale, 448.0)
    scale_e4m3 = scale.to(tl.float8e4nv)
    scale_value = scale_e4m3.to(tl.float32)
    quant = tl.where(
        scale_value != 0.0,
        tl.math.div_rn(tl.zeros_like(scale_value) + 1.0, scale_value * global_scale),
        0.0,
    )
    codes = _e2m1_encode(groups * quant[:, None])

    lo, hi = tl.split(tl.reshape(codes, (HEAD_DIM // 2, 2)))
    packed = (lo | (hi << 4)).to(tl.uint8)
    base = cache_ptr + blk * stride_cache_blk + (2 * head + SIDE) * stride_cache_slot
    tl.store(base + pos * DATA_ROW + tl.arange(0, HEAD_DIM // 2), packed)
    tl.store(
        base + SCALE_BASE + pos * SCALE_ROW + tl.arange(0, NUM_GROUPS),
        scale_e4m3.to(tl.uint8, bitcast=True),
    )


def nvfp4_kv_cache_view(kv_cache: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """``[blocks, 2 * num_kv_heads, block_size, 72]`` uint8 view of the cache."""
    if kv_cache.dtype != torch.uint8:
        kv_cache = kv_cache.view(torch.uint8)
    if kv_cache.dim() != 4 or kv_cache.shape[1] != 2 * num_kv_heads:
        kv_cache = kv_cache.view(kv_cache.shape[0], 2 * num_kv_heads, -1, 72)
    return kv_cache


def nvfp4_kv_insert(
    k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2 * num_kv_heads, block_size, 72] uint8
    slot_mapping: torch.Tensor,  # [num_tokens], < 0 skips the token
    k_scale: torch.Tensor,  # fp32 global scale
    v_scale: torch.Tensor,
) -> None:
    """Quantize K / V into the token-major NVFP4 head-slot cache."""
    num_tokens, num_kv_heads, head_dim = k.shape
    if num_tokens == 0 or kv_cache.numel() == 0:
        return
    assert v.shape == k.shape and k.stride(2) == 1 and v.stride(2) == 1
    assert head_dim % _GROUP == 0
    cache = nvfp4_kv_cache_view(kv_cache, num_kv_heads)
    assert cache.shape[3] == head_dim // 2 + head_dim // _GROUP
    assert cache.stride(3) == 1
    row = cache.stride(2)
    slots = slot_mapping[:num_tokens]
    for side, src, scale in ((0, k, k_scale), (1, v, v_scale)):
        _nvfp4_insert_kernel[(num_tokens, num_kv_heads)](
            src,
            cache,
            slots,
            scale,
            src.stride(0),
            src.stride(1),
            cache.stride(0),
            cache.stride(1),
            SIDE=side,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=cache.shape[2],
            DATA_ROW=row,
            SCALE_BASE=head_dim // 2,
            SCALE_ROW=row,
            UNIT_SCALE=False,
        )


def nvfp4_index_insert(
    index_k: torch.Tensor,  # [num_tokens, head_dim]
    index_cache: torch.Tensor,  # [num_blocks, block_size, 72] uint8
    slot_mapping: torch.Tensor,  # [num_tokens], < 0 skips the token
) -> None:
    """Quantize index keys into NVFP4 index pages (unit global scale)."""
    num_tokens, head_dim = index_k.shape
    if num_tokens == 0 or index_cache.numel() == 0:
        return
    assert index_k.stride(1) == 1 and head_dim % _GROUP == 0
    assert index_cache.dtype == torch.uint8 and index_cache.dim() == 3
    block_size = index_cache.shape[1]
    assert index_cache.shape[2] == head_dim // 2 + head_dim // _GROUP
    assert index_cache.stride(1) == index_cache.shape[2]
    assert index_cache.stride(2) == 1
    _nvfp4_insert_kernel[(num_tokens, 1)](
        index_k,
        index_cache,
        slot_mapping[:num_tokens],
        index_cache,  # unused with UNIT_SCALE
        index_k.stride(0),
        0,
        index_cache.stride(0),
        0,
        SIDE=0,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        DATA_ROW=head_dim // 2,
        SCALE_BASE=block_size * (head_dim // 2),
        SCALE_ROW=head_dim // _GROUP,
        UNIT_SCALE=True,
    )
