# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1250 (MI455, CDNA5) counterparts of the gfx950 helpers in
``flydsl_utils``, for the MiniMax-M3 NVFP4 FlyDSL kernels.

gfx1250 runs wave32 only and has WMMA instead of MFMA. It also lacks gfx950's
2-wide ``cvt_scalef32_pk_*`` conversions, ``permlane32_swap`` and
``s_waitcnt`` (FlyDSL's LLVM aborts the process on the conversions). The
architecture-neutral helpers (``perm``, ``dequant8``, ``key_order4``, ``exp2``,
``pk_bf16``/``pk_f16``, ``bits``/``as_f32``, ``run_compiled``) are reused from
``flydsl_utils``.

WMMA 16x16 accumulators: lane ``l``, element ``i`` holds D[(l // 16) * 8 + i][l % 16].
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.flydsl_utils import as_f32, bits, permlane_swap

WAVE = 32
# Cache-policy bits of buffer loads and stores (TH[2:0], SCOPE[4:3]). gfx950's
# sc1 (coherent across the XCDs' L2s) is device scope, sc0 | sc1 system scope.
TH_NT = 0x1
SCOPE_DEV = 0x10
SCOPE_SYS = 0x18
# E8M0 1.0 for the 8-wide conversions, one byte per scale_sel lane group.
E8M0_ONE = 0x7F7F7F7F


def xlane2(x, op):
    """``op`` over lanes L and L^16, broadcast to both (the rows of a WMMA 16x16
    accumulator column are spread over these two lanes)."""
    a, b = permlane_swap(rocdl.permlane16_swap, bits(x), bits(x))
    return op(as_f32(a), as_f32(b))


def wmma_16x16x32(a8, b8, acc8, fp16: bool):
    """v_wmma_f32_16x16x32_{bf16,f16}: 8-dword A/B (16 values), 8 x fp32 acc."""
    dtype = fx.Float16 if fp16 else fx.BFloat16
    op = rocdl.wmma_f32_16x16x32_f16 if fp16 else rocdl.wmma_f32_16x16x32_bf16
    a = Vec(a8).bitcast(dtype)
    b = Vec(b8).bitcast(dtype)
    r = op(
        Vec.make_type(8, fx.Float32),
        fx.as_ir_value(a),
        fx.as_ir_value(b),
        fx.as_ir_value(Vec(acc8)),
        reuseA=False,
        reuseB=False,
    )
    return Vec(r.result if hasattr(r, "result") else r)


def wmma_fp8_16x16x128(a16, b16, acc8):
    """v_wmma_f32_16x16x128_fp8_fp8: 16-dword e4m3 A/B (64 values), 8 x fp32 acc.

    Not exact: its accumulation is about 2e-3 relative off fp32 on random e4m3
    data. Kernels that must match gfx950's fp8 MFMA scores widen to bf16 and use
    ``wmma_16x16x32`` instead."""
    return Vec(
        rocdl.wmma_f32_16x16x128_fp8_fp8(
            Vec.make_type(8, fx.Float32), Vec(a16), Vec(b16), Vec(acc8)
        )
    )


def pack16(*quads):
    """4 x (4 x i32) -> the 16 x i32 operand of the fp8 16x16x128 WMMA."""
    return Vec.from_elements([Vec(q)[k] for q in quads for k in range(4)], fx.Int32)


def cvt_fp4x8_f16(word):
    """The 8 e2m1 codes of ``word`` (low nibble first) as 8 fp16 (exact)."""
    return Vec(
        rocdl.cvt_scale_pk8_f16_fp4(
            Vec.make_type(8, fx.Float16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Int32(E8M0_ONE)),
            0,
        )
    )


def cvt_fp4x2_f16(word, sel):
    """The two e2m1 codes of byte ``sel`` of ``word`` as two fp16 (exact).
    Prefer ``cvt_fp4x8_f16`` for whole dwords."""
    r = cvt_fp4x8_f16(fx.Int32(word) >> (8 * sel))
    return Vec.from_elements([r[0], r[1]], fx.Float16)


def cvt_fp8x2_f16(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` as two fp16 (exact)."""
    half = (fx.Int32(word) >> (16 if hi else 0)).to(fx.Int16)
    r = llvm.call_intrinsic(
        Vec.make_type(2, fx.Float16),
        "llvm.amdgcn.cvt.pk.f16.fp8",
        [fx.as_ir_value(half)],
        [],
        [],
    )
    return Vec(r)


def cvt_fp8x2_f32(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` as two fp32 (exact). Goes
    through fp16: gfx1250's assembler rejects ``v_cvt_pk_f32_fp8`` as FlyDSL
    emits it."""
    return cvt_fp8x2_f16(word, hi).to(fx.Float32)


def cvt_fp8x2_bf16(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` as one dword of two bf16
    (exact: every e4m3 value is a bf16)."""
    f = cvt_fp8x2_f32(word, hi)
    return fx.Int32(rocdl.cvt_pk_bf16_f32(fx.as_ir_value(f[0]), fx.as_ir_value(f[1])))


def _fp8x8(word0, word1):
    return fx.as_ir_value(
        Vec.from_elements([fx.Int32(word0), fx.Int32(word1)], fx.Int32)
    )


def cvt_fp8x8_f16(word0, word1):
    """The 8 e4m3 bytes of two dwords (word0 first) as 8 fp16 (exact)."""
    return Vec(
        rocdl.cvt_scale_pk8_f16_fp8(
            Vec.make_type(8, fx.Float16),
            _fp8x8(word0, word1),
            fx.as_ir_value(fx.Int32(E8M0_ONE)),
            0,
        )
    )


def cvt_fp8x8_bf16(word0, word1):
    """The 8 e4m3 bytes of two dwords as 4 dwords of bf16 pairs (exact)."""
    return Vec(
        rocdl.cvt_scale_pk8_bf16_fp8(
            Vec.make_type(8, fx.BFloat16),
            _fp8x8(word0, word1),
            fx.as_ir_value(fx.Int32(E8M0_ONE)),
            0,
        )
    ).bitcast(fx.Int32)


def f16x8_fp8(v8):
    """8 fp16 in [-448, 448] -> two dwords of RNE e4m3 bytes."""
    return Vec(
        rocdl.cvt_scalef32_pk8_fp8_f16(
            Vec.make_type(2, fx.Int32),
            fx.as_ir_value(Vec(v8)),
            fx.as_ir_value(fx.Float32(1.0)),
        )
    )


def f32x2_fp8(a, b, old=0, hi=False):
    """Two fp32 -> two RNE e4m3 bytes in the low (``hi``: high) half of ``old``.
    Callers clamp to +-448 first, as the gfx950 kernels do."""
    return fx.Int32(
        rocdl.cvt_pk_fp8_f32(
            T.i32,
            fx.as_ir_value(fx.Float32(a)),
            fx.as_ir_value(fx.Float32(b)),
            fx.as_ir_value(fx.Int32(old)),
            bool(hi),
        )
    )


def e4m3_byte(x):
    """RNE e4m3 byte of fp32 ``x`` in [-448, 448]."""
    return f32x2_fp8(x, 0.0) & fx.Int32(0xFF)


def e4m3_value(byte):
    """fp32 value of an e4m3 byte."""
    return cvt_fp8x2_f32(fx.Int32(byte) & fx.Int32(0xFF), False)[0]


def e2m1x8(vals):
    """8 fp32 -> one dword of 8 e2m1 codes (low nibble first), round to nearest
    even and saturating, as ``cvt_scalef32_pk8_fp4_f32`` with a unit scale."""
    v = Vec.from_elements([fx.Float32(x) for x in vals], fx.Float32)
    return fx.Int32(
        rocdl.cvt_scalef32_pk8_fp4_f32(
            T.i32, fx.as_ir_value(v), fx.as_ir_value(fx.Float32(1.0))
        )
    )


def wait_load(n=0):
    rocdl.s_wait_loadcnt(n)


def wait_store(n=0):
    """Stores are not counted by the load counter on gfx1250: publish-then-flag
    protocols wait here before raising the flag."""
    rocdl.s_wait_storecnt(n)


def wait_ds(n=0):
    rocdl.s_wait_dscnt(n)


def wait_async(n=0):
    rocdl.s_wait_asynccnt(n)
