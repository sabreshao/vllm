# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx1250 FlyDSL helpers of the MiniMax-M3 NVFP4 kernels
(``amd/ops/flydsl_gfx1250.py``) against torch: WMMA results and lane layouts,
fp4 / fp8 conversions on every code, the L^16 cross-lane reduction, and the
store-wait before a ticket."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm FlyDSL kernels", allow_module_level=True)

from vllm.platforms.rocm import on_gfx1250  # noqa: E402

if not on_gfx1250():
    pytest.skip("gfx1250 instructions", allow_module_level=True)

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops  # noqa: E402
from flydsl._mlir.dialects import llvm as _llvm  # noqa: E402
from flydsl.expr.typing import Vector as Vec  # noqa: E402

from tests.kernels.attention.minimax_m3_nvfp4_ref import E2M1_GRID  # noqa: E402
from vllm.models.minimax_m3.amd.ops import flydsl_gfx1250 as g  # noqa: E402
from vllm.models.minimax_m3.amd.ops.flydsl_utils import (  # noqa: E402
    as_f32,
    bits,
    global_i32_ptr,
    maxf_nn,
)

LANES = 32


def _run(body, inp: torch.Tensor, n_out: int, name: str) -> torch.Tensor:
    """One wave32 workgroup: ``body(lane, load)`` returns ``n_out`` i32 values
    per lane, where ``load(i)`` reads dword ``i`` of ``inp``. Returns
    ``[32, n_out]`` int32."""
    inp = inp.contiguous().view(torch.int32).flatten().cuda()
    out = torch.full((LANES * n_out,), -1, dtype=torch.int32, device="cuda")
    n_in = inp.numel()

    @flyc.kernel(name=f"t1250_{name}", known_block_size=[LANES, 1, 1])
    def kernel(arg_in: fx.Int64, arg_out: fx.Int64):
        # FlyDSL keys its compile cache on source and scalar closure values,
        # not on ``body``: capturing ``name`` keeps every body its own kernel.
        assert name
        lane = fx.Int32(fx.thread_idx.x)
        rin = buffer_ops.create_buffer_resource_from_addr(
            arg_in, num_records_bytes=fx.Int64(n_in * 4)
        )
        rout = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=fx.Int64(LANES * n_out * 4)
        )

        def load(i):
            return fx.Int32(
                buffer_ops.buffer_load(rin, fx.Int32(i), vec_width=1, dtype=fx.Int32)
            )

        vals = body(lane, load)
        assert len(vals) == n_out
        for j, v in enumerate(vals):
            buffer_ops.buffer_store(fx.as_ir_value(fx.Int32(v)), rout, lane * n_out + j)

    @flyc.jit
    def launch(arg_in: fx.Int64, arg_out: fx.Int64, stream: fx.Stream):
        kernel(arg_in, arg_out).launch(
            grid=(1, 1, 1), block=(LANES, 1, 1), stream=stream
        )

    launch.kernel_name = f"t1250_{name}"
    flyc.compile(launch, inp.data_ptr(), out.data_ptr(), torch.cuda.current_stream())
    torch.accelerator.synchronize()
    return out.view(LANES, n_out).cpu()


def _acc_to_matrix(acc: torch.Tensor) -> torch.Tensor:
    """[32 lanes, 8] fp32 accumulators -> D[16, 16]."""
    d = torch.empty(16, 16)
    for lane in range(LANES):
        for i in range(8):
            d[(lane // 16) * 8 + i, lane % 16] = acc[lane, i]
    return d


@pytest.mark.parametrize("fp16", [False, True])
def test_wmma_16x16x32(fp16):
    """A row m and B column n live in lane m % 16 (n % 16); lanes 16-31 hold
    the other half of K. D[(l // 16) * 8 + i][l % 16] is element i of lane l."""
    torch.manual_seed(0)
    dt = torch.float16 if fp16 else torch.bfloat16
    a = torch.randn(16, 32).to(dt)
    b = torch.randn(32, 16).to(dt)
    # lane l: row/col l % 16, K slots 8h..8h+7 then 16+8h..16+8h+7 (h = l // 16)
    ks = [
        list(range(8 * h, 8 * h + 8)) + list(range(16 + 8 * h, 24 + 8 * h))
        for h in (0, 1)
    ]
    a_lanes = torch.stack([a[ln % 16, ks[ln // 16]] for ln in range(LANES)])
    b_lanes = torch.stack([b[ks[ln // 16], ln % 16] for ln in range(LANES)])
    inp = torch.cat([a_lanes.view(torch.int32), b_lanes.view(torch.int32)])  # [64, 8]

    def body(lane, load):
        a8 = Vec.from_elements([load(lane * 8 + k) for k in range(8)], fx.Int32)
        b8 = Vec.from_elements([load((32 + lane) * 8 + k) for k in range(8)], fx.Int32)
        acc = g.wmma_16x16x32(a8, b8, Vec.filled(8, 0.0, fx.Float32), fp16)
        return [bits(acc[i]) for i in range(8)]

    acc = _run(body, inp, 8, f"wmma32_{int(fp16)}").view(torch.float32)
    ref = a.float() @ b.float()
    torch.testing.assert_close(_acc_to_matrix(acc), ref, rtol=1e-5, atol=1e-4)


def test_wmma_fp8_16x16x128():
    """Lane layout of the fp8 WMMA. Its internal accumulation is not exact fp32
    (about 2e-3 relative on these values, deterministic), which is why the
    gfx1250 index scorers widen e4m3 to bf16 and use the exact bf16 WMMA."""
    torch.manual_seed(1)
    a = (torch.randn(16, 128) * 2).to(torch.float8_e4m3fn)
    b = (torch.randn(128, 16) * 2).to(torch.float8_e4m3fn)
    # K order only has to agree between A and B: lane half h takes K 64h..64h+63.
    a_lanes = torch.stack(
        [a[ln % 16, 64 * (ln // 16) : 64 * (ln // 16) + 64] for ln in range(LANES)]
    )
    b_lanes = torch.stack(
        [b[64 * (ln // 16) : 64 * (ln // 16) + 64, ln % 16] for ln in range(LANES)]
    )
    inp = torch.cat(
        [a_lanes.contiguous().view(torch.int32), b_lanes.contiguous().view(torch.int32)]
    )

    def body(lane, load):
        a16 = Vec.from_elements([load(lane * 16 + k) for k in range(16)], fx.Int32)
        b16 = Vec.from_elements(
            [load((32 + lane) * 16 + k) for k in range(16)], fx.Int32
        )
        acc = g.wmma_fp8_16x16x128(a16, b16, Vec.filled(8, 0.0, fx.Float32))
        return [bits(acc[i]) for i in range(8)]

    acc = _run(body, inp, 8, "wmma_fp8").view(torch.float32)
    ref = a.float() @ b.float()
    torch.testing.assert_close(_acc_to_matrix(acc), ref, rtol=5e-3, atol=2e-2)


def test_cvt_fp4x8_f16_every_code():
    codes = torch.arange(16)
    words = torch.tensor(
        [
            int(sum(int(c) << (4 * i) for i, c in enumerate(codes[8 * w : 8 * w + 8])))
            for w in (0, 1)
        ]
    )
    words = torch.where(words >= 1 << 31, words - (1 << 32), words).int()

    def body(lane, load):
        h = g.cvt_fp4x8_f16(load(lane & 1)).bitcast(fx.Int32)
        return [h[i] for i in range(4)]

    out = (
        _run(body, words, 4, "fp4x8")[:2]
        .contiguous()
        .view(torch.float16)
        .float()
        .flatten()
    )
    grid = torch.tensor(E2M1_GRID)
    ref = torch.cat([grid, -grid])
    assert torch.equal(out, ref)


def test_cvt_fp4x2_f16_byte_select():
    word = torch.tensor([0x76543210], dtype=torch.int64)
    word = torch.where(word >= 1 << 31, word - (1 << 32), word).int()

    def body(lane, load):
        return [g.cvt_fp4x2_f16(load(0), s).bitcast(fx.Int32)[0] for s in range(4)]

    out = _run(body, word, 4, "fp4x2")[0].contiguous().view(torch.float16).float()
    assert torch.equal(out, torch.tensor(E2M1_GRID))


def _e4m3_bytes():
    b = torch.arange(256, dtype=torch.int32)
    return b[(b & 0x7F) != 0x7F]  # drop the NaN encodings


def test_cvt_fp8x2_every_byte():
    """e4m3 -> f16, f32 and bf16 on every finite byte, both halves of the dword."""
    b = _e4m3_bytes()
    words = (b | (b << 8) | (((b + 1) % 256) << 16) | (((b + 2) % 256) << 24)).to(
        torch.int64
    )
    words = torch.where(words >= 1 << 31, words - (1 << 32), words).int()
    ref = b.to(torch.uint8).view(torch.float8_e4m3fn).float()
    for name, fn, cast in [
        (
            "f16",
            lambda w: g.cvt_fp8x2_f16(w, False).bitcast(fx.Int32)[0],
            torch.float16,
        ),
        ("bf16", lambda w: g.cvt_fp8x2_bf16(w, False), torch.bfloat16),
    ]:
        got = []
        for c in range(0, len(words), LANES):
            chunk = words[c : c + LANES]
            chunk = torch.cat([chunk, chunk[:1].expand(LANES - len(chunk))])

            def body(lane, load, fn=fn):
                return [fn(load(lane))]

            got.append(
                _run(body, chunk, 1, f"fp8x2_{name}_{c}")[: len(words[c : c + LANES])]
            )
        got = torch.cat(got).view(cast).view(-1, 2).float()
        assert torch.equal(got[:, 0], ref) and torch.equal(got[:, 1], ref), name

    def body(lane, load):
        f = g.cvt_fp8x2_f32(load(lane), True)
        return [bits(f[0]), bits(f[1])]

    hi = torch.cat([b, b]).view(-1)[:LANES].clone()
    w = ((hi << 16) | (((hi + 1) % 256) << 24)).to(torch.int64)
    w = torch.where(w >= 1 << 31, w - (1 << 32), w).int()
    got = _run(body, w, 2, "fp8x2_f32_hi").view(torch.float32)
    torch.testing.assert_close(
        got[:, 0], hi.to(torch.uint8).view(torch.float8_e4m3fn).float(), rtol=0, atol=0
    )


def test_e4m3_roundtrip_and_rounding():
    """RNE e4m3 of fp32 in [-448, 448], ties included, against torch."""
    vals = torch.cat([
        torch.linspace(-448, 448, LANES * 2 - 8),
        torch.tensor([1.0625, 1.1875, -1.0625, 0.0, -0.0, 2.0**-9, 2.0**-10, 448.0]),
    ])  # fmt: skip

    def body(lane, load):
        x0, x1 = as_f32(load(lane * 2)), as_f32(load(lane * 2 + 1))
        pk = g.f32x2_fp8(x0, x1)
        return [pk, bits(g.e4m3_value(g.e4m3_byte(x0)))]

    out = _run(body, vals, 2, "e4m3")
    ref = vals.to(torch.float8_e4m3fn)
    got = (
        (out[:, 0] & 0xFFFF).to(torch.int16).view(torch.uint8).view(torch.float8_e4m3fn)
    )
    assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
    assert torch.equal(out[:, 1].view(torch.float32), ref[0::2].float())


def _ref_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 code, ties to the even code, saturating at 6, signed."""
    grid = torch.tensor(E2M1_GRID)
    dist = (x.abs()[..., None] - grid).abs()
    is_min = dist == dist.amin(-1, keepdim=True)
    ids = torch.arange(8)
    even = torch.where(is_min & (ids % 2 == 0), ids, 8).amin(-1)
    mag = torch.where(even < 8, even, dist.argmin(-1))
    return mag | (torch.signbit(x).long() << 3)


def test_e2m1x8_rounding():
    """Ties to even, saturation above 6 and signed zeros, as the reference."""
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0]
    vals = torch.cat([
        torch.tensor(ties), -torch.tensor(ties),
        torch.tensor([0.0, -0.0, 0.2499, 0.2501, 6.0, 100.0, -100.0, 1e-30]),
        torch.linspace(-7, 7, LANES * 8 - 24),
    ])  # fmt: skip

    def body(lane, load):
        return [g.e2m1x8([as_f32(load(lane * 8 + k)) for k in range(8)])]

    out = _run(body, vals, 1, "e2m1x8")[:, 0].long() & 0xFFFFFFFF
    got = torch.stack([(out >> (4 * k)) & 0xF for k in range(8)], 1).flatten()
    assert torch.equal(got, _ref_e2m1_codes(vals))


def test_xlane2_max():
    vals = torch.randn(LANES)

    def body(lane, load):
        return [bits(g.xlane2(as_f32(load(lane)), maxf_nn))]

    out = _run(body, vals, 1, "xlane2")[:, 0].view(torch.float32)
    ref = torch.maximum(vals, vals.roll(16))
    assert torch.equal(out, ref)


def test_store_wait_before_ticket():
    """Every workgroup stores a partial, waits on the store counter, then takes a
    ticket; the last arriver sums all partials and must see every one."""
    wgs, rounds = 256, 20
    for r in range(rounds):
        part = torch.zeros(wgs, dtype=torch.int32, device="cuda")
        ticket = torch.zeros(1, dtype=torch.int32, device="cuda")
        out = torch.zeros(1, dtype=torch.int32, device="cuda")

        @flyc.kernel(name="t1250_ticket", known_block_size=[LANES, 1, 1])
        def kernel(arg_part: fx.Int64, arg_ticket: fx.Int64, arg_out: fx.Int64):
            lane = fx.Int32(fx.thread_idx.x)
            wg = fx.Int32(fx.block_idx.x)
            rp = buffer_ops.create_buffer_resource_from_addr(
                arg_part, num_records_bytes=fx.Int64(wgs * 4)
            )
            if lane == 0:
                buffer_ops.buffer_store(
                    fx.as_ir_value(wg + 1), rp, wg, cache_modifier=g.SCOPE_DEV
                )
                g.wait_store()
                old = fx.Int32(
                    _llvm.AtomicRMWOp(
                        _llvm.AtomicBinOp.add,
                        global_i32_ptr(arg_ticket).llvm_ptr,
                        fx.Int32(1).ir_value(),
                        _llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    ).result
                )
                if old == wgs - 1:
                    acc = fx.Int32(0)
                    for i in range(wgs):
                        acc = acc + fx.Int32(
                            buffer_ops.buffer_load(
                                rp, fx.Int32(i), vec_width=1, dtype=fx.Int32,
                                cache_modifier=g.SCOPE_DEV,
                            )
                        )  # fmt: skip
                    ro = buffer_ops.create_buffer_resource_from_addr(
                        arg_out, num_records_bytes=fx.Int64(4)
                    )
                    buffer_ops.buffer_store(fx.as_ir_value(acc), ro, fx.Int32(0))

        @flyc.jit
        def launch(
            arg_part: fx.Int64,
            arg_ticket: fx.Int64,
            arg_out: fx.Int64,
            stream: fx.Stream,
        ):
            kernel(arg_part, arg_ticket, arg_out).launch(
                grid=(wgs, 1, 1), block=(LANES, 1, 1), stream=stream
            )

        launch.kernel_name = "t1250_ticket"
        flyc.compile(
            launch, part.data_ptr(), ticket.data_ptr(), out.data_ptr(),
            torch.cuda.current_stream(),
        )  # fmt: skip
        torch.accelerator.synchronize()
        assert int(out.item()) == wgs * (wgs + 1) // 2, r
