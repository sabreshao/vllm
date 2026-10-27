# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An MXFP8 copy riding along with a bf16 activation.

A producer kernel that can emit its bf16 output MXFP8-quantized as well (e4m3
data [M, K] plus row-major E8M0 scales [M, K // 32], bit-identical to
`mxfp8_e4m3_quantize` of the bf16 output) attaches the pair to the bf16
tensor. Consumers that would quantize that same tensor take the pair instead:
the ROCm block32 MXFP8 linears and MoRI's MXFP8 dispatch. Unlike a
QuantizedActivation, the bf16 tensor still flows to consumers that need it
(router gate, compressor, indexer), so producer call sites stay unchanged.

The pair is tied to the tensor object: views and copies do not carry it, and
the producer must not hand out a tensor it later mutates in place.
"""

import torch

_ATTR = "_vllm_mxfp8_companion"


def empty_mxfp8(
    shape: tuple[int, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uninitialized (e4m3 data, E8M0 scale) buffers for a [..., K] tensor."""
    *rows, hidden = shape
    return (
        torch.empty(shape, dtype=torch.float8_e4m3fn, device=device),
        torch.empty((*rows, hidden // 32), dtype=torch.uint8, device=device),
    )


def attach_mxfp8(x: torch.Tensor, data: torch.Tensor, scale: torch.Tensor) -> None:
    assert data.shape == x.shape and scale.shape == (*x.shape[:-1], x.shape[-1] // 32)
    setattr(x, _ATTR, (data, scale))


def get_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The (e4m3 data, E8M0 scale) attached to `x` by its producer, if any."""
    return getattr(x, _ATTR, None)
