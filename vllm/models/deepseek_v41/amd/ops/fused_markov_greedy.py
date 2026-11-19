# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's greedy Markov chain in two launches a draft step.

With greedy drafts (no draft logits cache), vLLM's loop in
`DSparkSpeculator._sample_markov_chain` runs for each of the 5 draft steps:
the Markov embedding gather, the bias GEMM (rows x 129280 x 256, fp32 out),
the fp32 add to the base logits, a two-kernel argmax and the copy of the
tokens. At DP8 with ~64 requests a rank that is ~50 us a step on MI355X,
most of it the GEMM and the add, which write and read back the fp32
rows x vocab bias.

Here a step is:
  * `_step_kernel`: each program takes a block of the vocabulary, computes
    the bias of every row for it with one dot over the Markov rank, adds the
    base logits and stores each row's block maximum and its token id. The
    bias never leaves registers.
  * `_pick_kernel`: one program a row reduces the block maxima to the token,
    writes it into the draft tokens and gathers its Markov embedding for the
    next step (step 0's embedding comes from the anchor token).

Numerics: the bias is an fp32 sum of the 256 bf16 products, in another order
than the GEMM's, rounded to bf16 when the head is bf16 (as is the sum with
the bf16 logits); a bias can differ in its last bit.
NaN logits count as -inf and ties take the first token, as `greedy_argmax`.
Draft tokens only change what the target verifies, never its output.

On by default on ROCm; VLLM_DSPARK_FUSED_GREEDY_MARKOV=0 keeps vLLM's loop.
"""

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

BLOCK_V = 128  # vocabulary tokens a program (swept on MI355X: 128 x 32 rows, 4 warps)
BLOCK_R = 32  # rows a dot
_buffers: dict = {}


@triton.jit
def _step_kernel(
    emb_ptr,  # [rows, RANK] bf16 Markov embedding of the previous tokens
    w2_ptr,  # [vocab, RANK] bf16 markov_w2
    base_ptr,  # base logits of row r at r * base_stride + v (fp32)
    base_stride,
    out_max_ptr,  # [rows, nb] fp32
    out_arg_ptr,  # [rows, nb] int32
    rows,
    vocab,
    nb,
    RANK: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BF16_BIAS: tl.constexpr,  # bf16 head: the GEMM writes a bf16 bias
    BF16_SUM: tl.constexpr,  # bf16 logits + bf16 bias round to bf16
):
    pid = tl.program_id(0)
    v = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    vmask = v < vocab
    k = tl.arange(0, RANK)
    w = tl.load(
        w2_ptr + v[:, None].to(tl.int64) * RANK + k[None, :],
        mask=vmask[:, None],
        other=0.0,
    )
    wt = tl.trans(w)  # [RANK, BLOCK_V]
    for r0 in range(0, rows, BLOCK_R):
        r = r0 + tl.arange(0, BLOCK_R)
        rmask = r < rows
        e = tl.load(
            emb_ptr + r[:, None] * RANK + k[None, :], mask=rmask[:, None], other=0.0
        )
        base = tl.load(
            base_ptr + r[:, None].to(tl.int64) * base_stride + v[None, :],
            mask=rmask[:, None] & vmask[None, :],
            other=float("-inf"),
        ).to(tl.float32)
        bias = tl.dot(e, wt)  # [BLOCK_R, BLOCK_V] fp32, stays in registers
        if BF16_BIAS:
            bias = bias.to(tl.bfloat16).to(tl.float32)
        x = base + bias
        if BF16_SUM:
            x = x.to(tl.bfloat16).to(tl.float32)
        x = tl.where(x != x, float("-inf"), x)
        x = tl.where(vmask[None, :], x, float("-inf"))
        idx = tl.argmax(x, axis=1)
        mx = tl.max(x, axis=1)
        tl.store(out_max_ptr + r * nb + pid, mx, mask=rmask)
        tl.store(out_arg_ptr + r * nb + pid, pid * BLOCK_V + idx, mask=rmask)


@triton.jit
def _pick_kernel(
    max_ptr,  # [rows, nb] fp32 (unused at step 0)
    arg_ptr,  # [rows, nb] int32
    nb,
    input_ids_ptr,
    anchor_idx_ptr,
    tokens_ptr,  # draft_tokens [max_num_reqs, n_spec] int64
    tokens_stride,
    w1_ptr,  # [vocab, RANK] bf16 markov_w1
    emb_ptr,  # [rows, RANK] bf16 output embedding
    STEP: tl.constexpr,  # draft step whose token this picks; -1 = anchor
    GATHER: tl.constexpr,
    RANK: tl.constexpr,
    NB_POW2: tl.constexpr,
):
    row = tl.program_id(0)
    if STEP < 0:
        tok = tl.load(input_ids_ptr + tl.load(anchor_idx_ptr + row)).to(tl.int64)
    else:
        c = tl.arange(0, NB_POW2)
        m = tl.load(max_ptr + row * nb + c, mask=c < nb, other=float("-inf"))
        p = tl.argmax(m, axis=0)
        tok = tl.load(arg_ptr + row * nb + p).to(tl.int64)
        tl.store(tokens_ptr + row * tokens_stride + STEP, tok)
    if GATHER:
        k = tl.arange(0, RANK)
        tl.store(emb_ptr + row * RANK + k, tl.load(w1_ptr + tok * RANK + k))


def _why_not(spec) -> str | None:
    head = getattr(getattr(spec.model, "model", None), "markov_head", None)
    if head is None:
        return "no Markov head"
    if spec.draft_logits is not None:
        return "probabilistic drafts"
    if getattr(spec, "_draft_topk", None) is not None:
        return "top-k Markov sampling"
    if spec.acceptance_estimator is not None:
        return "the acceptance estimator"
    if spec.use_confidence_head:
        return "the confidence head"
    if spec._d2t_scatter_index is not None:
        return "a reduced draft vocabulary"
    if getattr(spec, "draft_watermarker", None) is not None:
        return "a draft watermarker"
    if getattr(spec.model, "has_recurrent_markov_head", False):
        return "a recurrent Markov head"
    if hasattr(spec.model, "apply_output_logit_transform"):
        return "an output logit transform"
    lp = spec.model.logits_processor
    if lp.scale != 1.0 or lp.soft_cap is not None:
        return "a scaled or soft-capped logits processor"
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return "Markov weights not bf16"
    if w1.shape != w2.shape or not (w1.is_contiguous() and w2.is_contiguous()):
        return "Markov weight layout"
    if w1.shape[1] not in (64, 128, 256):
        return f"Markov rank {w1.shape[1]}"
    if head.markov_w2.quant_method.__class__.__name__ != "UnquantizedEmbeddingMethod":
        return "a quantized Markov head"
    if getattr(head.markov_w2, "tp_size", 1) != 1:
        return "a sharded Markov head"
    return None


def enabled() -> bool:
    return envs.VLLM_DSPARK_FUSED_GREEDY_MARKOV


def sample_sequential(spec, num_reqs: int, head_hidden: torch.Tensor) -> bool:
    """Greedy `_sample_sequential` with the fused launches. Returns False,
    having done nothing, when the step takes vLLM's loop."""
    if not enabled():
        return False
    why = _why_not(spec)
    if why is not None:
        logger.info_once("DSpark fused greedy Markov off: %s", why)
        return False
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = spec.model.compute_draft_logits(sample_hidden)
    # The loop's bias has the head dtype (fp32 with an fp32 head_dtype, else
    # the model's bf16) and `logits + bias` runs in the promoted dtype.
    head_dtype = getattr(spec.model.logits_processor, "head_dtype", None)
    bias_dtype = torch.float32 if head_dtype == torch.float32 else torch.bfloat16
    if (
        base_logits.dtype not in (torch.float32, torch.bfloat16)
        or base_logits.stride(-1) != 1
    ):
        logger.info_once("DSpark fused greedy Markov off: %s logits", base_logits.dtype)
        return False
    bf16_sum = base_logits.dtype == torch.bfloat16 and bias_dtype == torch.bfloat16
    head = spec.model.model.markov_head
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    vocab, rank = w2.shape
    assert base_logits.shape[-1] == vocab
    base_logits = base_logits.view(num_reqs, n_spec, vocab)
    nb = triton.cdiv(vocab, BLOCK_V)
    key = (base_logits.device, vocab, rank)
    if key not in _buffers:
        rmax = spec.max_num_reqs
        _buffers[key] = (
            torch.empty(rmax, nb, dtype=torch.float32, device=base_logits.device),
            torch.empty(rmax, nb, dtype=torch.int32, device=base_logits.device),
            torch.empty(rmax, rank, dtype=torch.bfloat16, device=base_logits.device),
        )
        logger.info_once(
            "DSpark fused greedy Markov: %d steps, vocabulary %d, rank %d",
            n_spec,
            vocab,
            rank,
        )
    bmax, barg, emb = _buffers[key]
    tokens = spec.draft_tokens
    common = dict(RANK=rank, NB_POW2=triton.next_power_of_2(nb))
    _pick_kernel[(num_reqs,)](
        bmax, barg, nb, spec.input_buffers.input_ids, spec._anchor_idx, tokens,
        tokens.stride(0), w1, emb, STEP=-1, GATHER=True, num_warps=1, **common,
    )  # fmt: skip
    for step in range(n_spec):
        _step_kernel[(nb,)](
            emb, w2, base_logits[:, step], base_logits.stride(0), bmax, barg,
            num_reqs, vocab, nb, RANK=rank, BLOCK_V=BLOCK_V, BLOCK_R=BLOCK_R,
            BF16_BIAS=bias_dtype == torch.bfloat16, BF16_SUM=bf16_sum, num_warps=4,
        )  # fmt: skip
        _pick_kernel[(num_reqs,)](
            bmax, barg, nb, spec.input_buffers.input_ids, spec._anchor_idx, tokens,
            tokens.stride(0), w1, emb, STEP=step, GATHER=step + 1 < n_spec,
            num_warps=4, **common,
        )  # fmt: skip
    return True
