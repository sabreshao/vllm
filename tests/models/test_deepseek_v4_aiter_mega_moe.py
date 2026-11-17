# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_shared_expert_quant_fse_compatible,
)
from vllm.models.deepseek_v4.amd import mega_moe as mega_moe_module
from vllm.models.deepseek_v4.amd import model as amd_model_module
from vllm.models.deepseek_v4.amd.mega_moe import (
    DeepseekV4AiterMegaMoEExperts,
)
from vllm.models.deepseek_v4.amd.model import DeepseekV4MoE
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import aux_stream

HIDDEN = 64
INTER = 64


def _make_experts(
    monkeypatch,
    *,
    ep_rank=0,
    ep_size=1,
    num_experts=2,
    intermediate_size=INTER,
    max_num_batched_tokens=3000,
    swiglu_limit=None,
    fuse_shared_expert=False,
):
    monkeypatch.setattr(
        mega_moe_module,
        "get_ep_group",
        lambda: SimpleNamespace(
            rank_in_group=ep_rank, world_size=ep_size, cpu_group="ep"
        ),
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
        parallel_config=SimpleNamespace(enable_eplb=False),
    )
    return DeepseekV4AiterMegaMoEExperts(
        vllm_config,
        num_experts=num_experts,
        top_k=2,
        hidden_size=HIDDEN,
        intermediate_size=intermediate_size,
        swiglu_limit=swiglu_limit,
        fuse_shared_expert=fuse_shared_expert,
    )


@pytest.mark.cpu_test
def test_experts_load_into_local_slots(monkeypatch):
    model = torch.nn.Module()
    model.ffn = torch.nn.Module()
    model.ffn.experts = _make_experts(monkeypatch, ep_rank=1, ep_size=2, num_experts=4)
    params = dict(model.named_parameters())
    mapping = fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=4,
        routed_experts_prefix="",
    )

    shapes = {
        ("w1", "weight"): (INTER, HIDDEN // 2),
        ("w3", "weight"): (INTER, HIDDEN // 2),
        ("w2", "weight"): (HIDDEN, INTER // 2),
        ("w1", "weight_scale"): (INTER, HIDDEN // 32),
        ("w3", "weight_scale"): (INTER, HIDDEN // 32),
        ("w2", "weight_scale"): (HIDDEN, INTER // 32),
    }
    shard_value = {"w1": 1, "w3": 2, "w2": 3}
    loaded = {}
    for expert in range(4):
        for (shard, suffix), shape in shapes.items():
            name = f"ffn.experts.{expert}.{shard}.{suffix}"
            weight = torch.full(
                shape, 10 * expert + shard_value[shard], dtype=torch.uint8
            )
            for param_name, weight_name, expert_id, shard_id in mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                param = params[mapped]
                if param.weight_loader(
                    param,
                    weight,
                    mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                ):
                    loaded[name] = mapped
                    break

    assert sorted({n.split(".")[2] for n in loaded}) == ["2", "3"]
    assert len(loaded) == 2 * len(shapes)
    for slot, expert in enumerate((2, 3)):
        for suffix in ("weight", "weight_scale"):
            w13 = params[f"ffn.experts.w13_{suffix}"][slot]
            w2 = params[f"ffn.experts.w2_{suffix}"][slot]
            assert torch.all(w13[:INTER] == 10 * expert + 1)
            assert torch.all(w13[INTER:] == 10 * expert + 2)
            assert torch.all(w2 == 10 * expert + 3)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens_across_dp, expected",
    [([4, 128], "decode"), ([4, 129], "prefill")],
)
def test_runtime_selected_by_max_tokens_across_dp(
    monkeypatch, tokens_across_dp, expected
):
    experts = _make_experts(monkeypatch)
    experts._runtimes = [(128, "decode"), (4096, "prefill")]
    monkeypatch.setattr(mega_moe_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        mega_moe_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor(tokens_across_dp)
            )
        ),
    )
    # Every rank must take the same runtime, so a small local batch still gets
    # the prefill runtime when another DP rank is over the decode capacity.
    assert experts.select_runtime(4) == expected


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "moe_backend, targets, fused",
    [
        ("aiter_mega_moe", None, False),
        ("aiter_mega_moe", {"*shared_experts*": "mxfp8"}, False),
        ("aiter_mega_moe", {"*shared_experts*": "mxfp4"}, True),
        ("aiter", {"*shared_experts*": "mxfp4"}, False),
    ],
)
def test_mega_moe_shared_expert_fusion_requires_mxfp4_target(
    moe_backend, targets, fused
):
    quant_config = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
    )
    quant_config._resolved_expert_dtype = "fp4"
    quant_config.packed_modules_mapping = {"gate_up_proj": ["w1", "w3"]}
    if targets is not None:
        quant_config.online_quantization_config = OnlineQuantizationConfig(
            QuantizationConfigArgs(targets=targets)
        )
    compatible, _ = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.3.ffn.experts",
        "model.layers.3.ffn.shared_experts",
        moe_backend=moe_backend,
    )
    assert compatible is fused


@pytest.mark.cpu_test
@pytest.mark.parametrize("fuse_shared_expert", [False, True])
def test_eplb_weight_views_alias_routed_slots(monkeypatch, fuse_shared_expert):
    experts = _make_experts(
        monkeypatch, ep_size=2, num_experts=4, fuse_shared_expert=fuse_shared_expert
    )
    experts._runtimes = [(128, "runtime")]
    params = (
        experts.w13_weight,
        experts.w13_weight_scale,
        experts.w2_weight,
        experts.w2_weight_scale,
    )
    views = experts.get_expert_weights()
    assert len(views) == len(params)
    for view, param in zip(views, params):
        # One contiguous row per routed expert; the shared slot is not offered.
        assert view.shape == (2, param.data[0].numel())
        assert view.is_contiguous()
        # EPLB writes a moved expert into its row in place.
        view[1].fill_(7)
        assert (param.data[1] == 7).all() and (param.data[0] == 0).all()
        if fuse_shared_expert:
            assert (param.data[2] == 0).all()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "supports_eplb, num_redundant, elastic_ep, error",
    [
        (False, 0, False, "only for DeepSeek-V4.1"),
        (True, 8, False, "num_redundant_experts=0"),
        (True, 0, True, "elastic EP"),
        (True, 0, False, None),
    ],
)
def test_mega_moe_eplb_config_validation(
    monkeypatch, supports_eplb, num_redundant, elastic_ep, error
):
    monkeypatch.setattr(
        amd_model_module.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True
    )
    monkeypatch.setattr(amd_model_module, "on_gfx950", lambda: True)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            all2all_backend="mori",
            mori_ep_v2=False,
            enable_eplb=True,
            eplb_config=SimpleNamespace(num_redundant_experts=num_redundant),
            enable_elastic_ep=elastic_ep,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="fp4")),
    )
    validate = amd_model_module._validate_aiter_mega_moe_config
    if error is None:
        validate(vllm_config, supports_eplb)
    else:
        with pytest.raises(NotImplementedError, match=error):
            validate(vllm_config, supports_eplb)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
def test_eplb_maps_to_physical_and_records_unpadded_load(monkeypatch):
    num_experts, num_tokens, num_unpadded = 384, 37, 30
    experts = _make_experts(monkeypatch, ep_size=8, num_experts=num_experts)
    ids = torch.randint(0, num_experts, (num_tokens, 2), device="cuda")
    ids = ids.to(torch.int32)
    # Without EPLB the ids pass through.
    assert experts.map_to_physical(ids) is ids

    perm = torch.randperm(num_experts, device="cuda")
    load = torch.zeros(1, num_experts, dtype=torch.int32, device="cuda")
    experts.set_eplb_state(
        0,
        expert_load_view=load,
        logical_to_physical_map=perm.view(1, num_experts, 1),
        logical_replica_count=torch.ones(
            1, num_experts, dtype=torch.long, device="cuda"
        ),
    )
    experts.eplb_state.should_record_tensor = torch.ones(
        (), dtype=torch.bool, device="cuda"
    )
    experts.eplb_state.num_unpadded_tokens_tensors = [
        torch.tensor(num_unpadded, dtype=torch.int32, device="cuda")
    ]

    out = experts.map_to_physical(ids)
    assert out.dtype == torch.int32
    torch.testing.assert_close(out.long(), perm[ids.long()])
    # Load is recorded per physical expert, for the unpadded rows only.
    expected = torch.bincount(
        perm[ids[:num_unpadded].long()].flatten(), minlength=num_experts
    )
    torch.testing.assert_close(load[0].long(), expected)

    experts.eplb_state.should_record_tensor.fill_(False)
    experts.map_to_physical(ids)
    torch.testing.assert_close(load[0].long(), expected)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
def test_shared_expert_route_appended(monkeypatch):
    experts = _make_experts(
        monkeypatch, ep_rank=3, ep_size=8, num_experts=384, fuse_shared_expert=True
    )
    ids = torch.randint(0, 384, (37, 2), device="cuda", dtype=torch.int32)
    ids[::5, 1] = -1
    weights = torch.rand(37, 2, device="cuda")
    out_weights, out_ids = experts.append_shared_expert(weights, ids)

    shared = torch.full((37, 1), 3 * 49 + 48, device="cuda", dtype=torch.int32)
    expected_ids = torch.cat([torch.where(ids >= 0, ids + ids // 48, ids), shared], 1)
    torch.testing.assert_close(out_ids, expected_ids)
    torch.testing.assert_close(
        out_weights, torch.cat([weights, torch.ones(37, 1, device="cuda")], 1)
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
@pytest.mark.parametrize("hash_routing", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 37])
@pytest.mark.parametrize("shared_stream", [False, True])
def test_mega_moe_routing_matches_reference(
    monkeypatch, hash_routing, num_tokens, shared_stream
):
    num_experts, top_k, vocab = 16, 4, 64
    scaling = 2.5
    device = "cuda"
    torch.manual_seed(0)

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(num_experts, HIDDEN, device=device) * 0.1
            self.e_score_correction_bias = (
                None if hash_routing else torch.randn(num_experts, device=device)
            )
            self.tid2eid = (
                torch.stack(
                    [
                        torch.randperm(num_experts, device=device)[:top_k]
                        for _ in range(vocab)
                    ]
                ).to(torch.int32)
                if hash_routing
                else None
            )
            self.bias_vl = None

        def forward(self, x):
            return x.float() @ self.weight.t(), None

    captured = {}

    class Experts:
        def __call__(self, x, topk_weights, topk_ids):
            captured["weights"], captured["ids"] = topk_weights, topk_ids
            return 2 * x

        def uses_single_kernel(self, num_tokens):
            # Keep the shared expert on the aux stream when shared_stream is set.
            return False

    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.gate = Gate()
    moe.experts = Experts()
    moe.shared_experts = lambda x: 3 * x
    moe.shared_experts_stream = aux_stream() if shared_stream else None
    moe.shared_experts_events = (
        (torch.cuda.Event(), torch.cuda.Event()) if shared_stream else None
    )
    moe.scoring_func = "sqrtsoftplus"
    moe.n_activated_experts = top_k
    moe.renormalize = True
    moe.routed_scaling_factor = scaling
    moe.image_sentinel_lo = 0

    x = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, vocab, (num_tokens,), device=device)
    out = moe._forward_mega_moe(x, input_ids)

    ids, weights = captured["ids"], captured["weights"]
    assert ids.dtype == torch.int32 and ids.is_contiguous()
    assert weights.dtype == torch.float32 and weights.is_contiguous()

    scores = torch.nn.functional.softplus(x.float() @ moe.gate.weight.t()).sqrt()
    if hash_routing:
        expected_ids = moe.gate.tid2eid[input_ids].long()
    else:
        expected_ids = (scores + moe.gate.e_score_correction_bias).topk(top_k).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights *= scaling / expected_weights.sum(dim=-1, keepdim=True)

    order = ids.long().sort(dim=-1).indices
    expected_order = expected_ids.sort(dim=-1).indices
    torch.testing.assert_close(
        ids.long().gather(1, order), expected_ids.gather(1, expected_order)
    )
    torch.testing.assert_close(
        weights.gather(1, order),
        expected_weights.gather(1, expected_order),
        rtol=1e-3,
        atol=1e-4,
    )
    torch.testing.assert_close(out, 5 * x)


@pytest.fixture
def mori_single_rank():
    """A one-rank process group with MoRI shmem, as MegaMoEV2 needs."""
    import torch.distributed as dist

    mori_shmem = pytest.importorskip("mori.shmem")
    from vllm.utils.network_utils import get_open_port

    device = torch.device("cuda", torch.accelerator.current_device_index())
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{get_open_port()}",
        rank=0,
        world_size=1,
        device_id=device,
    )
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    mori_shmem.shmem_torch_process_group_init("default")
    yield dist.group.WORLD
    mori_shmem.shmem_finalize()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not (current_platform.is_rocm() and amd_model_module.on_gfx950()),
    reason="Requires ROCm gfx950",
)
def test_eplb_inplace_expert_move_keeps_cuda_graph_output(
    monkeypatch, mori_single_rank
):
    """EPLB moves experts by copying their weight rows in place. MegaMoEV2 must
    read the moved weights on CUDA graph replay (no cached per-expert copy):
    after permuting the experts and routing to their new slots, the replayed
    output must match the one before the move."""
    aiter = pytest.importorskip("aiter")
    from aiter import dtypes

    # DeepSeek-V4.1-Flash expert shape; 48 experts as one EP8 rank holds.
    hidden, inter, num_experts, top_k, num_tokens = 5120, 2304, 48, 6, 96
    monkeypatch.setattr(
        mega_moe_module,
        "get_ep_group",
        lambda: SimpleNamespace(
            rank_in_group=0, world_size=1, cpu_group=mori_single_rank
        ),
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        parallel_config=SimpleNamespace(enable_eplb=True),
    )
    experts = DeepseekV4AiterMegaMoEExperts(
        vllm_config,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden,
        intermediate_size=inter,
        swiglu_limit=10.0,
    ).cuda()

    generator = torch.Generator(device="cuda").manual_seed(0)
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    for weight, scale, rows, cols in (
        (experts.w13_weight, experts.w13_weight_scale, 2 * inter, hidden),
        (experts.w2_weight, experts.w2_weight_scale, hidden, inter),
    ):
        for e in range(num_experts):
            w = torch.randn(
                rows, cols, dtype=torch.bfloat16, device="cuda", generator=generator
            )
            q, s = quantize(w * cols**-0.25, quant_dtype=dtypes.fp4x2)
            weight.data[e].copy_(q.view(torch.uint8).view(rows, cols // 2))
            scale.data[e].copy_(s.view(torch.uint8).view(rows, cols // 32))
    experts.finalize_weights()

    x = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    scores = torch.randn(num_tokens, num_experts, device="cuda", generator=generator)
    topk_weights, logical_ids = scores.softmax(dim=-1).topk(top_k, dim=-1)
    topk_weights = topk_weights.contiguous()
    logical_ids = logical_ids.to(torch.int32)
    ids = logical_ids.clone()

    def run():
        return experts.forward(x, topk_weights, ids, topk_ids_physical=True)

    eager_before = run().clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    graph.replay()
    torch.cuda.synchronize()
    before = out.clone()
    torch.testing.assert_close(before, eager_before)

    # Physical slot p now holds logical expert perm[p], as EPLB would leave it.
    perm = torch.randperm(num_experts, device="cuda", generator=generator)
    for rows in experts.get_expert_weights():
        rows.copy_(rows[perm].clone())
    slot_of = torch.empty_like(perm)
    slot_of[perm] = torch.arange(num_experts, device="cuda")

    # Stale ids now hit other experts: the check below can tell a move apart.
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.allclose(out, before)

    ids.copy_(slot_of[logical_ids.long()].to(torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, before)
    torch.testing.assert_close(run(), before)
