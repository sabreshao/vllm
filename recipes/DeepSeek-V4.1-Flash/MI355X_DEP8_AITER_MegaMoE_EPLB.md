# DeepSeek-V4.1-Flash on 8× MI355X: DP8/EP8 with AITER MegaMoE and EPLB

```bash
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_MORI_EP_V2=0 MORI_SHMEM_HEAP_SIZE=48G \
       VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True AITER_TRITON_LOG_LEVEL=ERROR OMP_NUM_THREADS=1 \
       VLLM_ENGINE_READY_TIMEOUT_S=3600

vllm serve deepseek-ai/DeepSeek-V4.1-Flash --served-model-name dsv41 --port 8000 \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --moe-backend aiter_mega_moe --all2all-backend mori \
  --engram-config '{"cpu_offload":false,"embedding_across_dp":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --attention-config '{"indexer_kv_dtype":"mxfp4"}' --block-size 128 \
  --max-num-batched-tokens 16384 --max-model-len 131072 --max-cudagraph-capture-size 1024 \
  --gpu-memory-utilization 0.85 --language-model-only \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --enable-eplb --eplb-config '{"window_size":1000,"step_interval":2000}'
```


# DeepSeek-V4.1-Flash on 8× MI355X: DP8/EP8 with AITER single megamoe kernel

打 [`patches/aiter-0.1.29-mega_moe-dsv41-flash.patch`](../../docs/patches/aiter-0.1.29-mega_moe-dsv41-flash.patch)（一个补丁，基线 amd-aiter 0.1.29，只改 `aiter/` 下的文件，补丁头列出了所含 commit）。它包含原先分开提供的三个补丁（下表，`0001` 由 V4.1-Flash 形状 commit 等价覆盖），另含 prefill large_m 所需的改动（见 [megamoe_large_m_dsv41f.md](../../docs/megamoe_large_m_dsv41f.md)）：

| 原补丁 | 内容 | 没有这部分时 |
|---|---|---|
| `0001-...-dsv41-flash.patch` | `small_m_eligible` 允许 5120 / 2304；每卡 ≤ 8 token 的档位在 V4.1 上会触发 `assert N_TILES % 4 == 0`，改为退回 `tile_n=128` | V4.1 不走单 kernel |
| `0002-...-any-experts-shared-slot.patch` | 每卡专家数从只允许 48 个放宽到 1～64 个，topk 放宽到 1～8，所以草稿 MoE（128 个专家、top-3）和融合共享专家后的布局（392 个专家、top-7）都能走单 kernel；新增 `SmallMRuntime(shared_experts=N)`，共享专家槽不参与热点专家和 FP8 combine 行的判断 | 草稿 MoE 和融合配置仍走多段 kernel |
| `0003-...-gemm-pipeline.patch` | GEMM2：K 循环完全展开，下一 K 步的 W2 提前发出，基址改为 uniform 以去掉 waterfall 循环；GEMM1：每卡 ≥256 token 的档位用 `tile_n=256` | 单 kernel 慢约 6% |

```bash
docker run -d --name dsv41f --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v <MODEL_DIR>:/models/DeepSeek-V4.1-Flash:ro \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_USE_RUST_FRONTEND=1 \
  -e MORI_SHMEM_HEAP_SIZE=16G -e OMP_NUM_THREADS=1 \
  --entrypoint sleep <IMAGE> infinity

docker exec -d dsv41f vllm serve /models/DeepSeek-V4.1-Flash \
  --host 0.0.0.0 --port 8000 --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --data-parallel-size 8 --tensor-parallel-size 1 --enable-expert-parallel \
  --all2all-backend mori_high_throughput --moe-backend aiter_mega_moe \
  --language-model-only --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --gpu-memory-utilization 0.9 \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --block-size 128 --max-model-len 131072 --no-swa-bounded-replay \
  --engram-config '{"cpu_offload":true}' \
  --max-cudagraph-capture-size 384 --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
```

