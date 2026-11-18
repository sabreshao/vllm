# DeepSeek-V4.1-Flash：decode 走 AITER 单 kernel MegaMoE（small_m）

2026-11-17，8× MI355X，DP8+EP8，`--moe-backend aiter_mega_moe`，DSpark nst5。

## 改动

decode 时（DP 间每卡最大 token 数 ≤ 512），MegaMoE 改走 AITER 的单 kernel 版本 `SmallMRuntime`：量化、dispatch、GEMM1、GEMM2、combine 在一次 launch 里完成。更大的 batch（prefill）仍走 `MegaMoEV2` 的多段 kernel。主模型的 MoE 和 DSpark 草稿模型的 MoE 都走这条路径。

- `vllm/models/deepseek_v4/amd/mega_moe.py`
  - `finalize_weights` 之后，另外构造容量 256 和 512 的 `SmallMRuntime`，所有层共享。每层 W2 的 dispatch 表在 CUDA graph 捕获前就建好（单 kernel 把 W2 指针放在这张表里，捕获期间不能新建）。
  - `forward` 按 DP 间最大 token 数选路径，所有 EP rank 走同一条路径。换层用 `bind_weights`。开启 EPLB 时，先把 topk id 映射成物理专家 id（`map_to_physical`），再选路径，所以单 kernel 和多段 kernel 收到的都是物理 id。
  - 融合共享专家时，用 `SmallMRuntime(shared_experts=1)` 告诉 AITER 每卡最后一个专家槽是共享专家。
  - 新增 `uses_single_kernel(num_tokens)`。
- `vllm/models/deepseek_v4/amd/model.py`：用单 kernel 时，共享专家不放到辅助流上，改为串行执行（见"已知问题"）。
- `tests/models/test_deepseek_v4_aiter_mega_moe.py`：`test_mega_moe_routing_matches_reference` 里替身 experts 补上 `uses_single_kernel`（返回 False），保持原来对辅助流路径的测试。
- `vllm/envs.py`：新增 `VLLM_ROCM_AITER_MEGA_MOE_SINGLE_KERNEL`，默认 1，设为 0 回到多段 kernel。

草稿模型不需要改代码：它和主模型用同一个 `DeepseekV4AiterMegaMoEExperts`，AITER 接受它的形状后就自动走单 kernel。

以下情况自动回退到多段 kernel，行为和改动前一样：
- AITER 没有单 kernel（< 0.1.29，例如本分支镜像自带的 0.1.28.post1）；
- AITER 的 `small_m_eligible` 不接受当前形状（例如没打下面的补丁）。

## 依赖：AITER 0.1.29 加三个补丁

按顺序打 [`patches/`](./patches/) 下的三个补丁，基线是 amd-aiter 0.1.29，补丁都只改 `aiter/ops/flydsl/kernels/mega_moe/` 下的文件：

| 补丁 | 内容 | 不打时 |
|---|---|---|
| `0001-...-dsv41-flash.patch` | `small_m_eligible` 允许 5120 / 2304；每卡 ≤ 8 token 的档位在 V4.1 上会触发 `assert N_TILES % 4 == 0`，改为退回 `tile_n=128` | V4.1 不走单 kernel |
| `0002-...-any-experts-shared-slot.patch` | 每卡专家数从只允许 48 个放宽到 1～64 个，topk 放宽到 1～8，所以草稿 MoE（128 个专家、top-3）和融合共享专家后的布局（392 个专家、top-7）都能走单 kernel；新增 `SmallMRuntime(shared_experts=N)`，共享专家槽不参与热点专家和 FP8 combine 行的判断 | 草稿 MoE 和融合配置仍走多段 kernel |
| `0003-...-gemm-pipeline.patch` | GEMM2：K 循环完全展开，下一 K 步的 W2 提前发出，基址改为 uniform 以去掉 waterfall 循环；GEMM1：每卡 ≥256 token 的档位用 `tile_n=256` | 单 kernel 慢约 6% |

- 补丁头里写了具体改动、实测和代价。
- 三个补丁都不改变主模型形状的计算结果：在均匀、zipf 和真实 decode 路由上，输出都与只打 0001 时逐位相同。
- V4.1 不需要单独的调优配置：单 kernel 的配置规则与形状无关，所有可调项和阈值扫描下来，默认值都在最优的 1% 以内。

## 实测

8× MI355X，镜像 `vllm/vllm-openai-rocm:nightly-9bc16ae6a34b1e0fe260e8d12b98568478bf38c3`（AITER 0.1.29）。在镜像的 vLLM 文件上打上本分支相对 `978aaf0f` 的全部 Python 改动（包括 #1 的 EPLB 和本 PR），再覆盖打完补丁的 AITER 文件。`tests/models/test_deepseek_v4_aiter_mega_moe.py` 在同样的环境里 24 个用例全部通过。

测法：真实文本 prompt，512 个请求一次性并发（每卡 64），temperature 0，`ignore_eos`，输出 8K。ITL 是同一请求相邻两个流式 chunk 的间隔（即每步间隔），取纯 decode 窗口内的值，2 次运行平均。单卡吞吐 = 512 × 前 1K 输出的接受长度 / ITL / 8。

| | 1K ITL 平均 / p99 (ms) | 1K 单卡 tok/s | 100K ITL 平均 / p99 (ms) | 100K 单卡 tok/s | gsm8k（1319 题） |
|---|---|---|---|---|---|
| 镜像 MegaMoE（改动前） | 38.92 / 49.42 | 6,026 | 40.35 / 52.75 | 5,805 | 0.907 |
| 本 PR，只打 0001 | 33.67 / 43.67 | 6,927（+15.0%） | 35.05 / 46.45 | 6,637（+14.3%） | 0.909 |
| **本 PR，0001～0003（默认）** | **32.41 / 42.73** | **7,237（+20.1%）** | **33.67 / 44.50** | **6,909（+19.0%）** | 0.903 |
| 本 PR，0001～0003 + 融合共享专家 | 30.99 / 40.24 | 7,466（+23.9%） | 32.28 / 42.45 | 7,057（+21.6%） | 0.895 |
| 本 PR，AITER 不打补丁（回退） | 38.77（1 次） | — | — | — | — |

- 默认配置下 ITL 降 16.7%（1K）/ 16.6%（100K）；加上融合共享专家后降 20.4% / 20.0%。
- 同一配置两次运行之间 ITL 相差 ≤ 0.02 ms。smoke 测试（答案 215）都正确，所有请求 512/0 成功。
- rebase 到 #1（EPLB）之前，同样的测试结果是：默认配置 32.67 / 33.62 ms，融合配置 31.00 / 32.30 ms，和上表一致。
- gsm8k 0.895～0.909 之间的差别都在标准误范围内：每个值约 0.8 个百分点，两个值之差约 1.2 个百分点。
- 融合共享专家时，前 1K 接受长度是 3.62 / 3.56（默认配置是 3.67 / 3.63），所以单卡吞吐只多 3.2% / 2.1%，见"已知问题"。

**叠加在其他优化上**（zovonoir 的 dsv41f-dp8ep8-opt 补丁 + dense GEMM 优化 + `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=1024`）：

| | 1K ITL | 1K 单卡 tok/s | 100K ITL | 100K 单卡 tok/s |
|---|---|---|---|---|
| 不用单 kernel | 36.62 | 6,362 | 37.99 | 6,124 |
| 只打 0001 | 32.44 | 7,192 | 33.79 | 6,904 |
| 0001～0003 | 31.13 | 7,544 | 32.41 | 7,227 |
| 0001～0003 + 融合 | 30.10 | 7,708 | 31.29 | 7,293 |

0001～0003 两行的数据重测过一次，ITL 相差 ≤ 0.5%。

**单 kernel 本身**（8 卡单测，随机 FP4 权重，CUDA graph 计时；真实路由是从服务里抓取的 decode 路由，主模型每卡 384 token，草稿每卡 320 token）：

| 形状 | 多段 kernel µs | 单 kernel µs（只打 0001 / 加 0002 / 0001～0003） |
|---|---|---|
| 主模型 384 专家 / top-6，均匀路由 | 447 | 282 / 280 / 277 |
| 主模型 384 专家 / top-6，真实路由 | — | 372 / 373 / 349～354 |
| 融合共享专家 392 / top-7，真实路由 | 557 | — / 388 / 363 |
| 草稿 128 / top-3，真实路由 | 378 | — / 229 / 222 |

- 和 FP32 参考对比，单 kernel 误差 0.25%，多段 kernel 2.66%。原因是多段 kernel 在容量 16384 时把 GEMM2 的结果按 FP8 传回 combine，单 kernel 传的是 BF16。
- 同样输入跑两次，结果逐位相同。每卡 0～512 token、各 rank token 数不同、全部 rank 空闲等情况都测过。
- 融合后，共享专家在单 kernel 里只多 10～15 µs；不融合时串行执行要 48 µs。
- 模型 trace（1K，典型层，只打 0001）：MegaMoE 每层从 490 µs 降到 374 µs。加 0002 后，草稿 MoE 从约 1,200 µs/步（3 次调用）降到 701 µs/步。

## 启动命令

镜像要求：包含本 PR 的 vLLM，AITER 0.1.29 并打上 `0001`～`0003`。

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

- 启动日志里出现 `AITER MegaMoE: batches of up to 512 tokens per rank use the single-kernel path`，说明单 kernel 已启用。主模型和草稿模型共用这一行日志。
- **融合共享专家**（可选，默认关闭）：
  - 在 `docker run` 里加 `-e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`；
  - 在 `vllm serve` 里加 `--quantization-config '{"targets":{"*shared_experts*":"mxfp4"}}'`。
  - 共享专家会从 FP8 重新量化成 MXFP4，作为每卡第 49 个专家进入单 kernel。

## 已知问题

- **不能和辅助流上的共享专家同时运行。** 单 kernel 要求整个 grid 同时驻留在 GPU 上（每 CU 2 个 CTA）。共享专家在辅助流上并行时，第一次 CUDA graph 捕获就会挂死：GPU 100%，没有报错。所以用单 kernel 时共享专家改为串行。单 kernel 本来就占满所有 CU，并行也没有收益。具体是哪个 kernel 形成了循环等待，还没有查清。
- **融合共享专家会改变数值。**
  - 原因：共享专家从 FP8 重新量化成 MXFP4。
  - 影响：前 1K 接受长度低 0～3%（1K / 100K 各 4 组对照，每组 2 次运行），抵掉了一部分 ITL 收益。在多段 kernel 上融合也有同样的现象。
  - gsm8k 没有看出差别。
  - 所以融合默认关闭。
- **0003 有寄存器溢出**：28 个 VGPR、16 个 SGPR，都在每个 tile 的前后处理里，K 循环里没有。没有测过 V4-Pro 形状打 0003 后的性能。
- **EPLB（#1）和单 kernel 一起用还没有端到端测过。**
  - 原因：开启 EPLB 时，路由要调用 #1 新增参数的 `_moe_C::topk_softplus_sqrt`，测试用的镜像里编译的 `_moe_C` 没有这些参数，服务启动就报错。不开 EPLB 不受影响。
  - 代码上，单 kernel 收到的已经是物理 id。EPLB 只在原地搬动权重，单 kernel 每次调用时按指针读权重（`bind_weights`），W2 dispatch 表也只存指针，所以搬动之后 CUDA graph 仍然有效。
  - 之前在另一个 EPLB 移植版本上，单 kernel + EPLB 实测能正常运行，但基本没有收益：p50 只降 0.2 ms。原因是不加冗余专家时，EPLB 拆不开单个热点专家。
  - 打 0002 之后，单 kernel 支持每卡 48 个以上的专家，技术上可以加冗余专家了，但 #1 目前拒绝冗余专家。
  - 用抓到的路由离线推算：8 个冗余专家时，最忙 rank 的 m-tile 从 76.4 降到 67.1，约省 30 µs/层。vLLM 默认的放置算法平衡的是行数，不是 64 行的 tile 数，再加更多冗余专家也没有进一步收益。

## 下一步

| 方向 | 依据 |
|---|---|
| 冗余专家 + 按 tile 数平衡的放置 | 离线推算约省 30 µs/层 |
| 共享专家以 FP8 权重融合进单 kernel（为共享专家槽单独走一条 FP8×FP8 的 GEMM 路径；B300 的 `sm100_fp8_fp4_mega_moe_impl` 用 `kNumSharedExperts=1` 就是这种做法） | 去掉融合带来的接受长度下降 |
| GEMM1 的 A 加载（24 µs）和 GEMM2 的 scatter epilogue（28 µs） | 真实路由下逐项去掉某一部分的测量 |
| dispatch、规划、combine 等不在 GEMM 里的开销，约 150 µs | 按每卡 m-tile 数线性拟合的截距 |
| 把 AITER 补丁提交到 ROCm/aiter | 去掉对本地补丁的依赖 |
| 查清和辅助流同时运行时挂死的原因 | 现在只是靠串行绕过 |

对比 B300（同一个 vLLM commit，DeepGEMM MegaMoE，已融合共享专家）：1K ITL 20.30 ms，单卡 11,413 tok/s。MI355X 镜像加本 PR 的默认配置是 32.41 ms、7,237 tok/s，B300 领先 1.58 倍；叠加其他优化后是 1.51 倍。
