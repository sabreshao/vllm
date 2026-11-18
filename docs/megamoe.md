# aiter MegaMoE（gfx950 / 相关变体）

本地笔记：对照 [ROCm/aiter](https://github.com/ROCm/aiter) 源码（`aiter/ops/flydsl/kernels/mega_moe*`）。重点是 **gfx950 上 EP 用的 `MegaMoEV2`**：逻辑上常说 prepare / stage1 / stage2 / combine，但 **不是一律四个独立 kernel launch**；其他模型多半共用同一套多段 kernel 族，差在 `NetworkProfile` 与是否启用 Pro 专属路径。

## 家族一览

| 入口 | 目标 | 说明 |
|---|---|---|
| **`MegaMoEV2`** | gfx950 EP（单节点） | 主路径：多段 FlyDSL（prepare / stage1 / stage2，combine 常融进 stage2） |
| **`MegaMoEInternode`** | 跨节点 EP（如 EP16） | 在 MegaMoEV2 协议上扩展；stage2 复用 GEMM2 core |
| **`MegaMoeTP` / `MegaMoeTPEngine`** | gfx950 **TP** MoE | **另一套**：单 launch（+ big-M）做 AG/RS 或 AR + GEMM1/2，不是下面四个 EP stage |
| **`MegaMoEGfx1250`** | gfx1250 | 独立实现，不走 gfx950 MegaMoEV2 四段 |

公开 import（懒加载）：

- `aiter.ops.flydsl.kernels.mega_moe` → `MegaMoEV2`, `MegaMoEConfig`, `select_mega_moe_config`, …
- `aiter.ops.flydsl.kernels.mega_moe_internode` → `MegaMoEInternode`
- `aiter.ops.flydsl.kernels.mega_moe_gfx1250` → `MegaMoEGfx1250`
- `aiter.ops.flydsl.mega_moe_tp` → TP 封装

源码目录：

```
aiter/ops/flydsl/kernels/mega_moe/          # MegaMoEV2 + prepare/stage1/stage2 + large_m/small_m
aiter/ops/flydsl/kernels/mega_moe_internode/
aiter/ops/flydsl/kernels/mega_moe_tp/
aiter/ops/flydsl/kernels/mega_moe_gfx1250/
```

量化：`MegaMoEV2` 仅 **`a4w4` / `a8w4`**。依赖 FlyDSL + MORI shmem。

## gfx950 MegaMoEV2：逻辑阶段 vs 实际 launch

逻辑上四段：

| 阶段 | 作用 |
|---|---|
| **prepare** | compact 路径：路由整理 +（常含）activation quant；fixed-slot **不走**独立 prepare |
| **stage1** | peer dispatch + W13 GEMM1 + SwiGLU/SiTUv2 + 中间量化 |
| **stage2** | W2 GEMM2 + 加权 peer combine（多数配置下 **combine 融在 stage2**） |
| **combine** | 名义第四段；独立 launch 少见，多由 stage2 / small_m / large_m 一并做完 |

`forward` 调度要点（`mega_moe_v2.py`）：

1. 若 **large_m 打开且 eligible** → 整段交给 `LargeMRuntime`（见下）
2. **fixed-slot**：host `quantize` → `_run_joint`（stage1 + stage2），**无** `_run_compact_prepare`
3. **compact**：`_run_compact_prepare`（可带 `quant_input`）→ `_run_joint(..., prepared=True)`

因此「四个 kernel」只是命名习惯；实际常见是 **2～3 次 launch**（compact 有 prepare 时偏 3）。

### Dispatch 模式

| 模式 | 何时 | 典型场景 | launch 大致 |
|---|---|---|---|
| **fixed-slot** | `mtpr ≤ FIXED_SLOT_MAX_MTPR`（默认 env `AITER_MEGA_FIXED_SLOT_MAX_MTPR=255`）且 epr 等条件满足 | decode / 小 MTPR | quant（host）+ stage1 + stage2 |
| **compact** | 更大 MTPR | prefill | prepare + stage1 + stage2 |

`TOKEN_BUCKETS`：1…32768（2 的幂档）。配置按 bucket 选 `Stage1Config` / `Stage2Config`。

### Pro 专属：small_m / large_m（更少 kernel）

仅 **DeepSeek-V4-Pro 形状**：`H=7168, I=3072, E=384, topk=6, EP8, a8w4, silu, swiglu_limit=10`，且无 `fanout_masks`。Flash（5120×2304）在 aiter 0.1.29 原版里**进不了** small_m。依次打上 [patches/](./patches/) 里的 `0001`～`0003` 三个补丁后：Flash 可用；每卡专家数放宽到 1～64、topk 放宽到 1～8，草稿 MoE 和融合共享专家的布局都能用；GEMM 也做了优化。vLLM 接线和实测见 [megamoe_single_kernel_dsv41f.md](./megamoe_single_kernel_dsv41f.md)。

| 路径 | 开关 | 条件补充 | Kernel 数 |
|---|---|---|---|
| **`small_m`** | `AITER_MEGA_SMALL_M=1` | gfx950、**256 CU**、每 rank ≤512 tok（默认 runtime 256/512） | **1**（quant+dispatch+GEMM1+GEMM2+combine） |
| **`large_m`** | `AITER_MEGA_LARGE_M=1`（或 ctor `large_m_enabled`） | `max_tok_per_rank==8192` | **2**：单独 quant + **一条** persistent（dispatch…combine） |

调用：`forward_small_m(...)`；large_m 走普通 `forward`（`set_large_m_enabled`）。small/large 与主路径 **不共享协议状态**，各 EP rank 必须选同一条。

## 注册网络（`NetworkProfile`）

按 **`(experts_per_rank, model_dim, inter_dim)`** 匹配；未命中用 **base config**（无专调表）。

| name | epr | H × I | 备注 |
|---|---|---|---|
| `r1` | 32 | 7168 × 2048 | `fp4_prelude` |
| `v4_pro` | 48（EPLB 族 52/56） | 7168 × 3072 | 可开 small_m / large_m；`fp4_prelude` |
| `kimi_k3` | 112 | 3584 × 3072 | act 常为 **situv2**；FP4/FP8 均可调 |
| `v41_flash_ep2` | 192 | 5120 × 2304 | Flash EP2；`fp4_prelude` |
| `v41_flash_ep8` | 48 | 5120 × 2304 | Flash EP8 |

模型差异：**同一 MegaMoEV2 多段 kernel 族**，差在 tuner（`_tune_r1` / `_tune_v4_pro` / `_tune_kimi_k3` / `_tune_v41_flash_*`）、形状、topk/act、是否 Pro 单/双 kernel 路径——**不是**另写一套「四分法」。

### DeepSeek-V4.1-Flash

- 形状：H=5120，I=2304，E=384，topk=6；EP2 → epr=192，EP8 → epr=48。
- prefill 走默认多段（prepare/stage1/stage2）。decode（每卡 ≤512 tok）在 aiter ≥ 0.1.29 并打上补丁后走 small_m 单 kernel，DSpark 草稿 MoE（128 专家、top-3）也一样（[megamoe_single_kernel_dsv41f.md](./megamoe_single_kernel_dsv41f.md)）；large_m 仍不可用。
- 相关 aiter PR：[aiter #6154](https://github.com/ROCm/aiter/pull/6154)（full-expert Flash MegaMoE tuning，draft）；vLLM 接线见 [#59685](https://github.com/vllm-project/vllm/pull/59685) 等。更广的 Flash PR 表见同目录 [dsv4.1f.md](./dsv4.1f.md)。

## 调参

选择入口：`select_mega_moe_config` / `build_mega_moe_bundle_plan`（`mega_moe_config.py`）。命中 profile 后跑对应 `_tune_*`。

主要旋钮：

| | 字段（示例） |
|---|---|
| **Stage1** | `sort_block_m`, `tile_n`/`tile_k`, `num_waves`, `grid_mult`, `num_dispatch_cu`, `async_a_copy`, `mfma_amajor`, `b_nt`, `prepare_quant_cu`, … |
| **Stage2** | `block_m`/`block_n`/`block_k`, `persist`/`persist_cu`, `use_nt`, `aligned_pair`, P2P quant（`none` / `fp8_blockwise_1x32` / `auto`）, … |

Benchmark：

- `op_tests/multigpu_tests/bench_mega_moe_v2.py`
- `op_tests/multigpu_tests/bench_mega_moe.py`
- 正确性：`test_mega_moe_v2.py`、`test_mega_moe_small_m.py` 等

加新网络：实现 `_tune_<name>(config, ctx)`，往 `_NETWORKS` 加一行 `NetworkProfile`。

## 环境变量（`mega_moe/envs.py`）

| 变量 | 默认 | 作用 |
|---|---|---|
| `AITER_MEGA_FIXED_SLOT_MAX_MTPR` | 255 | fixed-slot 上限 MTPR |
| `AITER_MEGA_COMBINE_MASK` | 0 | combine 跳过 top-k id `-1` |
| `AITER_MEGA_MOE_PRELOAD` | 0 | 构造时 preload AOT bundle |
| `AITER_MEGA_S1_EPI_EVEC` | 8 | stage1 epilogue 向量宽度 |
| `AITER_MEGA_S1_FIXED_KPAIR` | 1 | fixed-slot K-pair |
| `AITER_MEGA_DISPATCH_FAST_COPY` | 1 | compact dispatch fast copy |
| `AITER_MEGA_SMALL_M` | 0 | 启用 Pro small_m |
| `AITER_MEGA_LARGE_M` | 0 | 启用 Pro large_m |

## API 速查（`MegaMoEV2`）

```text
MegaMoEV2(rank, world_size, model_dim, inter_dim, experts, topk,
          quant, w1, w1_scale, w2, w2_scale, max_tok_per_rank, ...)

forward(x_bf16, wts, topk_ids, ...)           # 主入口；可能转 large_m
forward_prequant(x_q, scales, wts, topk_ids)  # 已量化 activation
forward_small_m(...)                          # 仅 small_m
quantize / bind_weights / bind_layer
preload_aot_bundles / set_large_m_enabled
```

## 和「单 kernel 融合全部」的关系

- **默认 EP MegaMoEV2**：多 launch（prepare/stage1/stage2），**不是** dispatch+gemm1+gemm2+combine 一条。
- **Pro `small_m`**：最接近全融合单 kernel。
- **Pro `large_m`**：quant 外再一条 persistent。
- **`MegaMoeTP`**：TP 场景下的另一套单/双路径融合，与 EP 四段无关。

## 一句话结论

gfx950 MegaMoE **不都是四个 kernel**。Flash / R1 / Kimi-K3 / 多数 Pro 共用 **MegaMoEV2 多段**（实际常 2～3 launch）；只有 **V4-Pro + env** 才走 1～2 kernel 的 small/large_m；模型间主要差 **NetworkProfile 与形状**，不是换另一套四分法。

## DeepSeek-V4.1-Flash：decode 走 AITER 单 kernel MegaMoE（small_m）

### 依赖：AITER 0.1.29 加三个补丁

按顺序打 [`patches/`](./patches/) 下的三个补丁，基线是 amd-aiter 0.1.29，补丁都只改 `aiter/ops/flydsl/kernels/mega_moe/` 下的文件：

| 补丁 | 内容 | 不打时 |
|---|---|---|
| `0001-...-dsv41-flash.patch` | `small_m_eligible` 允许 5120 / 2304；每卡 ≤ 8 token 的档位在 V4.1 上会触发 `assert N_TILES % 4 == 0`，改为退回 `tile_n=128` | V4.1 不走单 kernel |
| `0002-...-any-experts-shared-slot.patch` | 每卡专家数从只允许 48 个放宽到 1～64 个，topk 放宽到 1～8，所以草稿 MoE（128 个专家、top-3）和融合共享专家后的布局（392 个专家、top-7）都能走单 kernel；新增 `SmallMRuntime(shared_experts=N)`，共享专家槽不参与热点专家和 FP8 combine 行的判断 | 草稿 MoE 和融合配置仍走多段 kernel |
| `0003-...-gemm-pipeline.patch` | GEMM2：K 循环完全展开，下一 K 步的 W2 提前发出，基址改为 uniform 以去掉 waterfall 循环；GEMM1：每卡 ≥256 token 的档位用 `tile_n=256` | 单 kernel 慢约 6% |

- 补丁头里写了具体改动、实测和代价。
- 三个补丁都不改变主模型形状的计算结果：在均匀、zipf 和真实 decode 路由上，输出都与只打 0001 时逐位相同。
- V4.1 不需要单独的调优配置：单 kernel 的配置规则与形状无关，所有可调项和阈值扫描下来，默认值都在最优的 1% 以内。

- decode 时（DP 间每卡最大 token 数 ≤ 512），MegaMoE 改走 AITER 的单 kernel 版本 `SmallMRuntime`：量化、dispatch、GEMM1、GEMM2、combine 在一次 launch 里完成。更大的 batch（prefill）仍走 `MegaMoEV2` 的多段 kernel。主模型的 MoE 和 DSpark 草稿模型的 MoE 都走这条路径。

- `vllm/models/deepseek_v4/amd/mega_moe.py`
  - `finalize_weights` 之后，另外构造容量 256 和 512 的 `SmallMRuntime`，所有层共享。每层 W2 的 dispatch 表在 CUDA graph 捕获前就建好（单 kernel 把 W2 指针放在这张表里，捕获期间不能新建）。
  - `forward` 按 DP 间最大 token 数选路径，所有 EP rank 走同一条路径。换层用 `bind_weights`。开启 EPLB 时，先把 topk id 映射成物理专家 id（`map_to_physical`），再选路径，所以单 kernel 和多段 kernel 收到的都是物理 id。
  - 融合共享专家时，用 `SmallMRuntime(shared_experts=1)` 告诉 AITER 每卡最后一个专家槽是共享专家。
  - 新增 `uses_single_kernel(num_tokens)`。
- `vllm/models/deepseek_v4/amd/model.py`：用单 kernel 时，共享专家不放到辅助流上，改为串行执行（见"已知问题"）。
- `tests/models/test_deepseek_v4_aiter_mega_moe.py`：`test_mega_moe_routing_matches_reference` 里替身 experts 补上 `uses_single_kernel`（返回 False），保持原来对辅助流路径的测试。
- `vllm/envs.py`：新增 `VLLM_ROCM_AITER_MEGA_MOE_SINGLE_KERNEL`，默认 1，设为 0 回到多段 kernel。

- 草稿模型不需要改代码：它和主模型用同一个 `DeepseekV4AiterMegaMoEExperts`，AITER 接受它的形状后就自动走单 kernel。

以下情况自动回退到多段 kernel，行为和改动前一样：
- AITER 没有单 kernel（< 0.1.29，例如本分支镜像自带的 0.1.28.post1）；
- AITER 的 `small_m_eligible` 不接受当前形状（例如没打下面的补丁）。




### 实测

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



### 启动命令

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


### 已知问题

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
