# DeepSeek-V4.1-Flash：prefill 走 AITER large_m MegaMoE，DSpark 草稿阶段两项融合

2026-11-19，8× MI355X，DP8+EP8，`--moe-backend aiter_mega_moe`，DSpark nst5。基于 PR #1（EPLB）与 PR #2（decode 单 kernel，见 [megamoe.md](./megamoe.md)）。

## 改动

| commit | 内容 | 开关（默认） | 改输出 |
|---|---|---|---|
| prefill 走 large_m | DP 间每卡最大 token 数 ≥ 513 的 eager 调用改走 AITER `large_m`（一次量化 + 一个 persistent kernel） | `VLLM_ROCM_AITER_MEGA_MOE_LARGE_M`（1）、`VLLM_ROCM_AITER_MEGA_MOE_LARGE_M_MIN_TOKENS`（513） | 否（与多段 kernel 的 compact 路径逐位一致） |
| 草稿 greedy markov 融合 | DSpark greedy 草稿的 5 步 markov 链：每步的 bias GEMM、加法、argmax、token 拷贝合成 2 个 Triton launch | `VLLM_DSPARK_FUSED_GREEDY_MARKOV`（1） | 否（只影响草稿；接受长度一致） |
| 草稿 hc post 融合 | 草稿末尾的 hyper-connection post 由 fp32 torch einsum（约 7 个 kernel）改为 AITER 融合 `mhc_post`（与主模型层同一个 op） | `VLLM_DSPARK_FUSED_HC_POST`（1） | 否（只改草稿 hidden 的 bf16 舍入，目标模型逐个验证草稿 token） |

### prefill 走 large_m

| DP 间每卡最大 token 数 | 路径 |
|---|---|
| 1～512 | 单 kernel `SmallMRuntime`（PR #2） |
| 513～16384，eager | **large_m** |
| 513～`max-cudagraph-capture-size`，CUDA graph 捕获 | 多段 kernel（不变） |

- `vllm/models/deepseek_v4/amd/mega_moe.py`
  - 容量为 `max_num_batched_tokens`（向上取 2 的幂，16384）的 `MegaMoEV2` 构造时打开 large_m，按层数分配 protocol slot（主干层 + MTP 层 + 8 个余量）。`finalize_weights` 时每层领一个 slot。所有层仍共享这一个实例。
  - `forward`：单 kernel 不接的调用，若走到这个实例且 `uses_large_m()` 为真，`set_large_m_enabled(True)`，用 `bind_layer` 绑定本层权重和 slot 后调用；否则 `set_large_m_enabled(False)`，走多段 kernel。
  - `uses_large_m(num_tokens)`：DP 间最大 token 数 ≥ 阈值且不在 CUDA graph 捕获中。所有 EP rank 结果一致。
- `vllm/models/deepseek_v4/amd/model.py`：`uses_large_m` 为真时共享专家串行（同单 kernel，见"已知问题"）。
- 与 EPLB（PR #1）：`forward` 先把逻辑 id 映射成物理 id（router 已融合映射时跳过），再选路径，所以单 kernel、large_m、多段 kernel 都按物理 id dispatch。EPLB 在权重张量内原地搬移专家，各路径在 launch 时按指针读权重，重排后不需要重新绑定；fanout pair 清零只用于多段 kernel（单 kernel 与 large_m 没有 fanout mask）。8 卡迁移测试（rank 内置换 + 跨 rank 轮换）：迁移前后单 kernel（eager / graph）、large_m、多段 kernel 输出逐位一致。
- 自动回退到多段 kernel（行为同 base）：AITER 没有 large_m 参数，或 AITER 不支持当前形状与容量（large_m 只开放 V4.1-Flash 384 专家 / top-6 / 每卡 16384；融合共享专家的 392 / top-7 与草稿 128 / top-3 都不走 large_m）。

### 草稿阶段两项融合

`vllm/models/deepseek_v41/amd/ops/fused_markov_greedy.py`、`vllm/models/deepseek_v41/amd/dspark.py`。

- greedy markov：step kernel 对每个词表块用一次 dot 算出所有请求的 markov bias，加上 base logits（按 head dtype 复现原循环的 bf16 舍入），取块内 argmax；pick kernel 归约各块、写 token 并 gather 下一步的 markov embedding。只在 greedy 草稿（`draft_logits` 为空）、无 acceptance estimator / confidence head / 词表裁剪 / watermark 时启用，其余情况走原循环。
- hc post：`_aiter_mhc_supported`（hc 4，hidden % 256 == 0）时调 `torch.ops.vllm.mhc_post_aiter`，否则回到 einsum。
- 默认打开的理由：两项都不改目标模型验证后的输出，只影响草稿。markov 融合中 bias 的 fp32 求和顺序与原 GEMM 不同，bias 的最后一位可能不同，在 argmax 接近并列时草稿 token 可能与原循环不同；实测（真实 markov 权重，2 种负载 × 3 种子）草稿 token 全部一致，整网验收中接受长度一致。hc post 用的是主模型每层都在用的 op，与 einsum 相对误差 1.3e-5。草稿 token 的差异只影响接受率。

## 依赖：AITER 补丁

`docs/patches/` 下只需打一个补丁：[`aiter-0.1.29-mega_moe-dsv41-flash.patch`](./patches/aiter-0.1.29-mega_moe-dsv41-flash.patch)，相对 amd-aiter 0.1.29，内容见补丁头。它包含 PR #2 原有 `0001`～`0003` 的内容（V4.1 形状、每卡 1～64 专家 / top-1～8 与共享专家槽、GEMM 流水），另加 large_m 所需的 [ROCm/aiter #7699](https://github.com/ROCm/aiter/pull/7699)、`9a5c7eb0e`（V4.1-Flash 形状）、`79bb03386`（large_m 每卡 16384）、`0aded6237`（AOT）与 `168f0a237`（草稿 MoE 形状 E128 / top-3）。

```bash
cd "$(python -c 'import aiter, os; print(os.path.dirname(os.path.dirname(aiter.__file__)))')"
git apply /path/to/aiter-0.1.29-mega_moe-dsv41-flash.patch   # 目录在某个 git 仓库里时加 GIT_CEILING_DIRECTORIES=$PWD，或用 patch -p1
```

EPLB（PR #1）改了 `csrc`（`topk_softplus_sqrt` 融合映射），预编译镜像上需重编 `_moe_C`；从源码构建本分支时随构建完成。

## 推荐启动参数

在 PR #1 recipe（`recipes/DeepSeek-V4.1-Flash/MI355X_DEP8_AITER_MegaMoE_EPLB.md`）的命令上加 `--prefill-schedule-interval 16`。三个新开关默认即开，不需要设置。

`--prefill-schedule-interval K`（vLLM 已有参数）：DP 各 rank 只在 `step_counter % K == 0` 的 step 接纳 prefill，把各 rank 的 prefill 攒到同一批 step。MegaMoE 按 DP 间最大 token 数选路径、dispatch/combine 是 EP 集合通信，只要一个 rank 在 prefill，其余 rank 也要进 large_m 等它；对齐后"1 个 rank prefill、7 个 rank 等"的 step 大幅减少。实测（PR 状态，random，SLA 口径：mean TTFT ≤ 10 s、mean TPOT ≤ 50 ms）：

| case | 输出 tok/s off → K=16 | TTFT mean (s) | TPOT mean (ms) |
|---|---|---|---|
| 8k/0% c256 | 6862 → 8093（+17.9%） | 2.13 → 2.73 | 30.9 → 24.5 |
| 8k/30% c512 | 10764 → 13709（+27.4%） | 3.0 → 3.34 | 39.7 → 29.0 |
| 64k/0% c64 | 1455 → 1864（+28.1%） | 4.9 → 6.17 | 37.1 → 25.9 |
| 64k/30% c128 | 2498 → 2903（+16.2%） | 6.2 → 8.05 | 43.5 → 34.5 |
| 8k/90% c512 | 17039 → 18334（+7.6%） | 0.99 → 1.24 | 24.2 → 21.6 |
| 64k/90% c512 | 9083 → 9858（+8.5%） | 3.96 → 5.76 | 48.8 → 42.4 |
| 8k/0% c512 | 8666 → 11546（+33.2%），SLA 由不达标变为达标 | 4.00 → 4.31 | 50.1 → 34.5 |

K 扫描 2/4/8/16/32 中 16 在各 case 最好或接近最好；TTFT 随 K 上升（prefill 最多推迟 K−1 步），64k 高并发以 TTFT 为瓶颈时用 8（64k/0% c128 只有 K=8 达标）。接受长度各组一致。DP=1 时无此收益。

EPLB（recipe 中的 `--enable-eplb --eplb-config '{"window_size":1000,"step_interval":2000}'`）在本 PR 之上同样有效：c512 交错 A/B（off/on/on/off，均加 K=16）输出 tok/s 8k/90% +6.7%、8k/0% +11.9%、8k/30% +11.8%、64k/90% +9.7%，接受长度一致，gsm8k 96.44%。

## 实测

**large_m 算子**（8 卡，随机 FP4 权重，均匀路由，CUDA graph，8 rank 取 max）：

| 每卡 token | 多段 kernel (µs) | large_m (µs) | 加速 |
|---|---|---|---|
| 1024 | 635.8 | 529.5 | 1.20x |
| 2048 | 1020.5 | 802.6 | 1.27x |
| 4096 | 1481.9 | 1377.9 | 1.08x |
| 8192 | 2681.3 | 2518.0 | 1.06x |
| 16384 | 5042.2 | 4986.8 | 1.01x |

与 FP32 参考误差 0.058（阈值 0.10），与多段 kernel 逐位一致。每卡 ≤ 96 token 时 large_m 比多段 kernel 慢，192～512 时快但不如单 kernel，所以阈值取 513。

**整网**（单 kernel + large_m 相对镜像多段 kernel，random，输出 1K）：8k/0% 单卡输出 tok/s +20.5%，8k/30% +14.7%，8k/90% +16.9%，64k/0% +15.9%；SLA 最大并发 64k/30% 64 → 128、64k/90% 256 → ≥512。gsm8k 0.964 → 0.964。

**草稿两项**（单卡，冷 cache，CUDA graph，真实 markov / lm_head 权重；c512 稳态形状：每卡 62～63 个请求、草稿 M 320～325）：

| 项 | 8k/90% (µs) | 8k/0% (µs) | 占草稿阶段（约 3.0 ms/步） |
|---|---|---|---|
| greedy markov 5 步 | 224 → 130 | 233 → 130 | −3.1% / −3.4% |
| hc post | 87 → 10.5 | 80 → 8 | −2.5% / −2.4% |

合计每步约 170 µs（约占纯 decode step 34.7 ms 的 0.5%）。

## 已知问题

- **全驻留 kernel 不能和辅助流上的共享专家同时运行。** 单 kernel 与 large_m 都要求整个 grid 驻留在 GPU 上，与另一个 stream 上的 kernel 共驻会挂死（GPU 100%，无报错）；所以二者运行时共享专家串行。DP 间最大 token 数 ≥ 513 而本卡 ≤ 256（负载不均的 prefill）时同样适用。
- **CUDA graph 捕获不走 large_m**（同 sglang #47065），每卡 513～`max-cudagraph-capture-size` 的捕获仍是多段 kernel。
- **KV cache 少约 1.8%**（单 kernel 与 large_m 的 buffer）。
- **大 rank 的 dispatch**：DP 间负载不均时（1～2 个 rank 满 16384），大 rank 的 dispatch 约占 large_m 的 60～76%，每发送 rank 约 145～215 GB/s；已验证增加发送 CTA 与每 wave 在途行数无收益（传输路径饱和），下一步是按目标 rank 去重。
