# DeepSeek-V4.1-Flash DP8+EP8：仓库外的 aiter / MoRI patch

这些 patch 改的是 aiter 和 MoRI，不在 vLLM 仓库里，放在 [`patches/`](./patches/) 下，方便构建镜像。正式版本会另外提到 ROCm/aiter 和 ROCm/mori。

## 单 kernel MegaMoE（#2）：AITER 0.1.29 加三个补丁

按顺序打 `patches/` 下的三个补丁，基线是 amd-aiter 0.1.29，补丁都只改 `aiter/ops/flydsl/kernels/mega_moe/` 下的文件：

| 补丁 | 内容 | 不打时 |
|---|---|---|
| `0001-aiter-0.1.29-mega_moe-small_m-dsv41-flash.patch` | `small_m_eligible` 允许 5120 / 2304；每卡 ≤ 8 token 的档位在 V4.1 上会触发 `assert N_TILES % 4 == 0`，改为退回 `tile_n=128` | V4.1 不走单 kernel |
| `0002-aiter-0.1.29-mega_moe-small_m-any-experts-shared-slot.patch` | 每卡专家数从只允许 48 个放宽到 1～64 个，topk 放宽到 1～8，所以草稿 MoE（128 个专家、top-3）和融合共享专家后的布局（392 个专家、top-7）都能走单 kernel；新增 `SmallMRuntime(shared_experts=N)`，共享专家槽不参与热点专家和 FP8 combine 行的判断 | 草稿 MoE 和融合配置仍走多段 kernel |
| `0003-aiter-0.1.29-mega_moe-small_m-gemm-pipeline.patch` | GEMM2：K 循环完全展开，下一 K 步的 W2 提前发出，基址改为 uniform 以去掉 waterfall 循环；GEMM1：每卡 ≥256 token 的档位用 `tile_n=256` | 单 kernel 慢约 6% |

- 补丁头里写了具体改动、实测和代价。
- 三个补丁都不改变主模型形状的计算结果：在均匀、zipf 和真实 decode 路由上，输出都与只打 0001 时逐位相同。
- V4.1 不需要单独的调优配置：单 kernel 的配置规则与形状无关，所有可调项和阈值扫描下来，默认值都在最优的 1% 以内。
- AITER < 0.1.29（没有单 kernel）或没打补丁（`small_m_eligible` 不接受 V4.1 形状）时，自动回退到多段 kernel。详见 [megamoe.md](./megamoe.md)。

## DP8+EP8 decode 优化系列：aiter / MoRI 补丁

来自 jiale（zovonoir）的 `dsv4.1f_1112_nightly` 系列。

| 补丁 | 改了什么 | 用到它的 vLLM commit | 没打这个补丁时 |
|---|---|---|---|
| `aiter-0.1.28.post1-sparse_mla-extra_scrub_nan.patch` | gfx950 gluon `sparse_mla_fwd` / `pa_decode_sparse` 增加 `extra_scrub_nan`：把从 extra（top-k 压缩）cache 读到的 NaN 置 0 | dsv4.1 rocm: keep AITER gluon sparse MLA with a KV connector … | 配了"可能载入外部 KV"的 KV connector 时，DeepSeek V4.1 照旧退回 Triton sparse attention；不配 connector 时没有区别 |
| `aiter-0.1.28.post1-mhc_fused_post_pre_delayed-mxfp8_out.patch` | 融合的 delayed mHC seam（`mhc_fused_post_pre_delayed_rmsnorm`）增加 `mxfp8_out`：写 bf16 layer input 的同时写一份 MXFP8 | dsv4.1 rocm: mHC seam emits its output MXFP8 too … | aiter seam 不出 MXFP8 副本，后面的消费者自己量化。decode overlap 路径用的是 vLLM 自己的 collapse kernel，不受影响 |
| `mori-1.2.3.post1-intranode-ep8-hidden5120-topk6.patch` | MoRI IntraNode EP8 dispatch / combine 的启动参数规则，新增 hidden 5120、topk 6 | 无（需要 `MORI_EP_LAUNCH_CONFIG_MODE=AUTO`） | 用 MoRI 默认的启动参数 |

vLLM 侧会检查已安装 aiter 的函数签名（`rocm_aiter_ops.sparse_mla_supports_extra_scrub_nan()`、`mhc_fused_post_pre_delayed_supports_mxfp8_out()`），只在 aiter 支持时才传新参数。所以不打这些补丁也能正常运行，只是拿不到对应的收益。

两个 aiter 补丁在 0.1.28.post1 和 0.1.29 上都能打；sparse MLA 那个在 0.1.28.post1 上有 2 行偏移，`patch` 会自动处理，内容不变。

## 打法

在 amd-aiter 0.1.29、amd_mori 1.2.3.post1 上，按下面的顺序打全部补丁验证过（`vllm/vllm-openai-rocm:nightly-9bc16ae6…` 自带这两个版本）：

```bash
SITE=/usr/local/lib/python3.12/dist-packages
P=/path/to/docs/patches
cd $SITE
# 单 kernel MegaMoE（#2）
patch -p1 --forward --fuzz=0 < $P/0001-aiter-0.1.29-mega_moe-small_m-dsv41-flash.patch
patch -p1 --forward --fuzz=0 < $P/0002-aiter-0.1.29-mega_moe-small_m-any-experts-shared-slot.patch
patch -p1 --forward --fuzz=0 < $P/0003-aiter-0.1.29-mega_moe-small_m-gemm-pipeline.patch
# DP8+EP8 decode 优化系列
patch -p1 --forward --fuzz=0 < $P/aiter-0.1.28.post1-sparse_mla-extra_scrub_nan.patch
patch -p1 --forward --fuzz=0 < $P/aiter-0.1.28.post1-mhc_fused_post_pre_delayed-mxfp8_out.patch
patch -p1 --forward --fuzz=0 < $P/mori-1.2.3.post1-intranode-ep8-hidden5120-topk6.patch
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO   # MoRI 规则只在 AUTO 模式下生效
```

## 实测

- NaN 清洗（B=64、SWA 128、extra 256/512）：
  - 不注入 NaN：AITER 和 vLLM Triton kernel 的最大差为 2.0e-3 / 9.8e-4，清洗开或关结果相同。
  - 往被引用的 extra 行写入 fp8 NaN（0x7F）和 bf16 NaN 后：不清洗时输出含 NaN，清洗后全部有限。
- mHC seam 的 MXFP8 epilogue：collapse kernel 多花 0.5–0.85 µs；每个 seam 省掉 1–2 个约 4.4 µs 的量化 kernel。
- MoRI 规则（8 卡 `bench_mori_graph.py --variants intra`，一对 dispatch+combine，单位 µs）：T=1/8/64/128 分别为 40.2/40.5/55.1/75.9 → 37.7/38.9/50.2/68.4。
- 单 kernel MegaMoE 的实测见 [megamoe.md](./megamoe.md)。
