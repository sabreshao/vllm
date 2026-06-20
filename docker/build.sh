# ============================================================================
# DeepSeek-V4-Pro accuracy notes (gfx950 / MI355X) — read before building
# ============================================================================
# 1. aiter MUST be commit 971d98b8e (0.1.14rc1.dev318, CK submodule af7118e34).
#    aiter v0.1.15 regressed the gfx950 MXFP4 fused-MoE kernels (DeepSeek-V4
#    served garbage / GSM8K ~0.16). Pinned via --build-arg AITER_BRANCH below
#    (dockerfile.rocm_new clones then `git checkout`, so a commit SHA works).
# 2. aiter MUST be built AOT (PREBUILD_KERNELS=1 GPU_ARCHS=gfx950) — already the
#    case in dockerfile.rocm_new's build_aiter stage. A JIT/`setup.py develop`
#    aiter produces slightly different gfx950 kernels that amplify over the
#    ~60-layer model into degraded accuracy (GSM8K ~0.2). See aiter_build_jit_aot.md.
# 3. Attention/MoE path depends on which vLLM you build (two targets below):
#      - vllm-project (generic .amd.model native sparse-MLA): accurate with
#        `--moe-backend triton_unfused` (GSM8K ~0.96). The AITER fused MoE on
#        this path is NOT correct (gpt-oss-oriented mxfp4 oracle weight prep).
#      - Fangzhou-Ai dsv4-rocm-atom-full-attention (ATOM CSA full-attention +
#        ATOM MXFP4 MoE): accurate with the AITER fused MoE (GSM8K ~0.92-0.94).
# 4. Runtime (launch) flags required for DeepSeek-V4 MXFP4 MoE accuracy — these
#    are NOT baked into the image (model-specific):
#      export AITER_BF16_FP8_MOE_BOUND=0   # force fp8 a8w4; avoids SiLU-vs-Swiglu
#                                          # gate crash + the bf16-a16w4 path
#      export ATOM_MOE_GU_ITLV=1
#      --kv-cache-dtype fp8 --tokenizer-mode deepseek_v4
# ============================================================================

# --- Generic .amd.model image (native sparse-MLA; use Triton MoE for accuracy) ---
#DOCKER_BUILDKIT=1 docker build . \
#  --build-arg PYTORCH_ROCM_ARCH=gfx950 \
#  --build-arg AITER_ROCM_ARCH=gfx950 \
#  --build-arg AITER_BRANCH=971d98b8e \
#  --build-arg REMOTE_VLLM=1 \
#  --build-arg VLLM_REPO=https://github.com/vllm-project/vllm.git \
#  --build-arg VLLM_BRANCH=b4092176b9bc76839f76b0e91972a52e8b14ea2b \
#  -f docker/dockerfile.rocm_new \
#  -t sabreshao/vllm:aiter_0620

# --- Fangzhou-Ai ATOM full-attention image (accurate AITER fused MoE) ---
# Same aiter pin + AOT; only VLLM_REPO/BRANCH differ. Uncomment to build.
DOCKER_BUILDKIT=1 docker build . \
   --build-arg PYTORCH_ROCM_ARCH=gfx950 \
   --build-arg AITER_ROCM_ARCH=gfx950 \
   --build-arg AITER_BRANCH=971d98b8e \
   --build-arg REMOTE_VLLM=1 \
   --build-arg VLLM_REPO=https://github.com/Fangzhou-Ai/vllm.git \
   --build-arg VLLM_BRANCH=dsv4-rocm-atom-full-attention \
   -f docker/dockerfile.rocm_new \
   -t sabreshao/vllm:aiter_0620_full
