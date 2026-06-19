# AITER_BRANCH pinned to the ATOM-validated aiter for DeepSeek-V4 MXFP4 MoE.
# v0.1.15 regressed the gfx950 MXFP4 fused-MoE kernels; 971d98b8e (0.1.14rc1.dev318,
# CK submodule af7118e34) is verified working (GSM8K 0.94). Accepts branch/tag/commit.
DOCKER_BUILDKIT=1 docker build . \
  --build-arg PYTORCH_ROCM_ARCH=gfx950 \
  --build-arg AITER_ROCM_ARCH=gfx950 \
  --build-arg AITER_BRANCH=971d98b8e \
  --build-arg REMOTE_VLLM=1 \
  --build-arg VLLM_REPO=https://github.com/vllm-project/vllm.git \
  --build-arg VLLM_BRANCH=b4092176b9bc76839f76b0e91972a52e8b14ea2b \
  -f docker/dockerfile.rocm_new \
  -t sabreshao/vllm:aiter_0618
