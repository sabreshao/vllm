DOCKER_BUILDKIT=1 docker build . \
  --build-arg PYTORCH_ROCM_ARCH=gfx950 \
  --build-arg AITER_ROCM_ARCH=gfx950 \
  --build-arg REMOTE_VLLM=1 \
  --build-arg VLLM_REPO=https://github.com/Fangzhou-Ai/vllm.git \
  --build-arg VLLM_BRANCH=dsv4-rocm-atom-full-attention \
  -f docker/dockerfile.rocm_new \
  -t sabreshao/vllm:aiter_0618_a
