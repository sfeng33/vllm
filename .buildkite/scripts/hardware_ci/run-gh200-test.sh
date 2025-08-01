#!/bin/bash

# This script build the GH200 docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Skip the new torch installation during build since we are using the specified version for arm64 in the Dockerfile
python3 use_existing_torch.py

# Try building the docker image
DOCKER_BUILDKIT=1 docker build . \
  --file docker/Dockerfile \
  --target vllm-openai \
  --platform "linux/arm64" \
  -t gh200-test \
  --build-arg max_jobs=66 \
  --build-arg nvcc_threads=2 \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg torch_cuda_arch_list="9.0+PTX"

# Setup cleanup
remove_docker_container() { docker rm -f gh200-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image and test offline inference
docker run -e HF_TOKEN -e VLLM_WORKER_MULTIPROC_METHOD=spawn -v /root/.cache/huggingface:/root/.cache/huggingface --name gh200-test --gpus=all --entrypoint="" gh200-test bash -c '
    python3 examples/offline_inference/basic/generate.py --model meta-llama/Llama-3.2-1B
'
