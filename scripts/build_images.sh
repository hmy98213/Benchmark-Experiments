#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cuda_base="${CUDA_BASE_IMAGE:-vllm/vllm-openai:latest}"
ascend_base="${ASCEND_BASE_IMAGE:-registry.dev.huawei.com/flash_stor/vllm-ascend:v0.18.0rc1-openeuler}"

case "${1:-all}" in
  cuda)
    docker build \
      --network host \
      --build-arg "BASE_IMAGE=${cuda_base}" \
      -t suffix-vllm-cuda:latest \
      -f "${repo_root}/docker/Dockerfile.cuda" \
      "${repo_root}"
    ;;
  ascend)
    docker build \
      --network host \
      --build-arg "BASE_IMAGE=${ascend_base}" \
      -t suffix-vllm-ascend:latest \
      -f "${repo_root}/docker/Dockerfile.ascend" \
      "${repo_root}"
    ;;
  all)
    "$0" cuda
    "$0" ascend
    ;;
  *)
    echo "Usage: $0 [cuda|ascend|all]" >&2
    exit 2
    ;;
esac
