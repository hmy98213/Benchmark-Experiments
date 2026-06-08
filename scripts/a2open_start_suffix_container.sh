#!/usr/bin/env bash
set -euo pipefail

name="${A2OPEN_CONTAINER_NAME:-a2open_suffix_runner}"
image="${A2OPEN_IMAGE:-suffix-vllm-ascend:latest}"
repo_dir="${A2OPEN_REPO_DIR:-/data/a2open/Benchmark-Experiments}"
results_dir="${A2OPEN_RESULTS_DIR:-/data/a2open/results}"
model_dir="${A2OPEN_MODEL_DIR:-/data/model}"

mkdir -p "${results_dir}"

docker rm -f "${name}" >/dev/null 2>&1 || true

docker run -dit \
  --name "${name}" \
  --network host \
  --ipc host \
  --privileged \
  --shm-size 900g \
  --ulimit memlock=-1 \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /tmp:/tmp \
  -v "${model_dir}:/models" \
  -v "${repo_dir}:/workspace/suffix-bench" \
  -v "${results_dir}:/workspace/results" \
  "${image}" \
  bash

echo "[container] ${name} started from ${image}"
echo "[container] run:"
echo "  docker exec -it ${name} bash -lc 'cd /workspace/suffix-bench && python3 scripts/run_a2_container_experiments.py --config configs/a2_datastores384_open_models.yaml --preflight'"
