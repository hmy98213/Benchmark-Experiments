#!/usr/bin/env bash
set -euo pipefail

model_root="${MODEL_ROOT:-/data/model}"
modelscope_bin="${MODELSCOPE_BIN:-/data/model/.venv-modelscope/bin/modelscope}"
max_workers="${MAX_WORKERS:-8}"

download_model() {
  local repo_id="$1"
  local target="$2"

  mkdir -p "${target}"
  echo "[download] ${repo_id} -> ${target}"
  "${modelscope_bin}" download \
    --model "${repo_id}" \
    --local_dir "${target}" \
    --max-workers "${max_workers}"
}

download_model \
  "Eco-Tech/Qwen3.6-35B-A3B-w8a8" \
  "${model_root}/Qwen3.6-35B-A3B-w8a8"

download_model \
  "zai-org/GLM-4.5-Air-FP8" \
  "${model_root}/GLM-4.5-Air-FP8"

download_model \
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
  "${model_root}/DeepSeek-R1-Distill-Llama-70B"

echo "[download] done"
