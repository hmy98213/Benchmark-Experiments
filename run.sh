#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

config="${A2_EXPERIMENT_CONFIG:-configs/a2_container_experiments.yaml}"

python3 scripts/run_a2_container_experiments.py \
  --config "${config}" \
  "$@"
