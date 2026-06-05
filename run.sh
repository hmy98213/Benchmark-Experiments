#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

config="${A2_EXPERIMENT_CONFIG:-configs/a2_container_experiments.yaml}"

if [[ $# -gt 0 && "$1" != -* ]]; then
  config="$1"
  shift
fi

python3 scripts/run_a2_container_experiments.py \
  --config "${config}" \
  "$@"
