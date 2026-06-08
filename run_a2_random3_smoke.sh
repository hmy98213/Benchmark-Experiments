#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

config="${A2_RANDOM3_SMOKE_CONFIG:-configs/a2_container_experiments.yaml}"

if [[ "${1:-}" == "--skip-preflight" ]]; then
  shift
else
  python3 scripts/run_a2_container_experiments.py \
    --config "${config}" \
    --preflight
fi

python3 scripts/run_a2_container_experiments.py \
  --config "${config}" \
  "$@"
