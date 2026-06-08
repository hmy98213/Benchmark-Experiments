#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

config="${A2_RANDOM3_SMOKE_CONFIG:-configs/a2_random3_smoke.yaml}"

usage() {
  cat <<'EOF'
Usage:
  bash run_a2_random3_smoke.sh [runner args]

Examples:
  bash run_a2_random3_smoke.sh --dry-run
  bash run_a2_random3_smoke.sh
  bash run_a2_random3_smoke.sh --tp 6 --npu-devices 2,3,4,5,6,7
  bash run_a2_random3_smoke.sh --resume
  bash run_a2_random3_smoke.sh --skip-preflight --dry-run

The default config is configs/a2_random3_smoke.yaml. It only runs random3
smoke cases, not datastores/specbench/full matrices.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

echo "[a2-smoke] config=${config}"

if [[ "${1:-}" == "--skip-preflight" ]]; then
  shift
else
  echo "[a2-smoke] preflight"
  python3 scripts/run_a2_container_experiments.py \
    --config "${config}" \
    --preflight \
    "$@"
fi

echo "[a2-smoke] run"
python3 scripts/run_a2_container_experiments.py \
  --config "${config}" \
  "$@"
