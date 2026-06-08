#!/usr/bin/env bash
set -euo pipefail

config="${A2OPEN_CONFIG:-configs/a2_datastores384_open_models.yaml}"

python3 scripts/run_a2_container_experiments.py --config "${config}" --preflight "$@"
python3 scripts/run_a2_container_experiments.py --config "${config}" --dry-run "$@"
python3 scripts/run_a2_container_experiments.py --config "${config}" "$@"
