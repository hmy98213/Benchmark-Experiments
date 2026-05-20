#!/usr/bin/env bash
set -euo pipefail

image="${1:?usage: $0 IMAGE}"
shift || true

docker run --rm -i --entrypoint python3 "$@" "${image}" - <<'PY'
import importlib.metadata as md
for pkg in ["vllm", "vllm-ascend", "torch", "torch-npu", "arctic-inference", "transformers"]:
    try:
        print(f"{pkg}={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg}=<not installed>")
PY
