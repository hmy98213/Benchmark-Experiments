#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${repo_root}/data/spec_bench"
mkdir -p "${data_dir}"

spec_bench_url="https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl"
spec_bench_file="${data_dir}/question.jsonl"

if [[ ! -s "${spec_bench_file}" ]]; then
  curl -L --retry 5 --retry-delay 2 \
    -o "${spec_bench_file}" \
    "${spec_bench_url}"
fi

python3 - <<PY
from pathlib import Path
path = Path("${spec_bench_file}")
lines = sum(1 for _ in path.open("r", encoding="utf-8"))
print(f"Spec-Bench data ready: {path} ({lines} jsonl rows)")
PY
