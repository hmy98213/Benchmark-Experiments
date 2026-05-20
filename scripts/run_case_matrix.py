#!/usr/bin/env python3
"""Run a list of benchmark cases sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path, help="Matrix JSON containing a `cases` list.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--ready-timeout-s", type=int, default=1800)
    args = parser.parse_args()

    matrix = load_json(args.matrix)
    cases = matrix.get("cases", [])
    if not cases:
        raise ValueError("Matrix file must contain a non-empty `cases` list.")

    single_runner = REPO_ROOT / "scripts" / "run_single_case.py"
    failures: list[tuple[str, int]] = []
    for item in cases:
        case_path = (args.matrix.parent / item).resolve() if not Path(item).is_absolute() else Path(item)
        cmd = [
            sys.executable,
            str(single_runner),
            "--case",
            str(case_path),
            "--output-dir",
            str(args.output_dir),
            "--ready-timeout-s",
            str(args.ready_timeout_s),
        ]
        if args.keep_server:
            cmd.append("--keep-server")
        print("\n==> Running", case_path, flush=True)
        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            failures.append((str(case_path), completed.returncode))
            if matrix.get("stop_on_failure", True):
                break

    if failures:
        print("\nFailures:", flush=True)
        for path, code in failures:
            print(f"  {path}: exit {code}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
