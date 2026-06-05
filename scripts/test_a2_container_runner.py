#!/usr/bin/env python3
"""Self-test the A2 in-container runner without a real vLLM install."""

from __future__ import annotations

import csv
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_a2_container_experiments.py"


def run_cmd(args: list[str]) -> str:
    proc = subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(args)}")
    return proc.stdout


def write_fake_vllm(tmp: Path) -> Path:
    fake_py = tmp / "fake_vllm.py"
    fake_py.write_text(
        """\
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


args = sys.argv[1:]
if args and args[0] == "serve":
    port = int(args[args.index("--port") + 1])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/v1/models":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data":[]}')
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 30
    server.handle_request()
elif len(args) >= 2 and args[0] == "bench" and args[1] == "serve":
    result_dir = args[args.index("--result-dir") + 1]
    result_filename = args[args.index("--result-filename") + 1]
    os.makedirs(result_dir, exist_ok=True)
    result = {
        "completed": 2,
        "failed": 0,
        "request_throughput": 1.0,
        "output_throughput": 2.0,
        "total_token_throughput": 3.0,
        "mean_ttft_ms": 4.0,
        "mean_tpot_ms": 5.0,
        "mean_itl_ms": 6.0,
        "spec_decode_acceptance_rate": 0.5,
        "spec_decode_acceptance_length": 1.5,
    }
    with open(os.path.join(result_dir, result_filename), "w", encoding="utf-8") as f:
        json.dump(result, f)
    print("fake bench ok")
else:
    print("unexpected fake vllm args", args, file=sys.stderr)
    sys.exit(2)
""",
        encoding="utf-8",
    )

    if os.name == "nt":
        wrapper = tmp / "vllm.cmd"
        wrapper.write_text(
            f'@echo off\r\n"{sys.executable}" "{fake_py}" %*\r\n',
            encoding="ascii",
        )
    else:
        wrapper = tmp / "vllm"
        wrapper.write_text(
            f'#!/usr/bin/env sh\nexec "{sys.executable}" "{fake_py}" "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


def write_config(tmp: Path, vllm_bin: Path, output_dir: Path) -> Path:
    config = tmp / "config.yaml"
    config.write_text(
        f"""\
VLLM_BIN: "{vllm_bin.as_posix()}"
models: [fake_model_a, fake_model_b]
datasets: [random_tiny]
methods: [baseline, suffix]
TP: 1
DP: 1
NPU_DEVICES: "0"
OUTPUT_DIR: "{output_dir.as_posix()}"
PORT_BASE: 19400
READY_TIMEOUT_S: 30
STOP_ON_FAILURE: true
ASCEND_DEVICE_CHECK: off
ENV: {{}}
COMMON_SERVER_ARGS: {{}}
COMMON_BENCH_ARGS: {{}}
model_catalog:
  fake_model_a:
    path: fake-model-a
    served_model_name: fake-model-a
  fake_model_b:
    path: fake-model-b
    served_model_name: fake-model-b
dataset_catalog:
  random_tiny:
    dataset_name: random
    num_prompts: 2
method_catalog:
  baseline: null
  suffix:
    method: suffix
    num_speculative_tokens: 4
""",
        encoding="utf-8",
    )
    return config


def assert_summary(output_dir: Path) -> None:
    summary = output_dir / "summary.csv"
    if not summary.exists():
        raise AssertionError(f"summary.csv was not written: {summary}")
    with summary.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = {(row["model"], row["dataset"], row["method"]) for row in rows}
    expected = {
        ("fake_model_a", "random_tiny", "baseline"),
        ("fake_model_a", "random_tiny", "suffix"),
        ("fake_model_b", "random_tiny", "baseline"),
        ("fake_model_b", "random_tiny", "suffix"),
    }
    if keys != expected:
        raise AssertionError(f"Unexpected summary rows: {keys}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a2-runner-self-test-") as raw_tmp:
        tmp = Path(raw_tmp)
        output_dir = tmp / "out"
        vllm_bin = write_fake_vllm(tmp)
        config = write_config(tmp, vllm_bin, output_dir)

        run_cmd([sys.executable, str(RUNNER), "--config", str(config), "--preflight"])
        run_cmd([sys.executable, str(RUNNER), "--config", str(config)])
        assert_summary(output_dir)
        resume_output = run_cmd([sys.executable, str(RUNNER), "--config", str(config), "--resume"])
        for model in ("fake_model_a", "fake_model_b"):
            for method in ("baseline", "suffix"):
                case = f"a2_{model}_random_tiny_{method}"
                if f"Skipping completed case: {case}" not in resume_output:
                    raise AssertionError(f"{case} was not skipped during resume")

    print("A2 runner self-test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
