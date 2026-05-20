#!/usr/bin/env python3
"""Run one vLLM serving benchmark case in Docker.

The script starts a vLLM OpenAI-compatible server container, waits for it to be
ready, runs `vllm bench serve` inside the same container, and stores artifacts
under the host result directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def docker_rm(container: str) -> None:
    run(["docker", "rm", "-f", container], check=False)


def wait_ready(port: int, timeout_s: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status < 500:
                    print(f"Server ready: {url}", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"Server did not become ready within {timeout_s}s: {last_error}")


def as_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def add_key_values(cmd: list[str], options: dict[str, Any]) -> None:
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            cmd.append(flag)
        elif isinstance(value, list):
            for item in value:
                cmd.extend([flag, as_str(item)])
        else:
            cmd.extend([flag, as_str(value)])


def npu_devices(devices: list[int]) -> list[str]:
    args: list[str] = []
    for idx in devices:
        args.extend(["--device", f"/dev/davinci{idx}"])
    for dev in ["/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc"]:
        if Path(dev).exists():
            args.extend(["--device", dev])
    return args


def ascend_mounts() -> list[str]:
    candidates = [
        ("/usr/local/dcmi", "/usr/local/dcmi"),
        ("/usr/local/Ascend/driver/tools/hccn_tool", "/usr/local/Ascend/driver/tools/hccn_tool"),
        ("/usr/local/bin/npu-smi", "/usr/local/bin/npu-smi"),
        ("/usr/local/Ascend/driver/lib64", "/usr/local/Ascend/driver/lib64"),
        ("/usr/local/Ascend/driver/version.info", "/usr/local/Ascend/driver/version.info"),
        ("/etc/ascend_install.info", "/etc/ascend_install.info"),
        ("/etc/hccn.conf", "/etc/hccn.conf"),
    ]
    mounts: list[str] = []
    for host, container in candidates:
        if Path(host).exists():
            mounts.extend(["-v", f"{host}:{container}"])
    return mounts


def build_docker_run(case: dict[str, Any], result_dir: Path) -> list[str]:
    platform = case["platform"]
    container = case["container_name"]
    image = case["image"]
    port = int(case.get("port", 8000))
    model = case["model"]
    served_model_name = case.get("served_model_name", Path(model).name or model)

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "--network",
        "host",
        "--ipc",
        "host",
        "-v",
        f"{result_dir.resolve()}:/workspace/results",
        "-v",
        f"{REPO_ROOT.resolve()}:/workspace/suffix-bench:ro",
        "--entrypoint",
        case.get("entrypoint", "vllm"),
    ]

    for mount in case.get("mounts", []):
        cmd.extend(["-v", mount])

    if platform == "cuda":
        gpus = case.get("gpus", "all")
        cmd.extend(["--gpus", str(gpus)])
        hf_cache = case.get("hf_cache", str(Path.home() / ".cache" / "huggingface"))
        Path(hf_cache).mkdir(parents=True, exist_ok=True)
        cmd.extend(["-v", f"{hf_cache}:/root/.cache/huggingface"])
    elif platform == "ascend":
        devices = [int(x) for x in case.get("npu_devices", [0, 1, 2, 3])]
        cmd.extend(npu_devices(devices))
        cmd.extend(ascend_mounts())
        cmd.extend(["-e", f"ASCEND_RT_VISIBLE_DEVICES={','.join(map(str, devices))}"])
        for key, value in {
            "TASK_QUEUE_ENABLE": "1",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
        }.items():
            cmd.extend(["-e", f"{key}={value}"])
    else:
        raise ValueError(f"Unsupported platform: {platform}")

    for key, value in case.get("env", {}).items():
        cmd.extend(["-e", f"{key}={value}"])

    serve_cmd = [
        "serve",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
    ]
    add_key_values(serve_cmd, case.get("server_args", {}))
    speculative_config = case.get("speculative_config")
    if speculative_config:
        serve_cmd.extend(["--speculative-config", json.dumps(speculative_config, separators=(",", ":"))])

    cmd.append(image)
    cmd.extend(serve_cmd)
    return cmd


def build_bench_cmd(case: dict[str, Any], result_filename: str) -> list[str]:
    port = int(case.get("port", 8000))
    bench = dict(case.get("bench", {}))
    served_model_name = case.get("served_model_name", Path(case["model"]).name or case["model"])

    cmd = [
        "docker",
        "exec",
        case["container_name"],
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        served_model_name,
        "--save-result",
        "--save-detailed",
        "--result-dir",
        "/workspace/results",
        "--result-filename",
        result_filename,
    ]
    add_key_values(cmd, bench)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path, help="Path to one case JSON file.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Render commands and metadata without starting Docker.")
    parser.add_argument("--ready-timeout-s", type=int, default=1800)
    args = parser.parse_args()

    case = load_json(args.case)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.output_dir / case["name"] / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)
    result_filename = f"{case['name']}.json"

    server_cmd = build_docker_run(case, result_dir)
    bench_cmd = build_bench_cmd(case, result_filename)
    metadata = {
        "case": case,
        "started_at_utc": timestamp,
        "server_cmd": server_cmd,
        "bench_cmd": bench_cmd,
        "result_dir": str(result_dir),
    }
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
        return 0

    docker_rm(case["container_name"])
    try:
        run(server_cmd)
        wait_ready(int(case.get("port", 8000)), args.ready_timeout_s)
        result = run(bench_cmd, capture=True)
        (result_dir / "bench_stdout.log").write_text(result.stdout or "", encoding="utf-8")
    finally:
        if not args.keep_server:
            docker_rm(case["container_name"])

    print(f"Artifacts: {result_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
