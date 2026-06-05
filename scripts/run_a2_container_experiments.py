#!/usr/bin/env python3
"""Run A2 vLLM benchmarks directly inside a vLLM Ascend container."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FIELDS = [
    "model",
    "dataset",
    "method",
    "completed",
    "failed",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
    "result_file",
]


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value == "{}":
        return {}
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by the default config.

    PyYAML is used when available.  This fallback supports nested mappings,
    block lists of scalars, comments, and scalar values.
    """

    root = {}
    stack = [(-1, root)]
    lines = [line for line in text.splitlines()]
    for i, raw in enumerate(lines):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unexpected list item: {raw}")
            parent.append(parse_scalar(stripped[2:]))
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = parse_scalar(value)
            continue
        next_is_list = False
        for later in lines[i + 1 :]:
            later_line = later.split("#", 1)[0].rstrip()
            if not later_line.strip():
                continue
            later_indent = len(later_line) - len(later_line.lstrip(" "))
            if later_indent <= indent:
                break
            next_is_list = later_line.strip().startswith("- ")
            break
        container = [] if next_is_list else {}
        parent[key] = container
        stack.append((indent, container))
    return root


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = simple_yaml_load(text)
    if not isinstance(data, dict):
        raise RuntimeError(f"Config must be a mapping: {path}")
    return data


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    stdout: Any = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=stdout if stdout is not None else subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )


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


def merge_dicts(*items: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if item:
            merged.update(item)
    return merged


def cfg(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config.get(key, default)


def normalize_check_mode(value: Any, default: str = "warn") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "error" if value else "off"
    if isinstance(value, int):
        return "error" if value else "off"

    mode = str(value).strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "none": "off",
        "1": "error",
        "true": "error",
        "yes": "error",
        "on": "error",
    }
    return aliases.get(mode, mode)


def vllm_bin(config: dict[str, Any]) -> str:
    candidate = str(cfg(config, "VLLM_BIN", "vllm"))
    return shutil.which(candidate) or candidate


def vllm_bin_exists(config: dict[str, Any]) -> bool:
    candidate = str(cfg(config, "VLLM_BIN", "vllm"))
    if Path(candidate).is_absolute() or os.sep in candidate:
        return Path(candidate).exists()
    return shutil.which(candidate) is not None


def model_entry(config: dict[str, Any], key: str) -> dict[str, Any]:
    catalog = config["model_catalog"]
    if key not in catalog:
        raise KeyError(f"Unknown model {key!r}. Add it to model_catalog.")
    entry = dict(catalog[key])
    entry.setdefault("name", key)
    entry.setdefault("served_model_name", key)
    entry.setdefault("tp", cfg(config, "TP", 1))
    entry.setdefault("dp", cfg(config, "DP", 1))
    entry.setdefault("npu_devices", cfg(config, "NPU_DEVICES", "0"))
    return entry


def dataset_entry(config: dict[str, Any], key: str) -> dict[str, Any]:
    catalog = config["dataset_catalog"]
    if key not in catalog:
        raise KeyError(f"Unknown dataset {key!r}. Add it to dataset_catalog.")
    entry = dict(catalog[key])
    entry.setdefault("name", key)
    return entry


def method_entry(config: dict[str, Any], key: str) -> dict[str, Any] | None:
    catalog = config["method_catalog"]
    if key not in catalog:
        raise KeyError(f"Unknown method {key!r}. Add it to method_catalog.")
    spec = catalog[key]
    return None if spec is None else dict(spec)


def selected_list(config: dict[str, Any], key: str) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"Config field {key!r} must be a non-empty list.")
    return [str(item) for item in value]


def split_device_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_env(config: dict[str, Any], model: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cfg(config, "ENV", {}).items()})
    if model.get("npu_devices") is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(model["npu_devices"])
    env.update({str(k): str(v) for k, v in model.get("env", {}).items()})
    return env


def build_server_cmd(
    config: dict[str, Any],
    model: dict[str, Any],
    method_spec: dict[str, Any] | None,
    port: int,
) -> list[str]:
    server_args = merge_dicts(
        cfg(config, "COMMON_SERVER_ARGS", {}),
        model.get("server_args"),
    )
    server_args["tensor_parallel_size"] = int(model.get("tp", 1))
    dp = int(model.get("dp", 1))
    if dp > 1:
        server_args["data_parallel_size"] = dp

    cmd = [
        vllm_bin(config),
        "serve",
        str(model["path"]),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--served-model-name",
        str(model["served_model_name"]),
    ]
    add_key_values(cmd, server_args)
    if method_spec:
        cmd.extend(["--speculative-config", json.dumps(method_spec, separators=(",", ":"))])
    return cmd


def build_bench_cmd(
    config: dict[str, Any],
    model: dict[str, Any],
    dataset: dict[str, Any],
    port: int,
    result_dir: Path,
    result_filename: str,
) -> list[str]:
    bench_args = merge_dicts(
        cfg(config, "COMMON_BENCH_ARGS", {}),
        dataset,
        model.get("bench_args"),
    )
    bench_args.pop("name", None)
    if "tokenizer" not in bench_args and Path(str(model["path"])).is_absolute():
        bench_args["tokenizer"] = str(model["path"])

    cmd = [
        vllm_bin(config),
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        str(model["served_model_name"]),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]
    add_key_values(cmd, bench_args)
    return cmd


def wait_ready(port: int, timeout_s: int, process: subprocess.Popen[Any], log_path: Path) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            tail = ""
            if log_path.exists():
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
            raise RuntimeError(f"vLLM server exited before ready with code {process.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status < 500:
                    print(f"Server ready: {url}", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"Server did not become ready within {timeout_s}s: {last_error}")


def stop_process(process: subprocess.Popen[Any], timeout_s: int = 30) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[-1] if data else {}
    return data if isinstance(data, dict) else {}


def validate_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Benchmark result was not written: {path}")
    data = load_result(path)
    failed = int(data.get("failed") or data.get("num_failed_requests") or 0)
    if failed:
        completed = int(data.get("completed") or data.get("num_completed_requests") or 0)
        raise RuntimeError(f"Benchmark completed with failed requests: completed={completed}, failed={failed}")
    return data


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def check_path(label: str, value: Any, errors: list[str], warnings: list[str]) -> None:
    if value is None:
        return
    path = Path(str(value))
    if path.is_absolute():
        if not path.exists():
            errors.append(f"{label} does not exist: {path}")
    else:
        warnings.append(f"{label} is not an absolute path; preflight cannot verify it locally: {value}")


def add_preflight_issue(
    message: str,
    *,
    mode: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if mode == "error":
        errors.append(message)
    elif mode != "off":
        warnings.append(message)


def check_ascend_devices(
    model_key: str,
    model: dict[str, Any],
    *,
    mode: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if mode == "off":
        return

    devices = split_device_ids(model.get("npu_devices"))
    if not devices:
        return

    dev_root = Path("/dev")
    if not dev_root.exists():
        add_preflight_issue(
            f"model {model_key!r} requested NPU devices {devices}, but /dev is not available for Ascend checks",
            mode=mode,
            errors=errors,
            warnings=warnings,
        )
        return

    missing: list[str] = []
    for device in devices:
        if device.isdigit() and not (dev_root / f"davinci{device}").exists():
            missing.append(device)

    if missing:
        add_preflight_issue(
            f"model {model_key!r} requested NPU devices {','.join(missing)}, but /dev/davinci* entries are missing",
            mode=mode,
            errors=errors,
            warnings=warnings,
        )

    if not list(dev_root.glob("davinci*")):
        add_preflight_issue(
            "no /dev/davinci* devices are visible; make sure the Ascend container was started with NPU devices",
            mode=mode,
            errors=errors,
            warnings=warnings,
        )


def run_preflight(config: dict[str, Any], output_dir: Path) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        models = selected_list(config, "models")
        datasets = selected_list(config, "datasets")
        methods = selected_list(config, "methods")
    except Exception as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr, flush=True)
        return 1

    if not vllm_bin_exists(config):
        errors.append(f"vLLM executable not found: {cfg(config, 'VLLM_BIN', 'vllm')}")

    ascend_check_mode = normalize_check_mode(cfg(config, "ASCEND_DEVICE_CHECK", "warn"))
    if ascend_check_mode not in {"off", "warn", "error"}:
        errors.append("ASCEND_DEVICE_CHECK must be one of: off, warn, error")
        ascend_check_mode = "warn"

    case_count = len(models) * len(datasets) * len(methods)
    port_base = int(cfg(config, "PORT_BASE", 19000))
    print("Preflight", flush=True)
    print(f"  vllm: {vllm_bin(config)}", flush=True)
    print(f"  output_dir: {output_dir}", flush=True)
    print(f"  cases: {case_count}", flush=True)
    print(f"  ports: {port_base}..{port_base + case_count - 1}", flush=True)

    for i in range(case_count):
        port = port_base + i
        if not port_is_free(port):
            errors.append(f"port is already in use: {port}")

    for model_key in models:
        try:
            model = model_entry(config, model_key)
        except Exception as exc:
            errors.append(str(exc))
            continue
        check_path(f"model {model_key!r} path", model.get("path"), errors, warnings)
        check_ascend_devices(
            model_key,
            model,
            mode=ascend_check_mode,
            errors=errors,
            warnings=warnings,
        )
        print(
            f"  model {model_key}: path={model.get('path')} tp={model.get('tp')} "
            f"dp={model.get('dp')} npu_devices={model.get('npu_devices')}",
            flush=True,
        )

    for dataset_key in datasets:
        try:
            dataset = dataset_entry(config, dataset_key)
        except Exception as exc:
            errors.append(str(exc))
            continue
        if dataset.get("dataset_path") is not None:
            check_path(f"dataset {dataset_key!r} path", dataset.get("dataset_path"), errors, warnings)
        print(f"  dataset {dataset_key}: dataset_name={dataset.get('dataset_name')}", flush=True)

    for method_key in methods:
        try:
            method_entry(config, method_key)
        except Exception as exc:
            errors.append(str(exc))
            continue
        print(f"  method {method_key}: ok", flush=True)

    if warnings:
        print("\nWarnings:", flush=True)
        for warning in warnings:
            print(f"  {warning}", flush=True)

    if errors:
        print("\nErrors:", file=sys.stderr, flush=True)
        for error in errors:
            print(f"  {error}", file=sys.stderr, flush=True)
        return 1

    print("\nPreflight ok", flush=True)
    return 0


def case_name_for(model_key: str, dataset_key: str, method_key: str) -> str:
    return f"a2_{model_key}_{dataset_key}_{method_key}"


def row_from_result(
    *,
    model_key: str,
    dataset_key: str,
    method_key: str,
    result_path: Path,
    data: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "model": model_key,
        "dataset": dataset_key,
        "method": method_key,
        "result_file": str(result_path),
    }
    for field in SUMMARY_FIELDS:
        if field not in row:
            row[field] = data.get(field, "")
    return row


def find_completed_result(output_dir: Path, case_name: str) -> tuple[Path, dict[str, Any]] | None:
    case_dir = output_dir / case_name
    if not case_dir.exists():
        return None
    for run_dir in sorted((path for path in case_dir.iterdir() if path.is_dir()), reverse=True):
        result_path = run_dir / f"{case_name}.json"
        if not result_path.exists():
            continue
        try:
            return result_path, validate_result(result_path)
        except Exception:
            continue
    return None


def run_case(
    *,
    config: dict[str, Any],
    model_key: str,
    model: dict[str, Any],
    dataset_key: str,
    dataset: dict[str, Any],
    method_key: str,
    method_spec: dict[str, Any] | None,
    output_dir: Path,
    port: int,
    dry_run: bool,
    skip_completed: bool,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    case_name = case_name_for(model_key, dataset_key, method_key)
    if skip_completed and not dry_run:
        completed = find_completed_result(output_dir, case_name)
        if completed is not None:
            result_path, data = completed
            print(f"Skipping completed case: {case_name} -> {result_path}", flush=True)
            return row_from_result(
                model_key=model_key,
                dataset_key=dataset_key,
                method_key=method_key,
                result_path=result_path,
                data=data,
            )

    result_dir = output_dir / case_name / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)
    result_filename = f"{case_name}.json"
    server_log = result_dir / "server.log"
    bench_stdout = result_dir / "bench_stdout.log"
    result_path = result_dir / result_filename

    env = build_env(config, model)
    server_cmd = build_server_cmd(config, model, method_spec, port)
    bench_cmd = build_bench_cmd(config, model, dataset, port, result_dir, result_filename)
    metadata = {
        "case_name": case_name,
        "model": model_key,
        "dataset": dataset_key,
        "method": method_key,
        "started_at_utc": timestamp,
        "port": port,
        "env_overrides": {k: env[k] for k in sorted(set(cfg(config, "ENV", {})) | {"ASCEND_RT_VISIBLE_DEVICES"}) if k in env},
        "server_cmd": server_cmd,
        "bench_cmd": bench_cmd,
        "result_dir": str(result_dir),
    }
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
        return {
            "model": model_key,
            "dataset": dataset_key,
            "method": method_key,
            "result_file": str(result_path),
        }

    print(f"\n==> {case_name}", flush=True)
    server_process: subprocess.Popen[Any] | None = None
    with server_log.open("w", encoding="utf-8") as log:
        try:
            print("+", " ".join(shlex.quote(part) for part in server_cmd), flush=True)
            server_process = subprocess.Popen(
                server_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=REPO_ROOT,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            wait_ready(
                port,
                int(cfg(config, "READY_TIMEOUT_S", 1800)),
                server_process,
                server_log,
            )
            with bench_stdout.open("w", encoding="utf-8") as bench_log:
                run(bench_cmd, env=env, stdout=bench_log)
            data = validate_result(result_path)
        finally:
            if server_process is not None:
                if hasattr(os, "killpg") and server_process.poll() is None:
                    try:
                        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                        server_process.wait(timeout=30)
                    except Exception:
                        stop_process(server_process)
                else:
                    stop_process(server_process)

    return row_from_result(
        model_key=model_key,
        dataset_key=dataset_key,
        method_key=method_key,
        result_path=result_path,
        data=data,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "a2_container_experiments.yaml")
    parser.add_argument("--output-dir", type=Path, help="Override OUTPUT_DIR in the config.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Check vLLM, model paths, dataset paths, methods, and ports without running benchmarks.")
    parser.add_argument("--resume", action="store_true", help="Skip cases that already have a successful result JSON.")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    output_dir = args.output_dir or Path(cfg(config, "OUTPUT_DIR", "results/a2_container"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    if args.preflight:
        return run_preflight(config, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    port = int(cfg(config, "PORT_BASE", 19000))
    stop_on_failure = bool(cfg(config, "STOP_ON_FAILURE", True))
    skip_completed = bool(cfg(config, "SKIP_COMPLETED", False)) or args.resume

    for model_key in selected_list(config, "models"):
        model = model_entry(config, model_key)
        for dataset_key in selected_list(config, "datasets"):
            dataset = dataset_entry(config, dataset_key)
            for method_key in selected_list(config, "methods"):
                method_spec = method_entry(config, method_key)
                try:
                    rows.append(
                        run_case(
                            config=config,
                            model_key=model_key,
                            model=model,
                            dataset_key=dataset_key,
                            dataset=dataset,
                            method_key=method_key,
                            method_spec=method_spec,
                            output_dir=output_dir,
                            port=port,
                            dry_run=args.dry_run,
                            skip_completed=skip_completed,
                        )
                    )
                except Exception as exc:
                    failures.append(f"{model_key}/{dataset_key}/{method_key}: {exc}")
                    print(f"FAILED: {failures[-1]}", file=sys.stderr, flush=True)
                    if stop_on_failure:
                        summary_path = write_summary(rows, output_dir)
                        print(f"Partial summary: {summary_path}", flush=True)
                        return 1
                finally:
                    port += 1

    summary_path = write_summary(rows, output_dir)
    print(f"Summary: {summary_path}", flush=True)
    if failures:
        print("\nFailures:", file=sys.stderr, flush=True)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
