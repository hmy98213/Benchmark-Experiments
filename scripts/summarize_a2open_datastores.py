#!/usr/bin/env python3
"""Render A2 open-model datastore384 results as Markdown."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_a2_container_experiments as runner  # noqa: E402


METHOD_LABELS = {
    "baseline": "baseline",
    "mtp_2": "MTP-2",
    "ngram_2": "ngram-2",
    "suffix_2_cold": "suffix-2 cold",
    "suffix_2_warm": "suffix-2 warm",
    "mtp_ngram_concat_1p1": "MTP+ngram 1+1",
    "mtp_suffix_concat_1p1_cold": "MTP+suffix 1+1 cold",
    "mtp_suffix_concat_1p1_warm": "MTP+suffix 1+1 warm",
    "mtp_ngram_concat_2p2": "MTP+ngram 2+2",
    "mtp_suffix_concat_2p2_cold": "MTP+suffix 2+2 cold",
    "mtp_suffix_concat_2p2_warm": "MTP+suffix 2+2 warm",
}


def fmt_float(value: Any, digits: int = 2) -> str:
    if value in ("", None):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_pct(value: Any) -> str:
    if value in ("", None):
        return "-"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def fmt_speedup(value: Any, baseline: float | None) -> str:
    if baseline is None or baseline <= 0 or value in ("", None):
        return "-"
    try:
        return f"{float(value) / baseline:.2f}x"
    except (TypeError, ValueError, ZeroDivisionError):
        return "-"


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_rows(config_path: Path) -> list[tuple[str, str, str]]:
    config = runner.load_config(config_path)
    rows: list[tuple[str, str, str]] = []
    for model_key in runner.selected_list(config, "models"):
        model = runner.model_entry(config, model_key)
        methods = runner.selected_methods_for_model(config, model_key, model)
        for dataset_key in runner.selected_list(config, "datasets"):
            for method_key in methods:
                method_spec = runner.method_entry(config, method_key)
                for pass_name in runner.bench_pass_names(method_spec):
                    row_method = method_key if pass_name is None else f"{method_key}_{pass_name}"
                    rows.append((model_key, dataset_key, row_method))
    return rows


def render_markdown(summary_rows: list[dict[str, str]], expected: list[tuple[str, str, str]]) -> str:
    by_key = {
        (row["model"], row["dataset"], row["method"]): row
        for row in summary_rows
    }
    baselines: dict[str, float] = {}
    for row in summary_rows:
        if row["method"] != "baseline":
            continue
        try:
            baselines[row["model"]] = float(row["output_throughput"])
        except (TypeError, ValueError):
            pass

    lines = [
        "## A2 Datastore384 Open-Model Results",
        "",
        "Configuration: TP=4, `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, dataset=`datastores384`, `num_prompts=384`, `temperature=0`, `max_concurrency=1`.",
        "",
        "| Model | Method | Completed / Failed | Output tok/s | vs baseline | TPOT ms | Accept rate | Accept len |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    missing: list[tuple[str, str, str]] = []
    for model, dataset, method in expected:
        row = by_key.get((model, dataset, method))
        if row is None:
            missing.append((model, dataset, method))
            continue
        baseline = baselines.get(model)
        lines.append(
            "| {model} | {method} | {done} / {failed} | {out} | {speedup} | {tpot} | {acc_rate} | {acc_len} |".format(
                model=f"`{model}`",
                method=METHOD_LABELS.get(method, method),
                done=row.get("completed") or "-",
                failed=row.get("failed") or "-",
                out=fmt_float(row.get("output_throughput")),
                speedup=fmt_speedup(row.get("output_throughput"), baseline),
                tpot=fmt_float(row.get("mean_tpot_ms")),
                acc_rate=fmt_pct(row.get("spec_decode_acceptance_rate")),
                acc_len=fmt_float(row.get("spec_decode_acceptance_length")),
            )
        )

    lines.extend(["", "### Missing Or Failed Expected Rows", ""])
    if missing:
        for model, dataset, method in missing:
            lines.append(f"- `{model}` / `{dataset}` / `{METHOD_LABELS.get(method, method)}`")
    else:
        lines.append("- None. All expected rows are present in `summary.csv`.")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/a2_datastores384_open_models.yaml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary_rows = read_summary(args.summary_csv)
    expected = expected_rows(args.config)
    markdown = render_markdown(summary_rows, expected)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
