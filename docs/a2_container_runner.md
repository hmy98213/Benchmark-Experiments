# A2 Container Runner

Use this path when you are already inside a vLLM Ascend container.

It does not call `docker run`, `docker exec`, or `docker cp`.  It directly runs:

```bash
vllm serve ...
vllm bench serve ...
```

For every model x dataset x method combination, the runner starts a fresh vLLM
server, runs one benchmark, stops that server, then moves to the next
combination.

## Quick Start

```bash
cd /workspace/suffix-bench
vi configs/a2_container_experiments.yaml
bash run.sh --dry-run
bash run.sh
```

You can also pass a config file directly:

```bash
bash run.sh configs/a2_container_experiments.yaml --dry-run
bash run.sh configs/a2_container_experiments.yaml
```

If a long run stops halfway through, resume without rerunning completed cases:

```bash
bash run.sh configs/a2_container_experiments.yaml --resume
```

Results are written under:

```bash
results/a2_container/
```

The combined CSV summary is:

```bash
results/a2_container/summary.csv
```

## Main Config Fields

Edit `configs/a2_container_experiments.yaml`:

The exact layout is not strict.  The runner only needs three selected lists
(`models`, `datasets`, `methods`) plus catalog entries for the names in those
lists.  The catalog style below is recommended because it lets each model keep
its own path and topology while the top of the file stays easy to edit.

```yaml
models:
  - qwen35_9b
datasets:
  - random64
methods:
  - baseline
  - mtp
  - ngram
  - suffix
  - mtp_ngram_concat
  - mtp_suffix_concat

TP: 1
DP: 1
NPU_DEVICES: "0"
```

Inline YAML lists also work:

```yaml
models: [qwen35_9b, qwen3_32b]
methods: [baseline, mtp, suffix, mtp_suffix_concat]
```

Add model entries under `model_catalog`:

```yaml
model_catalog:
  your_model:
    path: /home/huangmy/models/YourModel
    served_model_name: your-model
```

Top-level `TP`, `DP`, and `NPU_DEVICES` are the defaults.  A model entry can
override them when that model needs a different topology.

For TP8:

```yaml
models:
  - qwen3_32b

model_catalog:
  qwen3_32b:
    path: /home/huangmy/models/Qwen3-32B
    served_model_name: qwen3-32b
    tp: 8
    dp: 1
    npu_devices: "0,1,2,3,4,5,6,7"
```

## Notes

- `NPU_DEVICES` uses container-visible device IDs, not necessarily host IDs.
- If `vllm` is not on `PATH`, set `VLLM_BIN` in the YAML config.
- `--resume` only skips cases with an existing successful result JSON.
- The default config assumes the benchmark repo is at `/workspace/suffix-bench`.
- `datastores384` expects `/workspace/suffix-bench/data/datastores/datastores_mixed_custom.jsonl`.
- `specbench100` expects `/workspace/suffix-bench/data/spec_bench/question.jsonl`.
- If `PyYAML` is unavailable, the runner uses a built-in parser for this config's YAML subset.
