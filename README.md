# Suffix Decoding Benchmark Harness

This repository contains a Docker-based benchmark harness for measuring vLLM speculative decoding on:

- `h100`: NVIDIA H100, CUDA vLLM.
- `a2`: Atlas A2 / 910B3, vLLM Ascend.

The scripts are configuration-driven. A single case starts one server container, waits for `/v1/models`, runs `vllm bench serve`, stores the benchmark JSON/logs, and then stops the container. A matrix is just a list of case JSON files.

## Layout

- `docker/`: tiny derived images that add `arctic-inference` to the base vLLM images.
- `configs/cases/`: one runnable benchmark case per JSON file.
- `configs/matrices/`: ordered case lists for repeated runs.
- `scripts/run_single_case.py`: run exactly one case.
- `scripts/run_case_matrix.py`: run a case matrix sequentially.
- `scripts/summarize_results.py`: collect vLLM benchmark JSON files into CSV.
- `docs/experiment_plan.md`: experiment design, platform notes, and reporting plan.

## Build Images

On H100:

```bash
cd ~/suffix-decoding-bench
bash scripts/build_images.sh cuda
```

On A2:

```bash
cd ~/suffix-decoding-bench
bash scripts/build_images.sh ascend
```

The default base images are:

- CUDA: `vllm/vllm-openai:latest`
- Ascend: `registry.dev.huawei.com/flash_stor/vllm-ascend:v0.18.0rc1-openeuler`

Override them with `CUDA_BASE_IMAGE=...` or `ASCEND_BASE_IMAGE=...`.

## Run One Case

```bash
python3 scripts/run_single_case.py \
  --case configs/cases/h100_qwen3_8b_random_suffix.json \
  --output-dir results
```

```bash
python3 scripts/run_single_case.py \
  --case configs/cases/a2_qwen3_235b_random_suffix.json \
  --output-dir results
```

## Run A Matrix

```bash
python3 scripts/run_case_matrix.py \
  --matrix configs/matrices/h100_smoke.json \
  --output-dir results
```

```bash
python3 scripts/run_case_matrix.py \
  --matrix configs/matrices/a2_smoke.json \
  --output-dir results
```

## Summarize

```bash
python3 scripts/summarize_results.py results --output results/summary.csv
```

## Notes

- The scripts assume the user can run Docker without sudo.
- The server containers use host networking and unique ports per case.
- Suffix decoding requires `arctic-inference`, installed in the derived images.
- Current A2 configs use the existing local model `/data/Qwen3-235B-A22B-Instruct-2507-w8a8-QuaRot`.
- Current H100 configs use `Qwen/Qwen3-8B`; the first run will download it into the mounted Hugging Face cache.
