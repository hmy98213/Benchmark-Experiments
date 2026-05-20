# Suffix Decoding Experiment Plan

## Goal

Measure whether Suffix Decoding brings practical serving gains on two target platforms:

- H100 + CUDA vLLM, used as the primary reproducibility platform.
- Atlas A2 / 910B3 + vLLM Ascend, used to verify Ascend feature availability and real deployment behavior.

The main metrics are TPOT, TTFT, ITL, output throughput, request throughput, total token throughput, and acceptance-related logs when available.

## Environment

Docker is the default route. The host account `huangmy` has been added to the `docker` group on both machines.

The local SSH config has been simplified to these aliases:

- `a2`
- `a2-root`
- `h100`
- `h100-root`

The benchmark harness builds small derived images:

- `suffix-vllm-cuda:latest` from `vllm/vllm-openai:latest`
- `suffix-vllm-ascend:latest` from `registry.dev.huawei.com/flash_stor/vllm-ascend:v0.18.0rc1-openeuler`

Both images install `arctic-inference==0.1.2`, because vLLM suffix decoding imports ArcticInference as the suffix proposer implementation.

## Benchmarks

Priority 0 follows the Suffix Decoding paper as closely as possible:

- Spec-Bench: use `vllm bench serve --dataset-name spec_bench`.
- SWE-Bench / SWE-Bench Verified traces: run later as trace-style OpenAI API calls after the serving harness is stable.
- AgenticSQL equivalent: use an open Text-to-SQL multi-stage pipeline if the original proprietary AgenticSQL trace is unavailable.

Priority 1 is platform characterization:

- `random`: general decode stress case.
- `prefix_repetition`: synthetic high-repetition case; this should favor suffix decoding more than random prompts.
- ShareGPT / HumanEval / GSM8K / ARC / BoolQ / AGIEval: good follow-up set for A2, matching the public vLLM Ascend suffix tutorial.

## Methods

Each benchmark should compare:

- Baseline: no speculative decoding.
- N-gram: `{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":5}`.
- Suffix: `{"method":"suffix","num_speculative_tokens":3}`.

EAGLE/MTP are useful follow-ups, but the first pass keeps the matrix small and model-free.

## Existing Cases

H100:

- `configs/cases/h100_qwen3_8b_random_baseline.json`
- `configs/cases/h100_qwen3_8b_random_ngram.json`
- `configs/cases/h100_qwen3_8b_random_suffix.json`
- `configs/cases/h100_qwen3_8b_prefixrep_baseline.json`
- `configs/cases/h100_qwen3_8b_prefixrep_suffix.json`
- `configs/cases/h100_qwen3_8b_specbench_baseline.json`
- `configs/cases/h100_qwen3_8b_specbench_suffix.json`

A2:

- `configs/cases/a2_qwen3_235b_random_baseline.json`
- `configs/cases/a2_qwen3_235b_random_suffix.json`
- `configs/cases/a2_qwen3_235b_prefixrep_baseline.json`
- `configs/cases/a2_qwen3_235b_prefixrep_suffix.json`

## Suggested Run Order

1. Build images on both machines.
2. Run one baseline case per platform with `num_prompts` temporarily lowered if model loading is the first risk.
3. Run suffix cases on the same dataset and concurrency.
4. Run the H100 Spec-Bench matrix.
5. Expand concurrency: `1, 4, 8, 16, 32`.
6. Expand output lengths: `128, 256, 512, 1024`.
7. Add SWE-Bench / agent traces once the OpenAI-compatible serving path is proven stable.

## Commands

Build:

```bash
bash scripts/build_images.sh cuda
bash scripts/build_images.sh ascend
```

Single case:

```bash
python3 scripts/run_single_case.py --case configs/cases/h100_qwen3_8b_random_suffix.json --output-dir results
```

Matrix:

```bash
python3 scripts/run_case_matrix.py --matrix configs/matrices/h100_smoke.json --output-dir results
```

Summary:

```bash
python3 scripts/summarize_results.py results --output results/summary.csv
```

## Reporting Template

For each platform and dataset, report:

- Model and tensor parallel size.
- Dataset, prompt count, request rate, max concurrency, input/output length.
- Baseline metrics.
- Speculative metrics.
- Speedup: baseline TPOT divided by speculative TPOT.
- Throughput gain: speculative output throughput divided by baseline output throughput.
- Stability notes: server startup issues, unsupported options, memory pressure, or acceptance-rate log availability.

## Source Notes

- vLLM supports `vllm bench serve` datasets including `random`, `prefix_repetition`, and `spec_bench`.
- vLLM suffix decoding requires ArcticInference.
- vLLM Ascend has documented suffix speculative decoding on Atlas A2 with the same core `--speculative-config '{"method":"suffix","num_speculative_tokens":3}'` pattern.
