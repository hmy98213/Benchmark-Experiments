# vLLM Bench Dataset Reference

This note separates two questions:

1. Which dataset loaders are supported by `vllm bench serve`.
2. Which workloads are appropriate for a broader speculative decoding survey.

The current CUDA image in this repo uses vLLM `0.21.0`. Its local CLI reports:

```text
--dataset-name {sharegpt,burstgpt,sonnet,random,random-mm,random-rerank,hf,custom,custom_mm,prefix_repetition,spec_bench,speed_bench}
```

The latest vLLM documentation also lists `custom_audio` and `custom_image`. Those are official latest-doc options, but they are not available in the current `suffix-vllm-cuda:latest` image.

## Official vLLM Dataset Loaders

These are official vLLM benchmark loaders or adapters. That does not mean vLLM owns the datasets; most are external datasets with vLLM-provided parsing logic.

| `--dataset-name` | Data source | Needs local download | Notes for speculative decoding |
| --- | --- | --- | --- |
| `random` | Synthetic, generated from tokenizer vocabulary | No | Good control for fixed ISL/OSL throughput. Not semantically natural; weak for judging draft quality. |
| `prefix_repetition` | Synthetic repeated-prefix prompt generator | No | Good stress case for n-gram / suffix style model-free speculation. It is synthetic and can overstate gains. |
| `sharegpt` | ShareGPT JSON, e.g. ShareGPT Vicuna cleaned split | Yes | Useful open-ended chat workload. Better than `random` for model-based speculative methods. |
| `burstgpt` | BurstGPT CSV | Yes | Useful if modeling bursty serving traces. |
| `sonnet` | Local sonnet text shipped/expected by benchmark code | Usually local | Deprecated-style synthetic/text benchmark; not a main target for this survey. |
| `hf` | Generic Hugging Face dataset adapter | Depends | Important adapter. vLLM docs use it for VisionArena, InstructCoder, AIMO, MT-Bench, Blazedit, ASR, and other datasets. |
| `custom` | Local JSONL with `prompt` and optional `output_tokens` | Yes | Best route for trace replay once we collect agent calls. |
| `custom_mm` | Local multimodal JSONL | Yes | Multimodal trace route. Not central unless testing VLMs. |
| `random-mm` | Synthetic multimodal inputs | No | Useful for VLM serving tests, not central for text speculative decoding. |
| `random-rerank` | Synthetic reranking requests | No | Only relevant to rerank endpoint benchmarking. |
| `spec_bench` | Spec-Bench JSONL from `hemingkx/Spec-Bench` | Yes | Important speculative decoding benchmark. Current repo implements this for H100. |
| `speed_bench` | NVIDIA SPEED-Bench prepared dataset | Yes | Good follow-up: has qualitative and throughput splits with entropy categories. |
| `custom_audio` | Latest vLLM docs option | Yes | Not available in current vLLM `0.21.0` image. |
| `custom_image` | Latest vLLM docs option | Yes | Not available in current vLLM `0.21.0` image. |

## Recommended Dataset Set For This Survey

The current repo has enough data for environment verification, but not enough for a final survey covering all speculative decoding schemes.

Keep as controls:

- `random`: fixed-length throughput control.
- `prefix_repetition`: upper-bound-ish repetition control for n-gram/suffix methods.

Keep as first real benchmark:

- `spec_bench`: standard speculative decoding benchmark; already wired into this repo.

Add before drawing broad conclusions:

- `sharegpt`: natural chat serving workload.
- `hf` + `likaixin/InstructCoder`: code generation workload shown in vLLM's own speculative decoding example.
- `hf` + `vdaita/edit_5k_char` or `vdaita/edit_10k_char`: Blazedit-like code editing workload, useful for repetition-heavy editing.
- `speed_bench`: newer vLLM-documented benchmark with qualitative and throughput splits.

Add for Suffix Decoding paper-style agentic claims:

- SWE-Bench / SWE-Bench Verified trace replay: not a vLLM built-in dataset. Use OpenHands or another coding agent, collect OpenAI-compatible LLM calls, then replay them through `custom`.
- AgenticSQL substitute: original AgenticSQL is proprietary. Use an open Text-to-SQL pipeline over Spider/BIRD-like data, collect stage-wise LLM calls, then replay them through `custom`.

## Practical Interpretation

For a survey across speculative decoding families:

- Model-based speculation, such as EAGLE/MTP-style methods, should be tested on natural prompts like ShareGPT, Spec-Bench, MT-Bench/AIMO/InstructCoder, and SPEED-Bench. Synthetic random prompts can be misleading.
- Model-free lookup methods, such as n-gram and suffix decoding, need repetition-heavy workloads: prefix repetition, code editing, agent traces, and SQL pipeline traces.
- A fair report should include both low-repetition and high-repetition workloads, otherwise the conclusion will be biased toward one family of methods.

