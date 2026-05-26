# vLLM Bench Dataset Flow

This note is based on the CUDA benchmark image used by this repo:

- image: `suffix-vllm-cuda:latest`
- vLLM: `0.21.0`
- package root inside container: `/usr/local/lib/python3.12/dist-packages/vllm`

## What A Dataset Means

For `vllm bench serve`, a dataset is not an accuracy benchmark by default. It is converted into a list of `SampleRequest` objects:

```python
@dataclass
class SampleRequest:
    prompt: str | list[str] | list[dict]
    prompt_len: int
    expected_output_len: int | None
    multi_modal_data: MultiModalDataDict | dict | list[dict] | None = None
    lora_request: LoRARequest | None = None
    request_id: str | None = None
```

Each item becomes one serving request. The benchmark sends `prompt` to the server, uses `expected_output_len` as `max_tokens`, then measures latency and throughput.

Important: `expected_output_len` is a generation length target, not a reference answer. vLLM records generated text, but it does not compare it with ground truth in `bench serve`.

## Main Code Paths

CLI entry:

```text
vllm/entrypoints/cli/benchmark/serve.py
```

Online serving benchmark:

```text
vllm/benchmarks/serve.py
```

Dataset loading and sampling:

```text
vllm/benchmarks/datasets/datasets.py
```

HTTP request implementation:

```text
vllm/benchmarks/lib/endpoint_request_func.py
```

## Loading Samples

`serve.py` calls `get_samples(args, tokenizer)` after tokenizer initialization. That function maps `--dataset-name` to a dataset class.

Relevant mappings:

```python
"spec_bench": lambda: SpecBench(
    dataset_path=args.dataset_path,
    category=args.spec_bench_category,
    disable_shuffle=args.disable_shuffle,
).sample(
    num_requests=args.num_prompts,
    tokenizer=tokenizer,
    output_len=args.spec_bench_output_len,
    enable_multimodal_chat=args.enable_multimodal_chat,
    request_id_prefix=args.request_id_prefix,
    no_oversample=args.no_oversample,
),
"sharegpt": lambda: ShareGPTDataset(
    random_seed=args.seed,
    dataset_path=args.dataset_path,
    disable_shuffle=args.disable_shuffle,
).sample(
    tokenizer=tokenizer,
    num_requests=args.num_prompts,
    output_len=args.sharegpt_output_len,
    enable_multimodal_chat=args.enable_multimodal_chat,
    request_id_prefix=args.request_id_prefix,
    no_oversample=args.no_oversample,
),
"random": lambda: RandomDataset(...).sample(...),
"prefix_repetition": lambda: PrefixRepetitionRandomDataset(...).sample(...),
```

## Synthetic Datasets

`random` and `prefix_repetition` do not download files.

`random` samples token lengths, generates token IDs from tokenizer vocabulary, decodes them into prompt text, then creates `SampleRequest(prompt, prompt_len, expected_output_len)`.

`prefix_repetition` generates synthetic prompts with repeated shared prefixes. This is useful for stressing reuse-oriented optimizations such as prefix/suffix matching, but it is not a real user task dataset.

## File-backed Datasets

### ShareGPT

`ShareGPTDataset` requires a local JSON file:

```python
with open(self.dataset_path, encoding="utf-8") as f:
    self.data = json.load(f)
self.data = [
    entry
    for entry in self.data
    if "conversations" in entry and len(entry["conversations"]) >= 2
]
```

It uses:

```python
prompt = entry["conversations"][0]["value"]
completion = entry["conversations"][1]["value"]
```

The completion is used to estimate output length if `--sharegpt-output-len` is not provided. It is not used as a reference answer for scoring.

### Spec-Bench

`SpecBench` requires a local JSONL file. vLLM's source comment points to:

```bash
wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl
```

The loader checks for a `turns` column and uses the first turn as the prompt:

```python
jsonl_data = pd.read_json(path_or_buf=self.dataset_path, lines=True)
if "turns" not in jsonl_data.columns:
    raise ValueError("JSONL file must contain a 'turns' column.")

for _, row in jsonl_data.iterrows():
    if (not self.category) or (self.category == row["category"]):
        prompt = row["turns"][0]
        self.data.append({"prompt": prompt})
```

Then it reuses `CustomDataset.sample()`, which applies the chat template and uses `--spec-bench-output-len` as output length.

## Sending Requests

For the default `vllm` backend, `endpoint_request_func.py` maps:

```python
ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
}
```

`async_request_openai_completions()` sends:

```python
payload = {
    "model": request_func_input.model_name
    if request_func_input.model_name
    else request_func_input.model,
    "prompt": request_func_input.prompt,
    "repetition_penalty": 1.0,
    "max_tokens": request_func_input.output_len,
    "logprobs": request_func_input.logprobs,
    "stream": True,
    "stream_options": {
        "include_usage": True,
    },
}
```

It measures:

- latency
- time to first token
- inter-token latency
- output token count
- generated text
- request errors

## Result Metrics

`calculate_metrics()` aggregates successful requests and token counts. It does not evaluate semantic correctness.

The output JSON includes fields such as:

- `duration`
- `completed`
- `failed`
- `total_input_tokens`
- `total_output_tokens`
- `request_throughput`
- `output_throughput`
- `total_token_throughput`
- `input_lens`
- `output_lens`
- `ttfts`
- `itls`
- `generated_texts`
- `errors`

If speculative decoding metrics are available from the server, vLLM also records acceptance statistics:

- `spec_decode_acceptance_rate`
- `spec_decode_acceptance_length`
- `spec_decode_num_drafts`

## Repo Data Setup

This repo provides a helper for Spec-Bench:

```bash
bash scripts/download_bench_data.sh
```

The H100 Spec-Bench cases point to the mounted container path:

```text
/workspace/suffix-bench/data/spec_bench/question.jsonl
```

