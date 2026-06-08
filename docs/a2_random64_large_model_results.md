# A2 Large Model Random64 Results

Source commit: `10ee2fd experiment results`

Result source:

- `results/a2_container/summary.csv`
- Per-case JSON files under `results/a2_container/*/<timestamp>/`

## Setup

- Machine: A2 / Ascend container
- vLLM: `/usr/local/python3.11.14/bin/vllm`
- Topology: `TP=8`, `DP=1`, `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- Dataset: `random64`
- Requests: 64
- Random input length: 1024
- Random output length: 256
- Random range ratio: 0.1
- Bench settings: `request_rate=inf`, `max_concurrency=4`, `temperature=0`, `ignore_eos=true`
- Server settings: `max_model_len=4096`, `max_num_batched_tokens=32768`, `enforce_eager=true`

Methods:

- `baseline`: no speculative decoding
- `ngram`: `num_speculative_tokens=4`, `prompt_lookup_min=2`, `prompt_lookup_max=5`
- `suffix`: `num_speculative_tokens=4`

Successful runs only include `baseline`, `ngram`, and `suffix`. There are no valid `mtp` or hybrid `mtp -> ngram/suffix` rows in this result set.

## Complete Results

| Model | Method | Completed | Failed | Output tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | Mean ITL ms | Accept rate | Accept len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen35_27b_w8a8_mtp` | baseline | 64 | 0 | 17.86 | 89.26 | 4011.10 | 207.18 | 207.24 |  |  |
| `qwen35_27b_w8a8_mtp` | ngram | 64 | 0 | 25.46 | 127.25 | 980.98 | 151.16 | 411.39 | 84.55% | 4.30 |
| `qwen35_27b_w8a8_mtp` | suffix | 64 | 0 | 36.06 | 180.22 | 806.42 | 102.97 | 335.37 | 87.34% | 3.77 |
| `deepseek_r1_distill_qwen_32b` | baseline | 64 | 0 | 27.21 | 135.88 | 564.18 | 143.59 | 143.60 |  |  |
| `deepseek_r1_distill_qwen_32b` | ngram | 64 | 0 | 49.73 | 248.35 | 447.48 | 77.02 | 143.64 | 68.64% | 3.70 |
| `deepseek_r1_distill_qwen_32b` | suffix | 64 | 0 | 55.45 | 276.93 | 298.87 | 69.39 | 144.31 | 62.19% | 2.38 |
| `glm_45_air_fp8` | baseline | 64 | 0 | 21.20 | 105.96 | 820.39 | 185.07 | 185.06 |  |  |
| `glm_45_air_fp8` | ngram | 64 | 0 | 89.81 | 448.91 | 552.98 | 41.69 | 188.74 | 96.81% | 4.73 |
| `glm_45_air_fp8` | suffix | 64 | 0 | 99.12 | 495.40 | 403.53 | 38.69 | 188.63 | 99.88% | 4.92 |

## Relative To Baseline

| Model | Method | Output speedup | TPOT reduction | TTFT reduction |
|---|---:|---:|---:|---:|
| `qwen35_27b_w8a8_mtp` | ngram | 1.43x | 27.0% | 75.5% |
| `qwen35_27b_w8a8_mtp` | suffix | 2.02x | 50.3% | 79.9% |
| `deepseek_r1_distill_qwen_32b` | ngram | 1.83x | 46.4% | 20.7% |
| `deepseek_r1_distill_qwen_32b` | suffix | 2.04x | 51.7% | 47.0% |
| `glm_45_air_fp8` | ngram | 4.24x | 77.5% | 32.6% |
| `glm_45_air_fp8` | suffix | 4.68x | 79.1% | 50.8% |

## Takeaways

- `suffix` is the best method in all three successful model groups by output throughput and TPOT.
- `ngram` also improves every model over baseline, but is consistently behind `suffix` on this dataset.
- `GLM-4.5-Air-FP8` shows the largest speculative decoding gain: `suffix` reaches 4.68x output throughput over baseline, with 99.88% acceptance rate and 4.92 accepted tokens on average.
- `Qwen3.5-27B-w8a8-mtp` improves from 17.86 output tok/s to 36.06 output tok/s with `suffix`, about 2.02x. Even though this model name includes MTP, this run did not include `mtp` or hybrid methods.
- `DeepSeek-R1-Distill-Qwen-32B` improves from 27.21 output tok/s to 55.45 output tok/s with `suffix`, about 2.04x. Its `suffix` acceptance rate is lower than `ngram`, but end-to-end throughput is still higher.

## Invalid Or Incomplete Attempts

Several directories contain `metadata.json` but no benchmark result JSON. These were not included in the tables above.

- `/models/DeepSeek-R1-0528-GPTQ-Int4-Int8Mix-Compact` failed during vLLM server initialization. The server log reports that the model directory does not contain `configuration_deepseek.py`.
- Earlier selected models such as `qwen3_235b_a22b_instruct_2507_w8a8_quarot` and `glm_47_awq` only have metadata-only attempts in the pulled result set, so there is no valid throughput or latency result for them here.
- Some metadata-only directories for the successful models appear to be interrupted or dry-run/retry artifacts. The authoritative successful rows are the 9 rows in `results/a2_container/summary.csv`.

`fusion_result.json` contains Ascend graph fusion pass counters, not benchmark latency or throughput metrics, so it is not used in the performance tables.
