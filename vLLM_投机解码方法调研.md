# vLLM 投机解码方法调研

## 一句话结论

vLLM 里的投机解码不是一个单独算法，而是一套统一的 **draft-then-verify** 框架：先用便宜的 proposer 生成多个 draft token，再让 target model 一次 forward 验证这些 token，接受最长正确前缀，从而减少大模型逐 token decode 的串行次数。

当前 vLLM 官方文档重点支持的用户可用方法包括：`Draft Model`、`EAGLE / EAGLE3`、`MTP`、`PARD`、`N-Gram`、`Suffix Decoding`。源码和 Speculators 文档里还可以看到 `DFlash`、`ngram_gpu` 等入口，其中 `DFlash` 是 vLLM Speculators 当前重点支持的新方向。


## TL;DR

- 投机解码的经典收益公式来自 Leviathan et al.：`Speedup ≈ (1 - α^(γ+1)) / ((1 - α)(1 + γc))`，其中 `α` 是 draft token 接受概率，`γ` 是每轮 draft token 数，`c` 是 draft 单步成本相对 target 单步成本。
- `EAGLE/EAGLE3`、`MTP`、`PARD`、`DFlash` 是 model-based / learned proposer，通常收益更高，但依赖额外权重、模型适配或训练。
- `N-Gram`、`Suffix Decoding` 是 training-free / retrieval-style proposer，门槛低、显存友好，适合代码编辑、模板输出、agent 循环、RAG 复制等重复性强的 workload。
- `Draft Model` 是最经典方案，理论清晰，但需要小模型和 target 模型 tokenizer / vocab / 分布足够接近。
- `PARD` 不是一种全新 verifier，而是把 draft model 的自回归 draft 变成 parallel draft，降低 draft 阶段串行开销。
- `DFlash` 是 vLLM Speculators 新方向，用 diffusion-style drafter 一次生成 token block，目标是解决 learned proposer 自身仍然自回归的问题。
- 本次还实现了一个实验性 `MTP -> ngram/suffix` 拼接 proposer：先用 MTP 生成第一段 draft，再把临时上下文交给 ngram 或 suffix 延长 draft，目的是验证 learned proposer 与 retrieval-style proposer 是否能叠加收益。
- vLLM 的核心实现位置是 `SpeculativeConfig`、`GPUModelRunner` 的 drafter 选择、各类 proposer、`SpecDecodeMetadata`、`RejectionSampler` 和 spec decode metrics。

## 0. 调研范围

这份报告关注 vLLM 官方支持的投机解码方法。

纳入重点调研：

- `draft_model`
- `eagle` / `eagle3`
- `mtp`
- `draft_model + parallel_drafting=true`，也就是 PARD 用法
- `ngram` / `ngram_gpu`
- `suffix`
- `dflash`
- 实验性 `mtp_ngram_concat` / `mtp_suffix_concat`

参考：

- [vLLM v0.21.0 Speculative Decoding 总览](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/)
- [vLLM `SpeculativeConfig` 方法枚举](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L34-L68)
- [vLLM GPUModelRunner 选择 drafter 的源码](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/worker/gpu_model_runner.py#L535-L610)

## 1. 投机解码共同范式

### 1.1 标准流程

投机解码的标准流程可以写成：

1. `propose`：用一个便宜的 proposer 生成 `k` 个 draft token。
2. `verify`：把 draft token 拼到当前上下文后，让 target model 一次 forward 计算多个位置的 logits。
3. `accept/reject`：从左到右验证 draft token，接受最长正确前缀。
4. `commit`：把接受 token 写回 request state 和 KV cache。
5. `repeat`：从最后接受位置继续下一轮。

这里的 proposer 可以是小模型、EAGLE head、MTP head、PARD 并行 draft model、n-gram 匹配器、suffix tree 或 DFlash block drafter。

### 1.2 为什么能快

LLM decode 通常是 memory-bound：每生成一个 token，都要读一遍大模型权重和已有 KV。投机解码的想法是，如果 target model 一次 forward 能验证多个 token，那么每轮 decode 能推进的 token 数就从 `1` 变成 `1 + accepted_draft_tokens`。

收益成立需要几个条件：

- proposer 足够便宜。
- draft token 接受率足够高。
- verification 的额外 query length 不会让 attention / KV / CUDA graph 开销爆掉。
- 当前服务处于中低 QPS 或 memory-bound 状态，而不是 target model 已经被大 batch 喂满。

vLLM 官方文档也把 speculative decoding 的适用场景定位为：降低中低 QPS、memory-bound workload 的 inter-token latency。

参考：

- [vLLM Speculative Decoding 总览](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)

### 1.3 收益公式

根据 `Fast Inference from Transformers via Speculative Decoding`，可以用下面这个公式估算理想 speedup：

```text
Speedup ≈ (1 - α^(γ+1)) / ((1 - α)(1 + γc))
```

符号含义：

- `γ`：每轮先 draft 的 token 数，对应 vLLM 里的 `num_speculative_tokens`。
- `α`：draft token 的平均接受概率，越接近 `1`，说明 proposer 和 target model 越一致。
- `c`：draft model 单步成本 / target model 单步成本。对于 `draft_model`、`EAGLE`、`MTP` 等 learned proposer，`c` 包含额外 forward、KV cache、调度开销；对于 `ngram` / `suffix`，`c` 通常更小，但不是严格为 0。

这个公式的直觉是：每轮 speculative decoding 平均能推进 `(1 - α^(γ+1)) / (1 - α)` 个 token，但也要付出 `1` 次 target verification 加 `γc` 的 draft 成本。论文还给出一个简单判断：如果 `α > c`，就存在某个 `γ` 可以带来加速；当 `γ=1` 时，公式退化为 `(1 + α) / (1 + c)`。

在 vLLM 里使用这个公式时，更适合把它看成上界或选型指南，而不是精确预测。真实收益还要扣掉 verification query length 变长、speculative KV / lookahead slots、CUDA graph、batch 调度、rejection sampler 和服务负载变化带来的开销。

参考：

- [Leviathan et al., Fast Inference from Transformers via Speculative Decoding, PMLR](https://proceedings.mlr.press/v202/leviathan23a.html)
- [论文 PDF，Theorem 3.8 给出 expected improvement factor](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)

### 1.4 投机解码的 lossless 性质

投机解码常见误解是“它会不会改变模型输出”。理论上，使用正确的 rejection sampling 时，投机解码采样结果应与直接从 target model 采样一致。vLLM 文档也把这一点称为 lossless guarantee，但同时提醒浮点误差、batch size、数值稳定性可能造成运行间输出差异。

工程上可以理解为：

- greedy 模式下，draft token 要和 target token 一致才被接受。
- sampling 模式下，rejection sampler 用 target/draft 分布修正采样，保持目标分布。
- 如果实现或数值不同，logprob 稳定性不一定完全一致。

参考：

- [vLLM 文档，Lossless guarantees of Speculative Decoding](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding)
- [vLLM `RejectionSampler` warmup 调用示例](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/worker/gpu_model_runner.py#L5978-L5983)

### 1.5 关键指标

| 指标 | 含义 | 用处 |
| --- | --- | --- |
| `num_drafts` | 发生了多少次 draft | 衡量 speculative loop 活跃度 |
| `num_draft_tokens` | proposer 生成了多少 draft token | 衡量 proposer 工作量 |
| `num_accepted_tokens` | draft token 中有多少被 target 接受 | 衡量真实推进量 |
| `acceptance rate` | `accepted / drafted` | 衡量 proposer 质量 |
| `mean acceptance length` | 平均每轮推进 token 数，vLLM 里 conventionally 包含 bonus token | 和 decode step 减少最相关 |
| `per-position acceptance` | 第 1/2/3/... 个 draft token 的接受概率 | 用于调 `num_speculative_tokens` |
| `draft throughput` | draft token/s | 衡量 proposer 自身是否成为瓶颈 |
| `accepted throughput` | accepted token/s | 更接近端到端收益 |

vLLM 源码里 `SpecDecodingStats` 和 `SpecDecodingLogging` 正是围绕这些指标组织的。

参考：

- [vLLM spec decode metrics 源码](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/metrics.py#L18-L116)

## 2. vLLM 可用方法全景表

### 2.1 用户文档主线方法

| 方法 | vLLM 配置入口 | 是否需要额外权重 | 是否需要训练/适配 | 典型收益 | 典型场景 |
| --- | --- | --- | --- | --- | --- |
| Draft Model | `method="draft_model"` | 需要小模型 | 通常不需要自己训练，但要模型族适配 | 高 | 通用 decode 加速 |
| EAGLE / EAGLE3 | `method="eagle"` / `"eagle3"` | 需要 EAGLE speculator | 需要 speculator 权重，官方/社区可用 | 高 | 通用聊天、代码、数学 |
| MTP | `method="mtp"` | 模型自带或 assistant checkpoint | 依赖模型原生 MTP | 高 | DeepSeek/MiMo/Gemma4/Qwen3 Next 等原生 MTP 模型 |
| PARD | `method="draft_model", parallel_drafting=true` | 需要 PARD 权重 | PARD adaptation | 高 | draft 阶段串行开销明显时 |
| N-Gram | `method="ngram"` | 不需要 | 不需要 | 低到中 | prompt/输出重复明显 |
| Suffix Decoding | `method="suffix"` | 不需要 draft model，但需要 Arctic Inference 包 | 不需要训练 | 低到中 | 高重复 agentic/code/RL rollout |

参考：

- [vLLM v0.21.0 方法选择表](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/#method-selection-at-a-glance)

### 2.2 源码/Speculators 里还能看到的方法

| 方法 | 状态 | 说明 |
| --- | --- | --- |
| `dflash` | vLLM Speculators 正式文档支持，`SpeculativeConfig` 和 `DFlashProposer` 中可见 | diffusion-style block drafter，一次生成 token block |
| `ngram_gpu` | 源码入口可见 | GPU 版 n-gram proposer，减少 CPU / D2H 开销 |
| `mtp_ngram_concat` / `mtp_suffix_concat` | 本次实验性本地补丁 | 先走 MTP proposer，再用临时上下文调用 ngram 或 suffix 追加 tail draft，最后复用原 verifier |

参考：

- [vLLM `SpeculativeMethod` 源码](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L59-L68)
- [vLLM drafter selection 源码](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/worker/gpu_model_runner.py#L551-L607)
- [vLLM Speculators Getting Started](https://docs.vllm.ai/projects/speculators/en/stable/user_guide/getting_started/)

## 3. vLLM 内部实现主线

### 3.1 配置层：`SpeculativeConfig`

`SpeculativeConfig` 是 vLLM speculative decoding 的统一配置入口。核心字段包括：

| 字段 | 作用 |
| --- | --- |
| `method` | 指定方法，例如 `draft_model`、`eagle3`、`mtp`、`ngram`、`suffix`、`dflash` |
| `model` | draft model / speculator / assistant checkpoint / custom proposer |
| `num_speculative_tokens` | 每轮最多 draft token 数 |
| `draft_tensor_parallel_size` | draft/speculator 的 tensor parallel size |
| `parallel_drafting` | PARD / parallel draft 开关 |
| `prompt_lookup_min/max` | n-gram 匹配窗口 |
| `suffix_decoding_*` | suffix tree 深度、global cache 请求数、概率阈值等 |
| `rejection_sample_method` | rejection sampling 策略 |
| `draft_sample_method` | draft 采样方式 |

`SpeculativeConfig` 还会自动推断 method。例如：

- `model` 是 `ngram` 或 `[ngram]` 时，归一化为 `ngram`。
- MTP 具体 model type 会被统一替换为 `mtp`。
- 模型名包含 `eagle3` / `dflash` 时，会自动推断对应 method。

参考：

- [vLLM `SpeculativeConfig` 字段](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L72-L185)
- [vLLM method inference 源码](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L520-L753)

### 3.2 Runner 层：选择 drafter

vLLM V1 的 `GPUModelRunner` 会根据 `speculative_config.method` 创建不同 drafter：

- `ngram` -> `NgramProposer`
- `ngram_gpu` -> `NgramProposerGPU`
- `suffix` -> `SuffixDecodingProposer`
- `draft_model` -> `DraftModelProposer`
- `eagle` / `eagle3` / `mtp` -> `EagleProposer` 或 `Gemma4Proposer`
- `dflash` -> `DFlashProposer`

参考：

- [vLLM `GPUModelRunner` drafter 初始化](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/worker/gpu_model_runner.py#L535-L610)

### 3.3 Proposer 层：生成 draft token

vLLM learned proposer 大多复用 `SpecDecodeBaseProposer`。它的 `propose(...)` 接收 target token、position、hidden states、采样 metadata、attention metadata 等，然后执行 draft model forward，输出 `[batch_size, num_speculative_tokens]` 的 `draft_token_ids`。

检索式 proposer 则更直接：

- `NgramProposer` 从 token history 中找最长 n-gram match。
- `SuffixDecodingProposer` 调 Arctic Inference 的 suffix cache。

参考：

- [vLLM `SpecDecodeBaseProposer.propose`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/llm_base_proposer.py#L421-L638)
- [vLLM propose dispatch](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/worker/gpu_model_runner.py#L4737-L4866)

### 3.4 Verifier 层：target model 一次验证

draft token 生成后，vLLM 会构造 `SpecDecodeMetadata`，把每个 request 的 draft token pad 成 batch tensor。target model 用扩展后的 query length 做 verification forward，然后 `RejectionSampler` 根据 target logits 和必要的 draft probs 进行接受/拒绝。

参考：

- [vLLM `SpecDecodeMetadata`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/metadata.py#L10-L53)
- [vLLM `RejectionSampler` 相关入口](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/sample/rejection_sampler.py)

## 4. 方法一：Draft Model

### 4.1 What

`Draft Model` 是最经典的 speculative decoding 方案：使用一个更小、更快的语言模型作为 drafter。这个小模型按当前上下文自回归生成若干个 draft token，然后由大 target model 一次性验证。

图示可以先看 NVIDIA 的 draft-target 动图：小模型先给出一串候选 token，target model 把这串候选接到上下文后做一次 verification forward，接受最长正确前缀。

![Draft-target speculative decoding 流程](assets/draft_target_approach.gif)

vLLM 配置示例：

```python
LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,
        "method": "draft_model",
    },
)
```

参考：[vLLM Draft Models 文档](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/draft_model/)

### 4.2 Why

大模型 decode 每 token 一次 forward，很容易 memory-bound。小 draft model 的 forward 更便宜，如果它猜出的 token 大部分能被 target 接受，就能减少 target model 的 decode step 数。

这里最容易疑惑的是：target 为什么能“一次验证多个 token”？关键不是 target 有了新的预测能力，而是 verification 阶段把 draft token 拼进序列，用 causal attention mask 同时算出多个位置的 logits。

假设 prefix 后面 draft 出 `d1 d2 d3 d4`。target verifier 看到的逻辑序列是：

```text
[prefix tokens] d1 d2 d3 d4
```

mask 不是让所有 draft token 互相乱看，而是普通 decoder causal mask 的扩展形式：第 `i` 个 draft 位置只能看 prefix 和它左边已经假定成立的 draft token，不能看右边未来 token。

```text
             可 attend 的 key/value
query/logit   prefix   d1   d2   d3   d4
-----------   ------   --   --   --   --
d1 位置          1      1    0    0    0
d2 位置          1      1    1    0    0
d3 位置          1      1    1    1    0
d4 位置          1      1    1    1    1
```

因此，这次 forward 会并行得到“在 prefix+d1 后预测 d2”“在 prefix+d1+d2 后预测 d3”……这些分布。验证器再从左到右比较 draft token 和 target 分布，一旦某个 token 被拒绝，后面的 draft token 也全部丢弃。图里 rejection sampling 的过程如下：

![Target model parallel verification and rejection sampling](assets/draft_verification_mask.gif)

如果 drafter 不是只给一条链，而是给一棵候选树，verification 的思想仍然一样，只是 attention mask 从“线性下三角”变成“祖先路径可见”。例如候选树是：

```text
prefix
├─ a
│  ├─ b
│  └─ c
└─ x
   └─ y
```

可以把树节点 flatten 成 `[a, b, c, x, y]` 后一次送进 target model，但每个 query 位置只能 attend 到 prefix 和自己所在路径上的祖先节点，不能看到兄弟节点、堂兄弟节点或其他分支：

```text
             可 attend 的 key/value
query/logit   prefix   a   b   c   x   y
-----------   ------   -   -   -   -   -
a 节点           1      1   0   0   0   0
b 节点           1      1   1   0   0   0
c 节点           1      1   0   1   0   0
x 节点           1      0   0   0   1   0
y 节点           1      0   0   0   1   1
```

这样一次 target forward 得到的是整棵树上各个节点对应路径条件下的 logits。验证时再沿树走：第一层 token `a/x` 由 prefix 上的 target 分布验证；如果 `a` 被接受，就用 `a` 节点的 logits 去验证它的孩子 `b/c`；如果走到某个节点被拒绝，就从该节点停止，并按 rejection sampling 规则采样 fallback token。这个 mask 的作用是保证每个节点的 hidden state 等价于“只输入从 root 到该节点这一条序列”时 target model 会得到的 hidden state，因此多分支候选也可以在一个 batch/一次 forward 里验证。

优点：

- 理论和 rejection sampling 机制最清楚。
- 不要求 target model 原生带 MTP head。
- 可以复用已有小模型作为 draft model。

缺点：

- 需要额外加载一个 draft model，占显存和初始化时间。
- draft model 与 target model tokenizer / vocab / 分布不匹配时，接受率会很差。
- draft model 自身仍然是自回归生成 `k` 个 token，`k` 大时 draft latency 也会上升。

### 4.3 How in vLLM

实现链路：

1. `SpeculativeConfig` 检测 `method="draft_model"`。
2. vLLM 为 draft model 创建 `draft_model_config` 和 `draft_parallel_config`。
3. `GPUModelRunner` 创建 `DraftModelProposer`。
4. `DraftModelProposer` 继承 `SpecDecodeBaseProposer`，实际 draft 逻辑走通用 learned proposer 路径。
5. verifier 仍然是 target model + `RejectionSampler`。

源码参考：

- [vLLM `DraftModelProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/draft_model.py#L17-L56)
- [vLLM `SpecDecodeBaseProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/llm_base_proposer.py#L55-L638)

### 4.4 适合什么场景

适合：

- 有同模型族小模型，例如 Qwen3-8B + Qwen3-0.6B。
- 请求 QPS 中低，目标是降低 inter-token latency。
- draft model 足够便宜，且接受率能稳定。

不适合：

- 高 QPS 下 target model 已被大 batch 吃满。
- draft model 占用显存导致 target batch size 下降。
- tokenizer/vocab 不匹配或小模型分布偏差大。

## 5. 方法二：EAGLE / EAGLE3

### 5.1 What

`EAGLE` 是 feature-level speculative decoding：它不是单纯用小模型在 token 层预测，而是利用 target model 的 hidden states 训练轻量 drafter 来预测未来 token。vLLM 支持 `method="eagle"` 和 `method="eagle3"`。

EAGLE 论文里的核心流程如下：target LLM 的 hidden feature 和 token embedding 一起喂给轻量 autoregression head，head 继续外推未来 feature/token，再交给 target verifier 验证。

![EAGLE pipeline](assets/eagle_pipeline.png)

vLLM / NVIDIA 对 EAGLE-3 的工程化介绍也强调：EAGLE head 不是独立小模型，而是从 target 的内部层抽取 feature，生成 token tree，再送回 target verifier。

![EAGLE-3 drafting mechanism](assets/eagle3_drafting_mechanism.gif)

vLLM 配置示例：

```python
LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    speculative_config={
        "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        "draft_tensor_parallel_size": 2,
        "num_speculative_tokens": 2,
        "method": "eagle3",
    },
)
```

参考：

- [vLLM EAGLE Draft Models 文档](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/eagle/)
- [vLLM Speculators Eagle3 文档](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/eagle3/)

### 5.2 Why

经典 draft model 的问题是 token-level 小模型未必足够贴近 target。EAGLE 的关键想法是利用 target model 的 hidden states，让 drafter 在更贴近 target 内部表示的位置做预测，从而提高接受率。

EAGLE3 的工程优势：

- 通用性强，是当前 vLLM / Speculators 生态重点支持方向。
- 可以使用预训练 speculator 权重。
- vLLM Speculators 支持自己训练 Eagle3 speculator。

代价：

- 需要额外 speculator 权重。
- 需要 target model 暴露 hidden states。
- 对模型结构、hidden state layer、TP、CUDA graph、attention backend 有更多工程约束。

### 5.3 How in vLLM

实现链路：

1. `SpeculativeConfig` 识别 `method="eagle"` 或 `method="eagle3"`。
2. `GPUModelRunner` 创建 `EagleProposer`。
3. target model forward 时保留 hidden states。
4. `EagleProposer` 使用 hidden states + token 信息生成 draft token。
5. target model verification。

EAGLE3 在 Speculators 文档里的流程是：

1. target model 产出选定层 hidden states。
2. 这些 hidden states 被拼接并投影，和 token embedding 一起输入 draft model。
3. draft model 自回归预测 K 个 token。
4. target model 一次验证 K 个 token。

源码参考：

- [vLLM `EagleProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/eagle.py#L10-L32)
- [vLLM `use_eagle`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L1059-L1060)
- [vLLM Eagle3 hidden state layer 相关逻辑](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/worker/gpu_model_runner.py#L5182-L5214)

### 5.4 适合什么场景

适合：

- 通用聊天、代码、数学推理等不一定有强文本重复的 workload。
- 希望比 n-gram/suffix 更稳定地提速。
- 有对应 target model 的 EAGLE/EAGLE3 speculator。

不适合：

- 没有可用 speculator，又不想训练。
- 部署链路非常保守，不想引入 hidden state 依赖。
- 显存刚好卡满，放不下 speculator。

### 5.5 参数和调优

重点参数：

- `model`：EAGLE/EAGLE3 speculator 权重。
- `method`：`eagle` 或 `eagle3`。
- `num_speculative_tokens`：draft token 数，过大可能 per-position acceptance 快速下降。
- `draft_tensor_parallel_size`：speculator 的 TP。

调优建议：

- 先从 `num_speculative_tokens=2~4` 开始。
- 看 per-position acceptance，如果第 3/4 位接受率很低，继续加深 draft 没意义。
- 如果 speculator 本身耗时占比高，考虑 PARD / DFlash 这类降低 draft 串行开销的方法。

参考论文：

- [EAGLE](https://arxiv.org/abs/2401.15077)
- [EAGLE-2](https://arxiv.org/abs/2406.16858)
- [EAGLE-3](https://arxiv.org/abs/2503.01840)

## 6. 方法三：MTP

### 6.1 What

`MTP` 是 Multi-Token Prediction。它不是外接小模型，而是 target model 自身包含原生多 token 预测能力，或者通过 assistant checkpoint 走 vLLM 的 MTP path。

下面这张图来自 `Better & Faster Large Language Models via Multi-token Prediction` / Graphcore Research Blog，展示的是 generic MTP：共享 trunk 产生上下文表示，多个 future-token heads 同时预测后续 token。在这种设计里，如果有 4 个 future-token heads，确实可以在一次 drafter forward 中得到 4 个未来位置的 draft logits，再交给 target verifier 验证。

![Multi-token prediction overview](assets/mtp_main_fig.png)

但要注意：DeepSeek-V3 的 MTP 不是这张图里的“多个 independent output heads 并行预测”结构。DeepSeek-V3 Technical Report 明确说，它不同于 Gloeckle et al. 的并行 heads，而是使用 sequential MTP modules，并在每个 prediction depth 保留完整 causal chain。也就是说，DeepSeek 的第 `k` 个 MTP depth 会基于上一 depth 的表示和前面 token embedding 继续往前滚，工程上更像重复调用轻量 MTP layer/module 逐步生成 draft token，而不是一次用 4 个独立 head 同时吐出 4 个 token。

因此可以把 MTP 分成两类理解：

| 类型 | draft 生成方式 | 代表 |
| --- | --- | --- |
| 并行 heads MTP | 一个 trunk hidden state 接多个 future-token heads，一次得到多个未来 token 分布 | Gloeckle et al. / Graphcore 图里的 generic MTP |
| 顺序 modules MTP | 每个 MTP depth/layer 预测一步，再把结果继续喂给下一 depth/layer，保留 causal chain | DeepSeek-V3 MTP、vLLM 的 DeepSeek MTP 路径 |

放在 EAGLE 后面理解会更顺：EAGLE 是“利用 target hidden states 训练一个轻量 head/assistant 去猜未来 token”，MTP 则像是把这种未来 token 预测能力更深地做进模型自身或同族 assistant checkpoint 里。工程上 vLLM 也把大多数 MTP 纳入 EAGLE-style learned proposer 基础设施；差别是 MTP 的约束更偏模型原生能力，而不是外接一个通用 EAGLE speculator。

vLLM 配置示例：

```python
LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
```

Gemma 4 assistant 示例：

```bash
vllm serve google/gemma-4-E2B-it \
  --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

参考：[vLLM MTP 文档](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/mtp/)

### 6.2 Why

MTP 的优势是配置和资源上比额外 draft model 更省：如果目标模型训练时已经带 MTP 模块，就不需要另找小模型，也不需要 draft-target tokenizer 对齐。

优点：

- 与 target model 原生绑定，通常分布更贴近。
- 不需要单独选择小模型。
- vLLM 可以从 target model 或 assistant checkpoint 推断路径。

缺点：

- 只适用于原生支持 MTP 的模型族。
- 不是所有 MTP 都支持较大的 `num_speculative_tokens`。

### 6.3 How in vLLM

`SpeculativeConfig` 里 `MTPModelTypes` 包含多个模型类型，例如 `deepseek_mtp`、`mimo_mtp`、`qwen3_next_mtp`、`gemma4_mtp` 等。配置中具体 MTP model type 会被统一归一化为 `method="mtp"`。

实现上，vLLM 把 `mtp` 纳入 `use_eagle()` 路径，也就是说很多 MTP proposer 复用 EAGLE/learned proposer 的基础设施。Gemma4 则有专门的 `Gemma4Proposer`。

以 DeepSeek MTP 为例，vLLM 的模型实现里会读取 `config.num_nextn_predict_layers`，构造对应数量的 MTP layers；`compute_logits(...)` 会用 `spec_step_idx % num_mtp_layers` 选择当前 speculative step 使用哪个 MTP layer。因此 `num_speculative_tokens` 更像“要让 proposer 连续 draft 几步”，不是简单等价于“模型有几个并行 head 就一次出几个 token”。最终这些 draft token 仍然会被 target model 一次 verification forward 验证。

源码参考：

- [vLLM `MTPModelTypes`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L34-L53)
- [vLLM MTP method 归一化](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L541-L545)
- [vLLM `use_gemma4_mtp`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L1051-L1057)
- [vLLM `Gemma4Proposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/gemma4.py#L31-L69)
- [vLLM `DeepSeekMTP`](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/deepseek_mtp/)
- [DeepSeek-V3 Technical Report, Multi-Token Prediction](https://ar5iv.labs.arxiv.org/html/2412.19437v2#S2.SS2)

### 6.4 适合什么场景

适合：

- 目标模型原生支持 MTP。
- 你希望 model-based speculation，但不想额外选择 draft model。
- 你使用的 vLLM 版本明确支持该模型族的 MTP path。

不适合：

- 普通 checkpoint 没有 MTP head。
- 模型经过转换/量化后丢失或破坏 MTP 权重。
- 想跨模型族复用 speculator。

## 7. 方法四：PARD / Parallel Draft Model

### 7.1 What

PARD 是 Parallel Draft Model。它不是改变 target verifier，而是把原本自回归的 draft model 适配成一次 forward 生成多个 draft token 的并行 drafter。在 vLLM 里，PARD 通过 `method="draft_model"` 加 `parallel_drafting=true` 使用。

PARD 论文/AMD 技术博客里的核心对比是：普通 AR draft model 要一格一格生成 draft token，而 PARD 在 draft 输入里放入多个 mask slot，让 ParaDraft model 并行预测多个未来位置。

![PARD parallel draft model](assets/pard_parallel_draft.jpg)

因此 MTP 和 PARD 可以放在一起看：MTP 是模型内生/assistant 形态的多 token 预测，PARD 是把外部 draft model 的 `k` 步自回归 draft 变成并行 draft。二者都在减少 proposer 自身的串行开销，但 verifier 仍然是 target model 的 speculative verification。

vLLM 配置示例：

```python
LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "model": "amd/PARD-Qwen3-0.6B",
        "num_speculative_tokens": 12,
        "method": "draft_model",
        "parallel_drafting": True,
    },
)
```

参考：[vLLM Parallel Draft Models 文档](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/parallel_draft_model/)

### 7.2 Why

经典 draft model 的隐性瓶颈是：draft model 也要自回归生成 `k` 个 token。即使小模型快，`k` 较大时 draft 阶段也会串行累积延迟。PARD 把 draft 过程并行化，目标是让 `num_speculative_tokens` 增大时 draft latency 不线性增长。

优点：

- 可以更大胆地使用较大的 draft depth。
- 适合低延迟场景。
- vLLM 文档已有在线/离线示例和预训练权重入口。

缺点：

- 需要 PARD adaptation 后的权重。
- 不是任意 draft model 打开 `parallel_drafting=true` 都能正常工作。
- 更大的 draft depth 会带来更多 verification KV / slot 开销。

### 7.3 How in vLLM

在 `SpeculativeConfig` 里，`parallel_drafting` 是一个通用开关，但注释明确说只兼容 EAGLE 和 draft model 方法。对于 PARD，vLLM 仍走 `DraftModelProposer`，但 `SpecDecodeBaseProposer` 里的 parallel drafting path 会以 masked slots / extra input slots 的方式组织 draft 输入。

源码参考：

- [vLLM `parallel_drafting` 配置](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L147-L152)
- [vLLM parallel drafting slot 计算](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L1040-L1048)
- [vLLM `SpecDecodeBaseProposer` parallel path](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/llm_base_proposer.py#L693-L811)

### 7.4 适合什么场景

适合：

- 有 PARD 权重。
- classic draft model 的 draft 阶段耗时已经明显。
- 希望较大 `num_speculative_tokens` 仍保持低 draft latency。

不适合：

- 没有 PARD 适配权重。
- memory/KV 已经紧张，大 draft depth 会进一步增加压力。

参考论文：

- [PARD](https://arxiv.org/abs/2504.18583)
- [vLLM P-EAGLE 博客](https://vllm.ai/blog/2026-03-13-p-eagle)

## 8. 方法五：N-Gram / Prompt Lookup / NgramGPU

### 8.1 What

`N-Gram` 是最简单的 training-free proposer。它不加载 draft model，而是在当前 prompt / 已生成 token 中找与当前 suffix 匹配的 n-gram，然后把历史匹配位置后面的 token 拿来作为 draft。

TensorRT-LLM 的 N-Gram 文档把这个过程画得很直观：新请求进来时，先把序列扫描成 `(key n-gram -> value continuation)` 的候选池；每生成一个新 token，再滑动更新这些候选对。

![N-Gram initial sequence scan](assets/ngram_init_scan.png)

![N-Gram per-token update](assets/ngram_per_token_update.png)

vLLM 配置示例：

```python
LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)
```

参考：[vLLM N-Gram Speculation 文档](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/n_gram/)

### 8.2 Why

很多生成任务不是完全开放式创造，而是复制/改写上下文：

- 代码编辑时复用原文件片段。
- RAG answer 从文档中摘录。
- JSON、YAML、Markdown、SQL 有固定模板。
- tool call 参数重复 prompt 中的实体。

这类场景下，“当前 suffix 以前出现过，那它后面跟着的 token 可能还会出现”是一个非常便宜但有效的假设。

优点：

- 不需要额外模型。
- 几乎零训练成本。
- 显存开销小。
- 对重复文本很有效。

缺点：

- 开放域聊天收益不稳定。
- 只能利用当前 request 内的 token history，跨请求记忆弱。
- CPU 版实现可能有 CPU / 同步开销。

### 8.3 How in vLLM

`NgramProposer` 的核心函数 `_find_longest_matched_ngram_and_propose_tokens` 会：

1. 取当前 token 序列。
2. 在 `[prompt_lookup_min, prompt_lookup_max]` 窗口内找最长 suffix n-gram match。
3. 如果找到，就返回匹配位置后的最多 `k = num_speculative_tokens` 个 token。
4. target model 验证。

“是不是哈希？”这个直觉是对的，但要分概念和 vLLM 当前实现：

- 从算法抽象看，它确实可以被理解成把历史 token 片段组织成 `key = n-gram`、`value = continuation` 的查找表；TensorRT-LLM 的示意图也很像这种 key-value 候选池。
- vLLM CPU 版 `NgramProposer` 当前并没有真的为每个 request 建哈希表。它把 token 序列反过来，然后用类似 KMP 的 `LPS(longest prefix suffix)` 方法，寻找“当前 suffix”在历史中出现过的最长匹配。这样能避免对每个 n 都朴素地切片比较一遍。
- vLLM GPU 版 `NgramProposerGPU` 也不是哈希表，而是用 `token_ids.unfold(...)` 生成滑动窗口视图，再把每个窗口和当前 trailing suffix 做向量化 equality，选出有匹配的最长 n-gram。

一个小例子：

```text
history:  A B C D  A B C E  A B C
current suffix = A B C

历史里较早位置出现过 A B C，后面接的是 D / E。
N-Gram proposer 会取某个匹配位置后面的 token 作为 draft，例如 D。
target verifier 再判断 D 是否真是当前上下文下 target 会接受的 token。
```

所以它不是语义检索，也不理解文本内容，只是在 token id 序列上做精确匹配。哈希表是很自然的实现方式，但 vLLM 这里 CPU 更接近 KMP/string matching，GPU 更接近批量滑窗比较。

`ngram_gpu` 是 GPU 版 n-gram proposer，目的是降低 CPU 参与和 D2H 同步开销。它在源码中有独立 `NgramProposerGPU`，但 v0.21.0 用户文档没有把它作为主线页面展开。

源码参考：

- [vLLM `NgramProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/ngram_proposer.py#L12-L285)
- [vLLM `NgramProposerGPU`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/ngram_proposer_gpu.py#L216-L363)
- [vLLM n-gram 默认参数处理](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L585-L618)

### 8.4 适合什么场景

适合：

- prompt 中有大量可复制内容。
- 输出格式高度结构化。
- 代码补全/代码编辑。
- 长上下文摘要或问答，答案会复制原文。

不适合：

- 开放域闲聊。
- 高创造性写作。
- prompt 和 output 没有明显重复。

### 8.5 参数和调优

重点参数：

- `prompt_lookup_min`：最小匹配 n-gram 长度。
- `prompt_lookup_max`：最大匹配 n-gram 长度。
- `num_speculative_tokens`：匹配后最多复制多少 token。

经验：

- `prompt_lookup_max` 太小会误匹配，接受率低。
- `prompt_lookup_max` 太大则命中少。
- 结构化/代码场景可以提高 `num_speculative_tokens`，开放式文本应保守。

## 9. 方法六：Suffix Decoding

### 9.1 What

`Suffix Decoding` 是 vLLM 中的 model-free / retrieval-style proposer。它和 n-gram 一样用 token pattern 做 draft，但更强：

- 可以匹配 prompt 和 previous generations。
- 使用频次统计选择更可能的 continuation。
- 每轮动态决定 speculative length。
- 有 global suffix tree 和 per-request local suffix tree。

Snowflake 的图很好地展示了它和普通 n-gram 的差异：它不是只找一个固定窗口，而是把历史输出组织成 suffix tree，先从当前 generation 的末尾 token 出发找到匹配路径，再扩展、打分、送给 target verifier。

![SuffixDecoding suffix tree speculation](assets/suffix_tree_speculation.webp)

vLLM 配置示例：

```python
LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 32,
    },
)
```

参考：[vLLM Suffix Decoding 文档](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/suffix/)

### 9.2 Why

Suffix Decoding 解决的是普通 n-gram 的两个短板：

- 普通 n-gram 只看当前上下文，suffix cache 可以跨历史 generation 利用重复。
- 普通 n-gram 通常固定 draft 长度，suffix decoding 根据匹配深度和频次概率动态调整。

适合的 workload：

- 代码编辑和 patch 生成。
- agentic loop，例如 self-reflection、self-consistency。
- RL rollout。
- 多轮任务规划。
- 模板化 SQL / JSON / DSL 生成。

不适合：

- 无明显重复的开放域聊天。
- 每个请求都完全独立、没有历史模式。
- CPU 已经成为瓶颈的高并发场景。

### 9.3 How in vLLM

vLLM 的 `SuffixDecodingProposer` 是 wrapper，真正的 suffix cache 和 tree 结构来自 Arctic Inference。

执行链：

1. vLLM 初始化 `SuffixDecodingProposer`。
2. proposer 内部创建 Arctic Inference 的 `SuffixDecodingCache`。
3. 每轮 decode 后，vLLM 将当前 request 的 token 状态传给 proposer。
4. proposer 用最近 token 匹配 suffix tree。
5. 根据频次统计选 continuation。
6. 输出 draft token 给 vLLM verifier。

源码参考：

- [vLLM `SuffixDecodingProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/suffix_decoding.py#L9-L97)
- [vLLM suffix config validation](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L808-L842)
- [ArcticInference Suffix Decoding 文档](https://arcticinference.readthedocs.io/en/latest/suffix-decoding.html)

### 9.4 Suffix cache 数据结构

Suffix Decoding 的核心是两个 tree：

| 结构 | 作用 |
| --- | --- |
| local tree | 当前 request 的 prompt 和已生成 token |
| global tree | 历史 request 的 generation，受 `suffix_decoding_max_cached_requests` 控制 |

tree 节点存储 token transition 和频次统计。draft 时，系统从当前 suffix 出发，在 tree 中找匹配路径，再按频次/概率阈值向前生成 continuation。


关键参数：

- `suffix_decoding_max_tree_depth`：最大树深。
- `suffix_decoding_max_cached_requests`：global cache 请求数上限，设 `0` 可关闭 global cache。
- `suffix_decoding_max_spec_factor`：speculative length 与 prefix match length 的比例上限。
- `suffix_decoding_min_token_prob`：频次估计概率阈值。

### 9.5 和 N-Gram 的区别

| 维度 | N-Gram | Suffix Decoding |
| --- | --- | --- |
| 历史范围 | 当前 request token history | 当前 request + 历史 generations |
| 数据结构 | 直接匹配 token array | suffix tree / suffix cache |
| continuation 选择 | 匹配位置后的 token | 基于频次统计的 continuation |
| speculative length | 通常固定上限 | 动态 |
| 额外依赖 | 无 | Arctic Inference |
| 适合场景 | prompt 内重复 | 跨请求/跨轮次重复 |

参考：

- [SuffixDecoding 论文](https://arxiv.org/abs/2411.04975)
- [Snowflake SuffixDecoding at Production Scale](https://www.snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/)

## 10. 方法七：DFlash

### 10.1 What

`DFlash` 是 vLLM Speculators 文档中的新方法：用一个小 diffusion-LLM draft model，一次 forward 预测一整个 token block，而不是像 EAGLE3 / draft model 那样自回归地产生 draft token。

vLLM Speculators 文档对 DFlash 的定义是：draft model 使用非因果 attention mask，让每个 query 同时 attend 到 verifier hidden states 和 mask token embeddings，从而一次生成所有 draft token。

![DFlash architecture](assets/dflash_architecture.webp)

参考：[vLLM Speculators DFlash 文档](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)

### 10.2 Why

EAGLE3、classic draft model 的问题是：drafter 自身仍然自回归。DFlash 想解决的是 draft 阶段的串行瓶颈。

优点：

- block-parallel draft，理论上 draft latency 对 draft length 更友好。
- Speculators 文档声称同步请求下相对 Eagle3 可有 2-3x 更大 speedup。
- 适合探索更高 speculation depth。

缺点：

- 新方法，仍在 active development。
- 需要 DFlash speculator 权重。

### 10.3 How in vLLM

源码里 `dflash` 是 `DFlashModelTypes`，也被纳入 `EagleModelTypes`。`SpeculativeConfig.use_dflash()` 判断 `method=="dflash"`，`GPUModelRunner` 会创建 `DFlashProposer`，并设置 `use_aux_hidden_state_outputs=True`。

实现链路：

1. target model 产出 hidden states / context features。
2. DFlash drafter 将 hidden states 和 mask token embeddings 一起输入 draft layers。
3. 一次 forward 产出 block draft logits。
4. target verifier 一次验证 draft block。

源码参考：

- [vLLM `DFlashModelTypes`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L54-L58)
- [vLLM `use_dflash`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L1062-L1063)
- [vLLM `DFlashProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/dflash.py#L21-L102)

### 10.4 适合什么场景

适合：

- 有 DFlash speculator。
- 想降低 learned proposer 的 draft 串行开销。
- 服务更偏同步请求 / 低延迟。
- 愿意接受新方法的兼容性验证成本。

不适合：

- GPU 架构未经验证。
- 没有对应 speculator 权重。

参考论文：

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)

## 11. 方法对比与选型建议

### 11.1 低门槛优先

如果目标是先跑起来：

1. `ngram`：最简单，不需要额外依赖。
2. `suffix`：需要安装 Arctic Inference，但不需要 draft model。
3. `mtp`：如果模型原生支持 MTP，配置也很简单。

### 11.2 稳定收益优先

如果目标是通用加速：

1. `eagle3`：当前 vLLM Speculators 重点方向。
2. `mtp`：模型原生支持时优先。
3. `draft_model`：有合适小模型时可用。

### 11.3 降低 draft 串行开销

如果 draft 阶段成为瓶颈：

1. `PARD`：parallel draft model。
2. `DFlash`：block-parallel diffusion-style draft。

### 11.4 重复性 workload

如果 workload 是代码、RAG、模板、agent loop：

1. `ngram`：当前 request 内重复强。
2. `suffix`：跨历史 generation 重复强。

## 12. 参考链接

### vLLM 官方文档

- [vLLM v0.21.0 Speculative Decoding 总览](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/)
- [vLLM Draft Models](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/draft_model/)
- [vLLM EAGLE Draft Models](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/eagle/)
- [vLLM MTP](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/mtp/)
- [vLLM N-Gram Speculation](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/n_gram/)
- [vLLM Parallel Draft Models](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/parallel_draft_model/)
- [vLLM Suffix Decoding](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/suffix/)
- [vLLM Speculators Getting Started](https://docs.vllm.ai/projects/speculators/en/stable/user_guide/getting_started/)
- [vLLM Speculators Eagle3](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/eagle3/)
- [vLLM Speculators DFlash](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)

### vLLM 源码

- [vLLM `SpeculativeMethod` 枚举](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L34-L68)
- [vLLM `SpeculativeConfig` 字段](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L72-L185)
- [vLLM method inference](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/config/speculative.py#L520-L753)
- [vLLM drafter selection](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/worker/gpu_model_runner.py#L535-L610)
- [vLLM propose dispatch](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/worker/gpu_model_runner.py#L4737-L4866)
- [vLLM `SpecDecodeBaseProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/llm_base_proposer.py#L55-L638)
- [vLLM `DraftModelProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/draft_model.py#L17-L56)
- [vLLM `EagleProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/eagle.py#L10-L32)
- [vLLM `Gemma4Proposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/gemma4.py#L31-L69)
- [vLLM `NgramProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/ngram_proposer.py#L12-L285)
- [vLLM `NgramProposerGPU`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/ngram_proposer_gpu.py#L216-L363)
- [vLLM `SuffixDecodingProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/suffix_decoding.py#L9-L97)
- [vLLM `DFlashProposer`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/dflash.py#L21-L102)
- [vLLM `SpecDecodingStats`](https://github.com/vllm-project/vllm/blob/9640970de20b15ade9eb3859825637f64e81ed8c/vllm/v1/spec_decode/metrics.py#L18-L116)

### 论文与博客

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [EAGLE](https://arxiv.org/abs/2401.15077)
- [EAGLE-2](https://arxiv.org/abs/2406.16858)
- [EAGLE-3](https://arxiv.org/abs/2503.01840)
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [PARD](https://arxiv.org/abs/2504.18583)
- [SuffixDecoding](https://arxiv.org/abs/2411.04975)
- [DFlash](https://arxiv.org/abs/2602.06036)
- [NVIDIA: An Introduction to Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [NVIDIA TensorRT-LLM: N-Gram Speculative Decoding](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog7_NGram_performance_Analysis_And_Auto_Enablement.html)
- [AMD: Accelerating Generative LLMs Inference with PARD](https://www.amd.com/en/developer/resources/technical-articles/accelerating-generative-llms-interface-with-parallel-draft-model-pard.html)
- [vLLM blog: How Speculative Decoding Boosts vLLM Performance](https://vllm.ai/blog/spec-decode)
- [vLLM blog: P-EAGLE](https://vllm.ai/blog/2026-03-13-p-eagle)
- [vLLM blog: EAGLE 3.1](https://vllm.ai/blog/2026-05-26-eagle-3-1)
- [Snowflake: SuffixDecoding at Production Scale](https://www.snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/)

## 13. 实验结果

### 13.1 核心结论

下面按方法归纳本次实验结果。不同方法受 vLLM 支持模型族限制，表中保留对应有效模型组合；结论口径统一按本次实验里的有效数据点描述，不把模型覆盖结果拆成另一轮实验。

| 方法 | 本次实验结论 | 关键数据点 | 解读 |
| --- | --- | --- | --- |
| EAGLE3 | 当前最稳定的 Qwen3-32B model-based 方案。 | Qwen3-32B Spec-Bench：H100 `1.54x`，A2 `1.85x`。 | 在真实 benchmark 上收益最清楚，优先级高于普通 draft model / PARD。 |
| DFlash | CUDA 上很强，尤其是 Spec-Bench 和 datastores。 | Qwen3.5-9B H100：Spec-Bench `1.82x`，datastores-384 `1.58x`。 | 依赖匹配的 DFlash speculator；A2 当前 vLLM-Ascend 软件栈不可用。 |
| MTP | 原生支持 MTP 的模型上有稳定收益，但收益大小取决于 nextn head 质量。 | Qwen3.5-9B：H100 datastores-384 `1.19x`，A2 datastores-384 `1.47x`；A2 Qwen3.6-35B-A3B datastores-384 `1.89x`。 | 适合模型原生带 MTP 或有 assistant checkpoint 的场景，普通 checkpoint 不能直接开启；即使 server 能起，也要看 draft 是否真的被 target 接受。 |
| MTP -> suffix | 实验性拼接方案里最稳定的正向信号。 | H100 Qwen3.5-9B：random `1.63x`、Spec-Bench `1.28x`、prefixrep `1.62x`、datastores-384 `1.31x`；A2 Qwen3.5-9B datastores-384 `1.95x`；A2 Qwen3.6-35B-A3B datastores-384 `2.32x`。 | MTP 给高置信第一段，suffix 用临时上下文补 tail；收益来自 acceptance length 和 TPOT 改善，不是单看 acceptance rate。 |
| MTP -> ngram | synthetic 上强，真实/混合任务上更依赖命中质量。 | H100 Qwen3.5-9B：random `1.48x`、Spec-Bench `1.12x`、prefixrep `1.50x`、datastores-384 `1.16x`；A2 Qwen3.5-9B datastores-384 `1.69x`；A2 Qwen3.6-35B-A3B datastores-384 `2.03x`。 | 比 MTP->suffix 略不稳定，ngram tail 的查找质量和调度开销更容易抵消收益；当 MTP 本身无效时不应期待 hybrid 有效。 |
| Suffix Decoding | 重复性场景有收益，但不能只看 synthetic。 | Qwen3-32B Spec-Bench：H100 `0.98x`，A2 `1.25x`；A2 Qwen3.6-35B-A3B datastores-384：cold `0.80x`、warm `1.11x`。 | 高重复 synthetic 上 acceptance 很高，真实和混合 workload 上收益明显更保守；warm 明显好于 cold，说明 suffix tree 的跨请求历史确实会改变结果。 |
| N-Gram / NgramGPU | 低门槛，但不适合作为一刀切的通用加速主方案。 | Qwen3-32B Spec-Bench：H100 ngram `0.85x`、ngram_gpu `0.80x`；A2 Qwen3.6-35B-A3B datastores-384 ngram `0.64x`。 | 更适合 prompt/输出重复很强的任务；在 datastores 上如果命中质量不足，检索式 proposer 的调度开销会压过收益。 |
| Draft Model | 本组模型组合下端到端收益较差。 | H100 Qwen3-32B Spec-Bench：acceptance rate `39.92%`，但吞吐只有 `0.33x` baseline。 | 接受率不是端到端收益；draft forward、KV cache、调度和显存占用会抵消收益。 |
| PARD | 当前结果不适合作为有效性能结论。 | H100 Qwen3-32B Spec-Bench：`66/200` 成功；synthetic random / prefix repetition 分别约 `0.79x` / `0.72x`。 | PARD 的方向是降低 draft 串行开销，但本组权重/后端组合存在稳定性和 workspace 约束问题。 |

### 13.2 Benchmark 说明

本次实验使用四类 benchmark。读结果时要把它们的定位分开：`Spec-Bench` 更接近真实任务主口径，`datastores-384` 是本次新增的真实混合数据集，两个 synthetic benchmark 主要用于诊断重复性、proposal 行为和后端开销。

| Benchmark | `vllm bench serve` 入口 | 本次用途 | 本次关键参数 |
| --- | --- | --- | --- |
| Spec-Bench | `--dataset-name spec_bench` | 真实任务主口径，用来判断方法是否有通用收益。Spec-Bench 官方定位是面向 speculative decoding 的统一评测，覆盖 multi-turn chat、summarization、RAG、translation、question answering、math reasoning 等子任务。 | 文件路径 `data/spec_bench/question.jsonl`；`spec_bench_output_len=512`；Qwen3-32B 主对比中 H100 取 200 prompts，A2 取 100 prompts。 |
| Synthetic Random | `--dataset-name random` | 固定输入/输出长度的可控吞吐压力测试，用来观察调度、KV、acceptance 和 proposer 开销；不代表自然语言质量或真实任务分布。 | `random_input_len=1024`，`random_output_len=256`，`random_range_ratio=0.1`，`ignore_eos=true`，`temperature=0`。 |
| Synthetic Prefix Repetition | `--dataset-name prefix_repetition` | 人工制造重复 prefix 的压力测试，主要观察 n-gram / suffix 这类检索式 proposer 在高重复 workload 下的上限表现。 | `prefix_len=512`，`suffix_len=512`，`num_prefixes=8`，`output_len=256`，`ignore_eos=true`，`temperature=0`。 |
| Datastores-384 | `--dataset-name custom` | 新增真实混合 benchmark，覆盖中文法律、摘要、数学、代码、MT-Bench、GPQA、HumanEval 等 16 个来源。 | 文件路径 `data/datastores/datastores_mixed_custom.jsonl`；共 384 条；`custom_output_len=256`；`ignore_eos=true`；`temperature=0`；`disable_shuffle=true`。 |

因此，本报告的通用结论优先看 `Spec-Bench` 和 `datastores-384`。`Synthetic Random` 和 `Synthetic Prefix Repetition` 的结果可以解释“为什么某个方法在重复/固定长度 workload 下会变好或变差”，但不能单独外推为真实服务收益。

参考：

- [Spec-Bench 项目页](https://sites.google.com/view/spec-bench)
- [Spec-Bench GitHub](https://github.com/hemingkx/Spec-Bench)
- [本仓库 vLLM bench dataset reference](docs/vllm_bench_dataset_reference.md)

### 13.3 通用实验命令模板

脚本实际重复执行的是两段命令：先用 Docker 启动 `vllm serve`，再在同一个容器里用 `vllm bench serve` 打流量并保存 JSON。下面给出通用命令模板，省略镜像名、device 挂载和结果目录挂载等 Docker 细节；不同方法只需要替换 `<speculative_config_json>`。

```bash
vllm serve <target_model_or_path> \
  --host 0.0.0.0 \
  --port <port> \
  --served-model-name <served_model_name> \
  --trust-remote-code \
  --distributed-executor-backend <mp_or_ray> \
  --tensor-parallel-size <tp_size> \
  --max-model-len <max_model_len> \
  --max-num-batched-tokens <max_num_batched_tokens> \
  --gpu-memory-utilization <gpu_memory_utilization> \
  --disable-uvicorn-access-log \
  <optional_extra_server_flags> \
  --speculative-config '<speculative_config_json>'

vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port <port> \
  --model <served_model_name> \
  --save-result \
  --save-detailed \
  --result-dir <result_dir> \
  --result-filename <result_filename>.json \
  --dataset-name <dataset_name> \
  <dataset_specific_args> \
  --num-prompts <num_prompts> \
  --request-rate <request_rate> \
  --max-concurrency <max_concurrency> \
  --temperature <temperature> \
  --tokenizer <target_model_or_path>
```

`<speculative_config_json>` 按方法替换即可：

| 方法 | `<speculative_config_json>` 形态 |
| --- | --- |
| Draft Model | `{"method":"draft_model","model":"<draft_model_or_path>","num_speculative_tokens":<k>}` |
| EAGLE / EAGLE3 | `{"method":"eagle3","model":"<eagle3_speculator_or_path>","num_speculative_tokens":<k>}` |
| MTP | `{"method":"mtp","num_speculative_tokens":<k>}` 或 `{"method":"mtp","model":"<assistant_checkpoint_or_path>","num_speculative_tokens":<k>}` |
| PARD | `{"method":"draft_model","model":"<pard_model_or_path>","parallel_drafting":true,"num_speculative_tokens":<k>}` |
| N-Gram | `{"method":"ngram","num_speculative_tokens":<k>,"prompt_lookup_min":<min_n>,"prompt_lookup_max":<max_n>}` |
| Suffix Decoding | `{"method":"suffix","num_speculative_tokens":<k>}` |
| DFlash | `{"method":"dflash","model":"<dflash_speculator_or_path>","num_speculative_tokens":<k>}` |
| Hybrid MTP -> N-Gram | `{"method":"mtp_ngram_concat","num_speculative_tokens":4,"hybrid_mtp_tokens":1,"prompt_lookup_min":2,"prompt_lookup_max":5}` |
| Hybrid MTP -> Suffix | `{"method":"mtp_suffix_concat","num_speculative_tokens":4,"hybrid_mtp_tokens":1}` |

Spec-Bench 数据集通常把 `<dataset_specific_args>` 写成 `--dataset-name spec_bench --dataset-path <spec_bench_question_jsonl> --spec-bench-output-len <output_len>`；Synthetic random 则替换为 `--dataset-name random --random-input-len <input_len> --random-output-len <output_len> --random-range-ratio <ratio> --ignore-eos`；datastores-384 使用 `--dataset-name custom --dataset-path /workspace/suffix-bench/data/datastores/datastores_mixed_custom.jsonl --num-prompts 384 --custom-output-len 256 --disable-shuffle --no-oversample --ignore-eos`。

### 13.4 Qwen3-32B Spec-Bench

Qwen3-32B 使用 TP=8。H100 使用 200 prompts，A2 使用 100 prompts。

| Platform | Method | Completed / Failed | Output tok/s | vs Baseline | TPOT ms | Acceptance Rate | Acceptance Len |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | baseline | 200 / 0 | 1214.9 | 1.00x | 6.38 | - | - |
| H100 | ngram | 200 / 0 | 1029.1 | 0.85x | 7.41 | 25.53% | 1.77 |
| H100 | ngram_gpu | 200 / 0 | 972.8 | 0.80x | 7.90 | 6.42% | 1.19 |
| H100 | suffix | 200 / 0 | 1184.9 | 0.98x | 6.54 | 25.45% | 1.45 |
| H100 | draft | 200 / 0 | 395.2 | 0.33x | 19.26 | 39.92% | 3.00 |
| H100 | PARD | 66 / 134 | invalid | invalid | invalid | - | - |
| H100 | EAGLE3 | 200 / 0 | 1870.8 | 1.54x | 3.83 | 41.58% | 2.25 |
| A2 | baseline | 100 / 0 | 62.8 | 1.00x | 63.03 | - | - |
| A2 | ngram | 100 / 0 | 57.1 | 0.91x | 68.44 | 25.12% | 1.75 |
| A2 | suffix | 100 / 0 | 78.5 | 1.25x | 50.48 | 25.94% | 1.45 |
| A2 | EAGLE3 | 100 / 0 | 116.2 | 1.85x | 33.06 | 42.21% | 2.27 |

H100 Spec-Bench PARD 不是有效性能结果：200 条请求中只有 66 条成功、134 条失败，并且没有可用 speculative acceptance 字段。server 日志中的直接原因是 `Workspace validation failed: token_num (256) * hidden_dim (1024) exceeds workspace max_token_num (51) * hidden_dim (5120)`。A2 的 draft/PARD Spec-Bench 没有纳入有效对比，原因同 random 上观察到的 vLLM Ascend draft proposer hidden-size mismatch。

### 13.5 Synthetic Random

synthetic random 主要用于观察重复性、缓存命中与 proposal 行为，不应替代 Spec-Bench 作为真实任务结论。

| Platform | Method | Completed / Failed | Output tok/s | vs Baseline | TPOT ms | Acceptance Rate | Acceptance Len |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | baseline | 64 / 0 | 1038.4 | 1.00x | 6.89 | - | - |
| H100 | ngram | 64 / 0 | 985.8 | 0.95x | 6.03 | 77.87% | 3.31 |
| H100 | ngram_gpu | 64 / 0 | 781.7 | 0.75x | 7.93 | 25.09% | 1.75 |
| H100 | suffix | 64 / 0 | 1279.7 | 1.23x | 5.03 | 69.99% | 2.49 |
| H100 | draft | 64 / 0 | 360.0 | 0.35x | 18.15 | 50.63% | 3.53 |
| H100 | PARD | 64 / 0 | 815.4 | 0.79x | 7.45 | 14.65% | 2.76 |
| H100 | EAGLE3 | 64 / 0 | 842.3 | 0.81x | 6.76 | 17.20% | 1.52 |
| A2 | baseline | 32 / 0 | 78.5 | 1.00x | 49.06 | - | - |
| A2 | ngram | 32 / 0 | 130.7 | 1.67x | 24.86 | 86.88% | 3.58 |
| A2 | suffix | 32 / 0 | 136.8 | 1.74x | 23.61 | 81.28% | 2.87 |
| A2 | draft | 0 / 32 | invalid | invalid | invalid | - | - |
| A2 | PARD | 1 / 31 | invalid | invalid | invalid | - | - |
| A2 | EAGLE3 | 32 / 0 | 65.5 | 0.84x | 53.76 | 14.88% | 1.45 |

A2 draft/PARD random 不是有效结果：draft 为 0/32 成功，PARD 为 1/32 成功。日志显示 vLLM Ascend 的 draft proposer 路径在 Qwen3-32B target hidden size 5120 与 0.6B draft hidden size 1024 上发生 shape mismatch。

### 13.6 Synthetic Prefix Repetition

| Platform | Method | Completed / Failed | Output tok/s | vs Baseline | TPOT ms | Acceptance Rate | Acceptance Len |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | baseline | 64 / 0 | 1091.4 | 1.00x | 6.49 | - | - |
| H100 | ngram | 64 / 0 | 916.7 | 0.84x | 6.97 | 68.92% | 3.04 |
| H100 | ngram_gpu | 64 / 0 | 825.3 | 0.76x | 8.06 | 15.36% | 1.46 |
| H100 | suffix | 64 / 0 | 1257.6 | 1.15x | 5.39 | 56.70% | 2.12 |
| H100 | draft | 64 / 0 | 309.4 | 0.28x | 20.87 | 41.68% | 3.08 |
| H100 | PARD | 64 / 0 | 787.2 | 0.72x | 7.91 | 12.32% | 2.48 |
| H100 | EAGLE3 | 64 / 0 | 840.8 | 0.77x | 6.72 | 16.18% | 1.49 |
| A2 | baseline | 32 / 0 | 80.7 | 1.00x | 48.32 | - | - |
| A2 | ngram | 32 / 0 | 69.0 | 0.86x | 51.69 | 44.44% | 2.33 |
| A2 | suffix | 32 / 0 | 101.0 | 1.25x | 35.98 | 47.91% | 1.86 |
| A2 | EAGLE3 | 32 / 0 | 67.3 | 0.83x | 51.93 | 19.79% | 1.59 |

### 13.7 Qwen3.5-9B 方法覆盖：MTP / DFlash / Hybrid

这一组用于覆盖 vLLM 中需要特定模型族或特定 speculator 权重的方法，同时验证实验性 `MTP -> ngram/suffix` 拼接 proposer。它不是 Qwen3-32B 的同模型对比，而是按 vLLM 当前支持的模型组合测试。

这里选 `Qwen3.5-9B` 是因为能够同时找到：

1. target 模型 `/home/external/huangmy/models/Qwen3.5-9B` 和 `/home/huangmy/models/Qwen3.5-9B`；
2. 原生 MTP 路径可识别的 Qwen3.5 MTP 结构；
3. 与该 target 配套的 DFlash speculator：`Qwen3.5-9B-DFlash`。

MTP 要求目标模型或 assistant checkpoint 原生支持 MTP，普通 Qwen3-32B checkpoint 不能直接开启 `method="mtp"`。DFlash 也不是 training-free 方法，而是需要和 target 模型匹配的 diffusion-style draft model / speculator。A2 侧权重齐全，但当前 vLLM-Ascend 0.18.0 镜像的 `SpeculativeConfig` 方法枚举不包含 `dflash`，因此 A2 的 DFlash case 均为 server init failed。

`MTP -> ngram/suffix` 是本次本地实验性实现：先用 Qwen3.5-9B 原生 MTP 从当前 committed prefix 生成 `hybrid_mtp_tokens=1` 个 draft token，再把 `committed prefix + current sampled tokens + MTP draft tokens` 当作临时上下文，继续用 ngram 或 suffix 追加最多 3 个 draft token，最后把拼接后的 draft list 交给 vLLM 原来的 speculative verifier。实现位置是 `docker/patch_vllm_hybrid_concat.py`，新增 method 为 `mtp_ngram_concat` 和 `mtp_suffix_concat`。


#### 13.7.1 H100 Qwen3.5-9B TP1

| Dataset | Method | Completed / Failed | Output tok/s | vs Baseline | TPOT ms | Acceptance Rate | Acceptance Len |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | baseline | 64 / 0 | 767.1 | 1.00x | 8.41 | - | - |
| random | MTP | 64 / 0 | 815.5 | 1.06x | 7.27 | 97.33% | 1.97 |
| random | ngram | 64 / 0 | 691.0 | 0.90x | 9.68 | 96.91% | 4.77 |
| random | suffix | 64 / 0 | 820.8 | 1.07x | 7.44 | 92.26% | 4.14 |
| random | MTP -> ngram | 64 / 0 | 1134.1 | 1.48x | 4.56 | 95.36% | 4.37 |
| random | MTP -> suffix | 64 / 0 | 1252.0 | 1.63x | 4.43 | 94.50% | 4.49 |
| random | DFlash | 64 / 0 | 831.9 | 1.08x | 5.86 | 17.13% | 3.57 |
| Spec-Bench | baseline | 100 / 0 | 960.0 | 1.00x | 7.34 | - | - |
| Spec-Bench | MTP | 100 / 0 | 1133.4 | 1.18x | 6.05 | 88.84% | 1.89 |
| Spec-Bench | ngram | 100 / 0 | 506.6 | 0.53x | 14.81 | 39.36% | 2.57 |
| Spec-Bench | suffix | 100 / 0 | 654.9 | 0.68x | 11.55 | 38.56% | 1.78 |
| Spec-Bench | MTP -> ngram | 100 / 0 | 1076.3 | 1.12x | 6.52 | 64.59% | 2.31 |
| Spec-Bench | MTP -> suffix | 100 / 0 | 1227.5 | 1.28x | 5.78 | 56.26% | 2.54 |
| Spec-Bench | DFlash | 100 / 0 | 1749.5 | 1.82x | 3.44 | 27.24% | 5.09 |
| prefixrep | baseline | 64 / 0 | 798.9 | 1.00x | 7.65 | - | - |
| prefixrep | MTP | 64 / 0 | 923.3 | 1.16x | 6.46 | 95.67% | 1.96 |
| prefixrep | ngram | 64 / 0 | 631.9 | 0.79x | 11.05 | 89.96% | 4.50 |
| prefixrep | suffix | 64 / 0 | 956.7 | 1.20x | 7.14 | 88.30% | 3.81 |
| prefixrep | MTP -> ngram | 64 / 0 | 1196.6 | 1.50x | 4.40 | 93.21% | 3.82 |
| prefixrep | MTP -> suffix | 64 / 0 | 1294.8 | 1.62x | 4.13 | 91.52% | 4.17 |
| datastores-384 | baseline | 384 / 0 | 991.7 | 1.00x | 7.52 | - | - |
| datastores-384 | MTP | 384 / 0 | 1176.3 | 1.19x | 6.16 | 90.47% | 1.90 |
| datastores-384 | ngram | 384 / 0 | 470.6 | 0.47x | 16.51 | 32.83% | 2.31 |
| datastores-384 | suffix | 384 / 0 | 643.1 | 0.65x | 12.03 | 37.46% | 1.76 |
| datastores-384 | MTP -> ngram | 384 / 0 | 1147.5 | 1.16x | 6.43 | 68.65% | 2.23 |
| datastores-384 | MTP -> suffix | 384 / 0 | 1299.1 | 1.31x | 5.66 | 57.78% | 2.61 |
| datastores-384 | DFlash | 384 / 0 | 1571.5 | 1.58x | 4.32 | 19.68% | 3.95 |

H100 上 `MTP -> suffix` 是 hybrid 里更稳定的一项，四个数据集都超过 baseline；`MTP -> ngram` 在 random / prefixrep 上很强，在 Spec-Bench 和 datastores 上仍超过 baseline，但相对 MTP-only 不总是更好。DFlash 在 Spec-Bench 和 datastores-384 上收益最大，说明 learned/block proposer 的上限仍然更高。

#### 13.7.2 A2 Qwen3.5-9B TP1

A2 的 Qwen3.5-9B 表同样统一使用 TP=1。random / Spec-Bench 来自早先 method coverage 补测，覆盖 baseline、MTP 和 DFlash 可用性；datastores-384 是按最新要求在同一张 NPU4 上串行跑完的正式 384 条结果，覆盖 baseline、MTP、ngram、suffix、`MTP -> ngram` 和 `MTP -> suffix` 六个有效方法。A2 当前 vLLM-Ascend 软件栈不能有效运行 DFlash，因此 DFlash 不列入 datastores-384 的有效对比。

| Dataset | Method | Completed / Failed | Output tok/s | vs Baseline | TPOT ms | Acceptance Rate | Acceptance Len |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | baseline | 64 / 0 | 37.7 | 1.00x | 91.43 | - | - |
| random | MTP | 64 / 0 | 41.0 | 1.09x | 75.52 | 97.36% | 1.97 |
| random | DFlash | server init failed | invalid | invalid | invalid | - | - |
| Spec-Bench | baseline | 100 / 0 | 42.4 | 1.00x | 88.17 | - | - |
| Spec-Bench | MTP | 100 / 0 | 66.8 | 1.58x | 53.62 | 89.07% | 1.89 |
| Spec-Bench | DFlash | server init failed | invalid | invalid | invalid | - | - |
| datastores-384 | baseline | 384 / 0 | 46.5 | 1.00x | 82.60 | - | - |
| datastores-384 | MTP | 384 / 0 | 68.4 | 1.47x | 54.12 | 90.64% | 1.91 |
| datastores-384 | ngram | 384 / 0 | 32.1 | 0.69x | 121.09 | 32.86% | 2.31 |
| datastores-384 | suffix | 384 / 0 | 45.4 | 0.98x | 84.78 | 38.28% | 1.78 |
| datastores-384 | MTP -> ngram | 384 / 0 | 78.7 | 1.69x | 46.77 | 67.85% | 2.22 |
| datastores-384 | MTP -> suffix | 384 / 0 | 90.4 | 1.95x | 40.22 | 57.68% | 2.59 |

A2 datastores-384 上的最强信号是 `MTP -> suffix`：`90.4 tok/s`，达到 baseline 的 `1.95x`，也比 MTP-only 高约 `32%`。`MTP -> ngram` 也有明确收益，为 baseline 的 `1.69x`。这里的关键不是 acceptance rate 更高：`MTP -> suffix` 的 acceptance rate 低于 MTP-only，但 acceptance length 从 `1.91` 提高到 `2.59`，TPOT 从 `54.12ms` 降到 `40.22ms`，最终吞吐最高。

### 13.8 A2 Qwen3.6-35B-A3B datastore384

这一组是在 A2 vLLM-Ascend suffix 镜像中，用 `Eco-Tech/Qwen3.6-35B-A3B-w8a8` 补测更大的 MTP-capable 模型。统一使用 `datastores-384`，TP=4，NPU `0,1,2,3`，NPU 7 不参与；`custom_output_len=256`，`temperature=0`，`ignore_eos=true`，`max_concurrency=16`。为了和本轮需求对齐，`MTP` / `ngram` / `suffix` 的 draft token 数为 `2`，hybrid 额外测试 `1+1` 和 `2+2`。

这张表只保留 Qwen3.6-35B-A3B-w8a8 的完整方法矩阵；其他模型不纳入本次收束后的报告口径。

| Model | Method | Completed / Failed | Output tok/s | vs Baseline | TPOT ms | Acceptance Rate | Acceptance Len |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-w8a8 | baseline | 384 / 0 | 84.3 | 1.00x | 186.78 | - | - |
| Qwen3.6-35B-A3B-w8a8 | MTP 2 | 384 / 0 | 159.6 | 1.89x | 92.59 | 82.88% | 2.66 |
| Qwen3.6-35B-A3B-w8a8 | ngram 2 | 384 / 0 | 54.3 | 0.64x | 285.44 | 50.84% | 2.02 |
| Qwen3.6-35B-A3B-w8a8 | suffix 2 cold | 384 / 0 | 67.7 | 0.80x | 230.13 | 38.78% | 1.59 |
| Qwen3.6-35B-A3B-w8a8 | suffix 2 warm | 384 / 0 | 93.8 | 1.11x | 163.95 | 57.11% | 2.01 |
| Qwen3.6-35B-A3B-w8a8 | MTP -> ngram 1+1 | 384 / 0 | 133.1 | 1.58x | 115.57 | 82.57% | 2.07 |
| Qwen3.6-35B-A3B-w8a8 | MTP -> suffix 1+1 cold | 384 / 0 | 144.5 | 1.71x | 106.55 | 66.69% | 2.26 |
| Qwen3.6-35B-A3B-w8a8 | MTP -> suffix 1+1 warm | 384 / 0 | 153.9 | 1.82x | 100.25 | 73.39% | 2.45 |
| Qwen3.6-35B-A3B-w8a8 | MTP -> ngram 2+2 | 384 / 0 | 171.1 | 2.03x | 88.37 | 72.39% | 2.90 |
| Qwen3.6-35B-A3B-w8a8 | MTP -> suffix 2+2 cold | 384 / 0 | 184.1 | 2.18x | 83.02 | 62.02% | 3.14 |
| Qwen3.6-35B-A3B-w8a8 | MTP -> suffix 2+2 warm | 384 / 0 | 195.5 | 2.32x | 77.99 | 68.40% | 3.57 |

这组 Qwen3.6 结果对前面 A2 表格有三点补充：

- `MTP -> suffix 2+2 warm` 达到 `2.32x` baseline，是本轮最强 hybrid 信号；`MTP -> ngram 2+2` 也达到 `2.03x`，说明在该模型和数据集上，更长的 `2+2` 拼接比 `1+1` 更有价值。
- 单独的 `MTP 2` 已经有 `1.89x`，但 hybrid 仍能把 acceptance length 从 `2.66` 提到 `3.57`，TPOT 从 `92.59 ms` 降到 `77.99 ms`。这说明拼接式 proposer 的收益不是只来自多给 draft token，而是来自 MTP 前段和 suffix tail 共同提高每次 target forward 接受的 token 数。
- 单独的检索式 proposer 在这组数据上并不稳：`ngram 2` 只有 `0.64x`，`suffix 2 cold` 只有 `0.80x`，`suffix 2 warm` 才略高于 baseline 到 `1.11x`。所以这组实验支持的结论是：对 MTP-capable 的 Qwen3.6，`MTP -> suffix/ngram` 比单独使用检索式 proposer 更可靠；其中 `MTP -> suffix 2+2 warm` 是当前最值得继续扩展的配置。

### 13.9 跨数据集解读

- `Spec-Bench` 和 `datastores-384` 比 synthetic random / prefixrep 更适合作为真实收益判断口径。synthetic 上的高 acceptance 主要说明 proposal 更容易命中重复模式，不能直接外推到真实服务。
- `MTP -> suffix` 是本次 hybrid concat 里最稳定的正向信号：H100 Qwen3.5-9B 在四个数据集上都超过 baseline，A2 Qwen3.5-9B datastores-384 达到 `1.95x` baseline，A2 Qwen3.6-35B-A3B datastores-384 进一步达到 `2.32x`。
- `MTP -> ngram` 也有收益，但更依赖数据分布。H100 random / prefixrep 上很强，Spec-Bench 和 datastores-384 上仍超过 baseline，但相对 MTP-only 不总是更好；A2 Qwen3.5-9B datastores-384 为 `1.69x`，A2 Qwen3.6-35B-A3B datastores-384 为 `2.03x`。
- datastores-384 的正式 384 条结果和早先 64 条 smoke test 有明显差异，因此报告以 384 条为准。H100 早先 64 条里 `MTP -> suffix` 只有 `0.90x` baseline，但 384 条正式口径为 `1.31x`；A2 早先 same-NPU 64 条为 `1.52x`，384 条正式口径为 `1.95x`。
- DFlash 在 H100 CUDA 上很强，Spec-Bench 为 `1.82x` baseline，datastores-384 为 `1.58x` baseline；A2 当前软件栈不支持有效运行 `method=dflash`，因此不能把 H100 DFlash 结论平移到 A2。

### 13.10 问题说明与无效结果

- A2 DFlash 失败原因不是权重缺失，而是当前 vLLM-Ascend 0.18.0 的 `SpeculativeConfig` 方法枚举不包含 `dflash`，server init 阶段即失败。
- H100 Spec-Bench PARD 不是有效性能点：200 条请求中只有 66 条成功，日志显示 `Workspace validation failed: token_num (256) * hidden_dim (1024) exceeds workspace max_token_num (51) * hidden_dim (5120)`。
- A2 Qwen3-32B draft/PARD 也不是有效性能点：random 上 draft 为 0/32 成功，PARD 为 1/32 成功，原因是 vLLM Ascend draft proposer 路径在 target hidden size 5120 与 draft hidden size 1024 之间触发 shape mismatch。
