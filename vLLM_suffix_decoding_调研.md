# 投机解码方案调研：vLLM Suffix Decoding / Suffix Cache 与主流方案

初版调研时间：2026-04-23  
补充更新：2026-05-20，增加 UCM / KV cache 存储视角，并扩展为投机解码方案全景调研。  
调研范围：基于 vLLM 官方文档、vLLM 主线源码、vLLM Speculators 文档、ArcticInference 文档与源码、主流 speculative decoding 论文/博客，以及 UCM 官方仓库与文档。

## 一句话结论

投机解码的共同核心是 **draft-then-verify**：先用便宜方法猜多个 token，再让 target model 一次性验证，从而把“每 token 一次大模型 forward”的串行瓶颈变成“每次 forward 接受多个 token”。不同方案的差异主要在 **draft token 从哪里来**：小 draft model、EAGLE/MTP/Medusa 这类 learned speculator、n-gram/suffix/REST 这类检索式 proposer，或 DFlash 这类新兴 block-parallel drafter。

在这些方案里，`Suffix Decoding` 是 vLLM 中一种 **model-free / retrieval-style speculative decoding** 实现。它不依赖额外 draft model，而是把历史 prompt / 历史输出压进一个 **suffix tree cache** 里，用最近生成的 token 序列做模式匹配，猜测后续 token，再交给 target model 一次性验证。

## TL;DR

- 投机解码不是单一算法，而是一族“proposer + verifier + acceptance/rejection”的系统设计。
- vLLM 官方当前把常见方法分为 EAGLE、MTP、Draft Model、PARD、MLP speculator、N-gram、Suffix Decoding；源码中还可以看到 `medusa`、`ngram_gpu`、`dflash`、`custom_class` 等方法入口。
- 高收益通常来自 learned proposer，例如 EAGLE/EAGLE-3、MTP、draft model、PARD、DFlash；低门槛通常来自 training-free proposer，例如 n-gram、suffix、REST。
- 性能好坏不能只看 acceptance rate，还要看 draft 成本、target verification 成本、batch/QPS、KV cache 压力、采样策略和 workload 重复性。
- 从 UCM 背景看，投机解码本身主要优化 decode 阶段；UCM 的 KV cache 存储主要优化 prefill 复用、HBM 压力和 KV 跨实例/跨层级流动。两者可互补，但不是同一层能力。

## 0. 给 UCM 组实习同学的阅读地图

从你在华为 UCM 组实习的背景看，`suffix_cache` 最值得注意的点不是“它也是 cache”，而是它代表了另一类和 KV cache 存储相邻的优化方向：**把历史序列中的可复用信息抽出来，用 retrieval / cache 的方式减少后续计算**。

不过这里要先把边界划清楚：

- `vLLM suffix_cache` 缓存的是 **token 序列模式**，用于猜测未来 token，属于 speculative decoding 的 proposer。
- `vLLM prefix cache` 缓存的是 **已经算好的 KV block**，用于跳过重复 prompt 的 prefill。
- `UCM` 关注的是 **KV cache 的外部化、持久化、分层传输、稀疏检索和存储后端管理**，它解决的是“KV 张量存在哪里、怎么查、怎么搬、什么时候搬”的问题。

因此，读这份文档可以按三层来看：

1. 先理解 `suffix_cache`：它为什么不是 KV cache，却能通过 token 级模式匹配减少 decoding step。
2. 再理解 vLLM 的 KV cache 实现：`KVCacheManager`、`BlockPool`、prefix hash、scheduler 和 `KVConnectorBase_V1` 怎么串起来。
3. 最后映射到 UCM：UCM 如何通过 `UCMConnector` 接入 vLLM 的 KVConnector 机制，把外部 KV 命中伪装成“已经计算过的 prefix”，并进一步扩展到 NFS / POSIX / 3FS / 压缩 / 稀疏注意力 / PD 分离等存储能力。

一句话说，`suffix_cache` 更像“token continuation 的检索器”，UCM 更像“KV tensor 的存储系统”。二者不是替代关系，而是可以在长上下文、多轮对话、agent workflow 中互补：前者减少 target model 验证轮数，后者减少重复 prefill 和 HBM 压力。

## 1. 投机解码总览：共同范式

### 1.1 标准流程

绝大多数投机解码都可以抽象成下面这个循环：

1. **Draft**：用一个更便宜的 proposer 生成 `k` 个候选 token，可能是一条链，也可能是一棵 candidate tree。
2. **Verify**：把这些候选 token 拼到 target model 输入后面，让 target model 一次 forward 计算多个位置的 logits。
3. **Accept / Reject**：从左到右验证候选 token，接受最长正确前缀；如果遇到错误，就用 target model 的分布采样或选择正确 token。
4. **Commit**：把接受的 token 写回请求状态和 KV cache，继续下一轮。

这个范式来自经典 speculative decoding / speculative sampling 工作。Leviathan 等人的论文强调，在正确的 rejection sampling 下，投机解码可以保持 target model 原本的输出分布；Chen 等人的 speculative sampling 也给出了相近的并行验证思路。

参考：

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)

### 1.2 为什么能快

LLM decode 阶段通常每生成一个 token 都要做一次 target model forward。对于中低 QPS、batch 不大、memory-bound 的场景，单 token decode 往往不能把 GPU 算力吃满，还要反复从 HBM 读大模型权重。投机解码的收益来自两个事实：

- target model 验证多个候选 token 时，可以把多个 query token 合到一次 forward 里，摊薄权重读取和调度开销。
- proposer 足够便宜时，即使有一部分 draft 被拒绝，只要平均每轮接受 token 数大于 1，就可能降低 inter-token latency。

这也是为什么 vLLM 官方文档把 speculative decoding 的典型适用场景写成 **中低 QPS、memory-bound workload**。当 QPS 很高、batch 很大、target model 已经接近 compute-bound 时，额外 draft 成本可能抵消收益。

参考：

- [vLLM 官方文档，Speculative Decoding](https://docs.vllm.ai/usage/speculative_decoding/)
- [vLLM 博客，How Speculative Decoding Boosts vLLM Performance](https://vllm.ai/blog/spec-decode)

### 1.3 评价指标

调研投机解码时，不建议只看“端到端速度提升”。更可复现的拆解指标是：

| 指标 | 含义 | 为什么重要 |
| --- | --- | --- |
| Mean acceptance length | 平均每轮实际推进多少 token，通常包括 verifier 产生的 bonus token | 直接决定 decode step 能减少多少 |
| Draft acceptance rate | draft token 中被接受的比例 | 衡量 proposer 质量，但不等于端到端收益 |
| Draft throughput | proposer 每秒能产出多少 draft token | proposer 太慢会吃掉收益 |
| Accepted throughput | 每秒接受多少 draft token | 更贴近真实收益 |
| Per-position acceptance rate | 第 1/2/3/... 个 draft token 分别被接受的概率 | 用于调 `num_speculative_tokens` 和 tree 形状 |
| TTFT / ITL / TPOT | 首 token 延迟、token 间延迟、每输出 token 时间 | 服务侧最常看的用户体验指标 |
| KV/cache overhead | draft/verifier 多消耗的 KV cache、显存、CPU/GPU 同步 | 决定能不能和长上下文、prefix cache、UCM 共存 |

vLLM 源码里 `SpecDecodingStats` 和 `SpecDecodingLogging` 正好体现了这组核心指标：draft 数、draft token 数、accepted token 数、per-position acceptance、mean acceptance length、accepted/drafted throughput。

参考：

- [vLLM spec decode metrics 源码](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/metrics.py#L18-L116)

## 2. 方案分类与选型表

### 2.1 按 proposer 来源分类

| 类别 | 代表方案 | Draft token 从哪里来 | 训练/额外模型 | 典型优势 | 典型风险 |
| --- | --- | --- | --- | --- | --- |
| 小模型 draft | Classic draft model、PARD | 单独的小语言模型 | 需要 draft 模型 | 通用性强，理论清晰 | 额外显存/加载/调参，draft-target 对齐难 |
| 特征级 learned proposer | EAGLE、EAGLE-2、EAGLE-3 | target hidden states + 轻量 draft head/model | 需要训练 speculator | 当前很强的通用方案 | 训练和部署复杂，hidden state 接口依赖框架 |
| 原生多 token 头 | MTP | target 模型自带 MTP module/head | 训练时内置 | 无需另配 draft model | 只适用于支持 MTP 的模型族 |
| 多头预测 | Medusa、MLP speculator | target 末层 hidden states 上接多个预测头 | 需要训练 head | 推理结构简单，draft 成本低 | 对模型格式和 checkpoint 支持有要求 |
| 检索式 proposer | N-gram、Suffix Decoding、REST | prompt/历史输出/外部 datastore 中的 token 片段 | 通常不需要训练 | 低门槛，适合重复文本/代码/agent | 泛化弱，开放域收益不稳定 |
| 自投机/早退 | Draft & Verify、LayerSkip | target 模型早层/跳层输出 | 可零训练或需继续训练 | 不需要单独小模型 | 需要模型结构支持 early exit / layer skipping |
| 树/多候选 verification | SpecInfer、Medusa tree、EAGLE dynamic tree | 多条候选 continuation | 视 proposer 而定 | 一次验证更多分支 | attention mask、KV、调度实现复杂 |
| block-parallel draft | DFlash | diffusion-style block drafter | 需要训练 speculator | 规避 AR draft 的串行瓶颈 | 新方案，硬件/后端兼容性还在演进 |

### 2.2 快速选型建议

| 场景 | 优先考虑 | 原因 |
| --- | --- | --- |
| 想快速试、不想训练、不想加载小模型 | N-gram / Suffix | 配置简单，额外显存小 |
| 代码编辑、长文摘录、RAG answer copying、agent 工具调用 | N-gram / Suffix / REST | 输出和上下文重复度高，检索式 draft 很容易命中 |
| 通用聊天、数学、代码生成，希望比较稳定提速 | EAGLE / EAGLE-3 | learned proposer 更能泛化 |
| 目标模型原生支持 MTP | MTP | 配置最省心，通常不需要额外 speculator |
| 有一个小 draft model，但 AR draft 成本太高 | PARD / ParallelSpec 类 | 并行生成多个 draft token，减少 drafter 串行开销 |
| 不想额外部署 draft model，但可接受模型继续训练 | LayerSkip / self-spec | 一个模型内完成 draft 和 verify |
| 大模型服务系统，希望探索树状多候选 | SpecInfer / Medusa tree / EAGLE dynamic tree | 更充分利用 verifier 并行性 |
| 想跟踪 2026 年新方向 | DFlash / block diffusion draft | 把 draft 本身从 AR 链改成并行 block 生成 |

## 3. 主要投机解码方案：What / Why / How

### 3.1 Classic Draft Model / Speculative Sampling

**What**：用一个小语言模型作为 draft model，先自回归生成多个候选 token，再用大 target model 一次性验证。

**Why**：小模型 forward 便宜；大模型验证多个 token 的成本低于逐 token decode 多次。只要 draft 模型和 target 模型足够接近，平均接受长度就会大于 1。

**How**：

1. draft model 从当前上下文生成 `k` 个 token，并记录 draft 分布。
2. target model 对 `k + 1` 个位置给出 logits。
3. rejection sampler 按 target/draft 概率比接受或拒绝，保证目标分布不变。
4. 接受若干 token；拒绝处从 target 的修正分布采样。

适合：有现成小模型、同 tokenizer、同模型族，且服务处于 decode memory-bound 的场景。

参考：

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [vLLM `DraftModelProposer`](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/draft_model.py#L17-L56)

### 3.2 EAGLE / EAGLE-2 / EAGLE-3

**What**：EAGLE 系列不是单纯在 token 层用小模型猜下一个 token，而是利用 target model 的 hidden states，训练轻量 drafter 在特征空间预测未来，从而生成 draft token。

**Why**：论文的核心直觉是：直接预测 token 不稳定，但在更靠近模型内部表示的 feature 层做 autoregression 更容易；同时引入前一个 token 能降低 feature uncertainty。EAGLE-2 进一步用 context-aware dynamic draft tree，根据当前上下文动态决定 tree 形状。EAGLE-3 则是当前工程界常用的强 speculator 方向之一，vLLM Speculators 文档也把它作为可训练/可部署算法。

**How**：

1. target model 正常 forward，输出 token 和中间 hidden states。
2. EAGLE drafter 接收 hidden states 和 token，生成多个 draft token。
3. target model 用一次 verification forward 验证 draft token。
4. 接受后，把 token 和必要状态写回，继续下一轮。

适合：通用聊天、代码、数学等重复性不一定强但希望稳定提速的场景。代价是需要 EAGLE speculator 权重和部署链路。

参考：

- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)
- [vLLM Speculators 文档，Eagle3](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/eagle3/)
- [vLLM `EagleProposer`](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/eagle.py#L10-L32)

### 3.3 MTP: Multi-Token Prediction

**What**：MTP 是模型原生带有多 token 预测能力。和外接 draft model 不同，MTP 的 speculator 是 target 模型训练时就包含的辅助模块或 head。

**Why**：如果模型本身已经训练了 MTP head，推理时不需要再找一个小模型，也不需要额外做 draft-target 模型族适配。vLLM 官方文档也强调：MTP 只适用于 vLLM 已支持的 MTP 模型族。

**How**：

1. target 模型输出 next token，同时 MTP module/head 预测未来若干 token。
2. speculative decoding 把这些未来 token 作为 draft。
3. target 主干再验证这些 draft。
4. 接受前缀，拒绝处回退到 target model 正常采样。

适合：DeepSeek、MiMo、GLM、Gemma 4、Qwen3 Next 等带原生 MTP 能力且 vLLM 已适配的模型。风险是 checkpoint/量化/加载链路可能丢失 MTP 权重，一旦 MTP head 不匹配，acceptance 会很低甚至不可用。

参考：

- [vLLM 官方文档，MTP](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [vLLM `SpeculativeConfig` 中的 MTP model types](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/config/speculative.py#L34-L68)
- [vLLM `Gemma4Proposer`](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/gemma4.py#L31-L69)

### 3.4 Medusa / MLP Speculator

**What**：Medusa 在 target model 上增加多个 decoding heads，每个 head 预测未来不同位置的 token。MLP speculator 可以理解成类似方向：在 hidden states 上接轻量预测模块，产出多 token draft。

**Why**：相比单独小模型，额外 head 的成本更低，且可以共享 target model 的大部分计算。Medusa 还会结合 tree-based attention，同时验证多个候选 continuation。

**How**：

1. target model forward 得到 hidden states。
2. 多个 head 并行预测未来 token。
3. 构造候选 token tree 或多 token 序列。
4. target model 一次验证候选，接受最长有效路径。

适合：可以接受训练/微调额外 head，并希望减少部署额外小模型复杂度的场景。限制是 checkpoint 格式、head 训练质量和框架支持。

参考：

- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [Medusa 官方 GitHub](https://github.com/FasterDecoding/Medusa)
- [vLLM `MedusaProposer`](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/medusa.py#L18-L70)

### 3.5 PARD / Parallel Draft Model

**What**：PARD 关注 draft 阶段自己的串行瓶颈。传统 draft model 也要自回归生成 `k` 个 token，`k` 越大，draft 自己也越慢。PARD 把 autoregressive draft model 改造成 parallel draft model，一次生成多个 draft token。

**Why**：在 EAGLE/draft model 这类 learned proposer 中，draft latency 会成为 hidden bottleneck。并行 draft 允许用更大的 speculation depth，而不线性增加 draft forward 次数。

**How**：

1. 使用经过 PARD adaptation 的 draft model。
2. 打开 `parallel_drafting`，一次输出多个 draft token。
3. target model 正常 verification。

适合：有 PARD 权重，且希望提升 learned proposer 在较大 `num_speculative_tokens` 下的收益。vLLM 文档给出了 `amd/PARD-Qwen3-0.6B` 这类预训练权重示例。

参考：

- [vLLM 官方文档，Parallel Draft Models](https://docs.vllm.ai/en/v0.19.1/features/speculative_decoding/parallel_draft_model/)
- [PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation](https://arxiv.org/abs/2504.18583)
- [P-EAGLE: Faster LLM inference with Parallel Speculative Decoding in vLLM](https://vllm.ai/blog/2026-03-13-p-eagle)

### 3.6 N-gram / Prompt Lookup Decoding

**What**：N-gram speculation 不用神经网络，而是在当前 prompt/已生成 token 中找和当前 suffix 相同的 n-gram，然后把历史中紧跟在匹配片段后的 token 拿来当 draft。

**Why**：很多任务会复制上下文，例如摘要、文档问答、代码编辑、工具调用参数、JSON/XML/Markdown 模板。此时“刚出现过的片段后面是什么”就是很强的 draft 信号。

**How**：

1. 取当前生成序列的 suffix，长度在 `prompt_lookup_min` 到 `prompt_lookup_max` 之间。
2. 在历史 token 中找最长匹配。
3. 把匹配位置后面的若干 token 作为 draft。
4. target model 验证。

适合：重复性强、格式化输出、代码/文档处理。缺点是开放域聊天里命中率不稳定。

参考：

- [vLLM 官方文档，N-Gram Speculation](https://docs.vllm.ai/en/v0.19.0/features/speculative_decoding/n_gram/)
- [vLLM `NgramProposer`](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/ngram_proposer.py#L12-L285)

### 3.7 Suffix Decoding

**What**：Suffix Decoding 是 n-gram / retrieval-style speculation 的增强版。它用 suffix tree 维护当前请求和历史请求中的 token 模式，并用频次统计估计 continuation。

**Why**：相比普通 n-gram，它不只是“在 prompt 中找一次最长匹配”，而是把历史 prompt / 历史输出都纳入一个 suffix cache，能跨请求利用重复模式，并动态决定 speculative length。

**How**：

1. 当前 request 建 local suffix tree。
2. 多个历史 request 的输出进入 global suffix tree。
3. 用最近 token 匹配 suffix tree。
4. 沿着频次最高、概率超过阈值的路径生成 draft。
5. target model 验证。

适合：agent 自反思、代码编辑、RL rollout、self-consistency、多轮规划等高重复 workload。后面第 5-9 节会详细展开。

参考：

- [vLLM 官方文档，Suffix Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/)
- [SuffixDecoding 论文](https://arxiv.org/abs/2411.04975)
- [Snowflake 博客，SuffixDecoding at Production Scale](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/)
- [vLLM `SuffixDecodingProposer`](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/suffix_decoding.py#L9-L97)

### 3.8 Self-Speculative / LayerSkip

**What**：自投机解码不使用额外小模型，而是让 target 模型的“子网络”先 draft，再由完整 target 模型 verify。常见方式是跳过部分中间层、early exit，或训练模型让早层也能给出可用 token 预测。

**Why**：可以避免双模型部署和额外显存，同时保持 target model 自身分布。Draft & Verify 提出通过跳层生成 draft，再由完整模型验证；LayerSkip 则通过 layer dropout 和 early exit loss 训练模型支持这种推理。

**How**：

1. 用较浅层或跳层模型快速生成 draft token。
2. 用完整模型一次 forward 验证。
3. 接受/拒绝逻辑仍遵循 speculative decoding。

适合：有 early-exit 训练 recipe 或模型本身支持 LayerSkip 的场景。对通用 checkpoint 可能不是即插即用。

参考：

- [Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168)
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)
- [Hugging Face 博客，Faster Text Generation with Self-Speculative Decoding](https://huggingface.co/blog/layerskip)

### 3.9 Retrieval-Based Speculative Decoding: REST / CREST / DReSD

**What**：REST 把 draft model 换成 retrieval datastore：根据当前上下文从外部语料或历史数据中检索相似片段，并把检索到的后续 token 当作 draft。

**Why**：很多生成任务会落入常见模式或语料片段，尤其代码、模板、RAG answer copying。检索式 proposer 不需要训练，也不需要额外神经网络 forward。

**How**：

1. 构建 token-level 或 embedding-level datastore。
2. 根据当前上下文查找相似片段。
3. 取相似片段的 continuation 作为 draft。
4. target model 验证。

适合：有可复用语料、领域文本重复度高、希望把 speculation 和检索系统结合的场景。对 UCM 来说，这类方法很值得关注，因为它把“存储系统”直接拉进 proposer 设计里。

参考：

- [REST: Retrieval-Based Speculative Decoding](https://arxiv.org/abs/2311.08252)
- [CREST: Effectively Compacting a Datastore For Retrieval-Based Speculative Decoding](https://arxiv.org/abs/2408.04678)
- [DReSD: Dense Retrieval for Speculative Decoding](https://arxiv.org/abs/2502.15572)

### 3.10 Tree-Based Speculative Verification: SpecInfer 等

**What**：传统 speculative decoding 常验证一条 draft 链；tree-based 方法则一次构造多条候选 continuation，target model 用树状 attention mask 并行验证整棵树。

**Why**：如果只猜一条路径，路径错了就提前停；树结构可以让 verifier 一次检查多个分支，提高“至少有一条分支前缀正确”的概率。

**How**：

1. 一个或多个 small speculative models 生成 candidate token tree。
2. target model 使用 tree attention mask 同时验证树上节点。
3. 系统选择被 target 接受的最长路径。

适合：服务系统愿意承受复杂的 tree mask、KV cache、调度实现，换取更高并行度的场景。Medusa、EAGLE-2 dynamic tree 也可以放到这个大方向里理解。

参考：

- [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781)

### 3.11 DFlash / Block Diffusion Draft

**What**：DFlash 是 2026 年出现的新方向：用小 diffusion-LLM draft model 一次预测一个 token block，而不是像 EAGLE/draft model 那样自回归地产生 draft 链。

**Why**：很多 learned proposer 的瓶颈在 draft 阶段仍然是 sequential AR。DFlash 通过 block-parallel generation，把 draft latency 和 speculation length 解耦。

**How**：

1. target model 产出 hidden states / context features。
2. DFlash drafter 用非因果 attention 和 mask token embedding，一次生成一整块 draft token。
3. target model 验证 draft block。
4. 接受最长有效前缀。

适合：想追踪最新 speculator 方向，且硬件/attention backend 能支持对应非因果 cross-attention 的场景。它还很新，工程兼容性需要谨慎验证。

参考：

- [vLLM Speculators 文档，DFlash](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)
- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [vLLM `DFlashProposer`](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/dflash.py#L21-L102)

## 4. vLLM 当前实现视角

### 4.1 配置入口

vLLM 统一通过 `speculative_config` 配置 speculative decoding。常见字段包括：

| 字段 | 作用 |
| --- | --- |
| `method` | 指定 speculation 方法，例如 `draft_model`、`ngram`、`suffix`、`mtp`、`eagle3`、`dflash`。 |
| `model` | draft model、EAGLE head、DFlash/MTP 辅助权重等。 |
| `num_speculative_tokens` | 每轮最多 draft token 数。 |
| `draft_tensor_parallel_size` | draft model 的 TP 设置。 |
| `parallel_drafting` | 是否启用并行 draft，适用于 EAGLE / draft model / PARD 等。 |
| `prompt_lookup_min/max` | n-gram proposer 的匹配窗口。 |
| `suffix_decoding_*` | suffix decoding 的树深、global cache 请求数、概率阈值等。 |

参考：

- [vLLM 官方文档，`--speculative-config` schema](https://docs.vllm.ai/usage/speculative_decoding/)
- [vLLM `SpeculativeConfig` 源码](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/config/speculative.py#L34-L185)

### 4.2 Proposer 映射表

从 vLLM V1 源码看，GPU model runner 会根据 `speculative_config.method` 选择不同 drafter：

| vLLM method | 对应 proposer | 方案类别 |
| --- | --- | --- |
| `ngram` | `NgramProposer` | 检索式 / prompt lookup |
| `ngram_gpu` | `NgramProposerGPU` | GPU 版 n-gram |
| `suffix` | `SuffixDecodingProposer` | suffix tree / retrieval-style |
| `draft_model` | `DraftModelProposer` | 小模型 draft |
| `eagle` / `eagle3` | `EagleProposer` | hidden-state learned proposer |
| `mtp` | `EagleProposer` 或模型特化 proposer | 原生 MTP / hidden-state proposer |
| `dflash` | `DFlashProposer` | diffusion-style block drafter |
| `medusa` | `MedusaProposer` | 多 head proposer |
| `custom_class` | `create_custom_proposer` | 用户自定义 proposer |

参考：

- [vLLM GPUModelRunner 选择 drafter 的源码](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/worker/gpu_model_runner.py#L535-L610)
- [vLLM spec_decode 模块目录](https://github.com/vllm-project/vllm/tree/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode)

### 4.3 通用执行链

vLLM 里 learned proposer 大多复用 `SpecDecodeBaseProposer`：

1. target model forward 后，runner 拿到 token、positions、hidden states。
2. proposer 生成 `draft_token_ids`。
3. runner 构造 `SpecDecodeMetadata`。
4. target model 对 draft token 做 verification forward。
5. `RejectionSampler` 接受/拒绝 draft token。
6. metrics 记录 accepted/drafted token、per-position acceptance 等。

参考：

- [vLLM `SpecDecodeBaseProposer.propose`](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/llm_base_proposer.py#L421-L638)
- [vLLM `SpecDecodeMetadata`](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/metadata.py#L10-L53)
- [vLLM `RejectionSampler` 调用位置](https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/worker/gpu_model_runner.py#L5978-L5983)

### 4.4 和 UCM/KV cache 的关系

从系统边界看：

- speculative decoding 主要减少 decode iteration 数。
- prefix cache / UCM 外部 KV cache 主要减少 prefill 重算和 HBM 占用。
- tree verification、parallel drafting、DFlash 这类方案会改变每轮 decode 的 query length 和 draft KV 使用方式，因此会影响 KV cache block 分配、CUDA graph、attention backend、KV quantization 兼容性。

所以 UCM 组如果要关注 speculative decoding，重点不是把所有 proposer 都实现一遍，而是看它们对 KV cache 系统提出的新要求：

- draft/verifier 是否需要额外 KV cache。
- rejected token 的 KV 怎么回滚或丢弃。
- tree/block draft 是否需要 branch KV 或特殊 attention mask。
- MTP/EAGLE/DFlash 是否依赖 target hidden states，是否引入额外 layer-wise buffer。
- n-gram/suffix/REST 的重复性信号能否反过来指导 UCM 的 dump、prefetch、admission policy。

## 5. Suffix Decoding 深入：What

### 5.1 从概念上看

`Suffix Decoding` 本质上是 speculative decoding 的一个 proposer。  
标准 speculative decoding 的套路是：

1. 先用一个便宜的方法生成若干个 draft token。
2. 再让 target model 一次性验证这些 draft。
3. 接受最长的正确前缀，拒绝后面的部分。

`Suffix Decoding` 的特殊之处在于：它不用小模型来 draft，而是用 **suffix tree + 频次统计** 来 draft，所以它属于 **model-free speculation**。

vLLM 官方文档对它的定义很直接：它像 n-gram 一样用最近 token 做模式匹配，但比 n-gram 更强，因为它：

- 可以同时匹配 prompt 和历史 generation。
- 用频次统计估计最可能的 continuation。
- 每一步动态决定要 speculate 多少 token，而不是固定长度。

参考：

- [vLLM 官方 Suffix Decoding 文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/)
- [SuffixDecoding 论文（arXiv:2411.04975）](https://arxiv.org/abs/2411.04975)

### 5.2 在 vLLM 代码里它对应什么

在 vLLM 主线里，用户看到的是 `method: "suffix"`；代码里对应的是 `SuffixDecodingProposer`：

- vLLM 会在 speculative config 中识别 `method == "suffix"`。
- 然后 `SuffixDecodingProposer` lazy-import `arctic_inference.suffix_decoding.SuffixDecodingCache`。
- 也就是说，**vLLM 负责把它接到 V1 speculative pipeline 里，而底层 suffix cache / suffix tree 的实现来自 ArcticInference**。

参考：

- [vLLM `SuffixDecodingProposer` 源码](https://github.com/vllm-project/vllm/blob/ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8/vllm/v1/spec_decode/suffix_decoding.py#L9-L97)
- [vLLM `SpeculativeConfig` 中的 suffix 配置与校验](https://github.com/vllm-project/vllm/blob/ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8/vllm/config/speculative.py#L160-L179)
- [vLLM `SpeculativeConfig._validate_suffix_decoding`](https://github.com/vllm-project/vllm/blob/ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8/vllm/config/speculative.py#L644-L678)

### 5.3 “suffix decoding”和“suffix cache”是什么关系

这两个词不能混着理解：

- `Suffix Decoding`：算法/策略层面的名字，是 speculative decoding 的一种。
- `SuffixDecodingCache` / `suffix cache`：它的底层数据结构与缓存系统，是实现该算法的核心组件。

更准确地说：

- `suffix decoding` 是“怎么猜 token”。
- `suffix cache` 是“拿什么来猜 token”。

在当前实现里，这个 cache 由两部分组成：

- **Global suffix tree**：缓存历史请求的输出，服务于跨请求复用。
- **Per-request local suffix tree**：缓存当前活跃请求的 prompt 和已生成 token，服务于当前请求内复用。

参考：

- [ArcticInference 文档：Suffix Decoding](https://arcticinference.readthedocs.io/en/latest/suffix-decoding.html)
- [ArcticInference `SuffixDecodingCache` 源码](https://github.com/snowflakedb/ArcticInference/blob/fba641f8ffbaa25f6715140f4dc85692d6cf7465/arctic_inference/suffix_decoding/cache.py#L59-L310)

## 6. Suffix Decoding 深入：Why

### 6.1 它要解决的不是“通用聊天”，而是“高重复 agentic workload”

论文和官方博客强调的核心观察是：

- 现在越来越多的推理 workload 不是一次性开放问答，而是 agent loop。
- 这类 workload 会反复出现长的重复片段。
- 比如代码修复、执行反馈后重写、multi-path reasoning、self-consistency、self-reflection、多 agent pipeline 等。

这种场景下，如果还沿用“固定长度 speculation + draft model”的思路，往往不能充分利用“重复内容很多”这个结构性特点。`Suffix Decoding` 直接把“历史相似序列”拿来做 continuation 猜测，因此很对症。

参考：

- [vLLM 官方文档：适合高重复任务](https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/)
- [Snowflake 博客：Fastest Speculative Decoding in vLLM with Arctic Inference and Arctic Training](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)

### 6.2 相比 model-based speculation，它几乎“零训练门槛”

和 EAGLE / draft model / MTP 这类方法相比，`Suffix Decoding` 的工程吸引力很强：

- 不需要训练额外 draft model。
- 不需要担心 draft model 与 target model 的配套问题。
- 不占用额外 GPU 去跑 drafter。
- 可以直接利用已有历史输出来增强后续请求。

所以它非常适合下面这种情况：

- 业务侧先要一个“尽快可用”的 speculative decoding 方案。
- workload 自身重复很强。
- 不想引入额外模型训练或额外服务复杂度。

vLLM 文档也把它放在“无额外 draft model、动态 speculation depth”的一类里。

参考：

- [vLLM Speculative Decoding 总览](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [ArcticInference 文档中的方法对比与 benchmark](https://arcticinference.readthedocs.io/en/latest/suffix-decoding.html)

### 6.3 相比 n-gram，它更像“带缓存和统计的增强版检索式 speculation”

它和 n-gram 都是 model-free，但不是同一个级别：

| 维度 | N-gram speculation | Suffix Decoding |
| --- | --- | --- |
| 匹配来源 | 主要看 prompt / 局部 n-gram | 同时看 prompt、当前已生成内容、历史请求输出 |
| continuation 选择 | 匹配到就沿着局部模式走 | 用节点频次估计 continuation 概率 |
| speculate 长度 | 通常更固定 | 根据匹配长度自适应 |
| 跨请求复用 | 很弱 | 强，依赖 global suffix tree |
| 最适合场景 | 轻量、简单重复 | 长输出、高重复、agentic workload |

可以把它理解成：**n-gram 是局部 prompt lookup；suffix decoding 是带跨请求缓存、频次统计和自适应长度控制的模式检索式 speculation。**

### 6.4 为什么它在重复任务上通常更强

因为它的收益来源正好和“重复”高度耦合：

- 匹配越长，说明当前上下文越像历史序列，acceptance 也通常越高。
- 匹配越长，它允许 speculate 越多 token。
- 历史数据越多，global tree 覆盖的 continuation 越丰富。
- 同一批/多轮相似请求跑久了以后，cache 会“越跑越热”。

vLLM 的 e2e 测试就专门验证了这一点：对同一组 prompt 连续跑多轮后，suffix decoding 的 acceptance 长度和 acceptance rate 都会提升，且最终 acceptance rate 期望超过 80%。

参考：

- [vLLM e2e 测试：cache warm-up 后 acceptance 提升](https://github.com/vllm-project/vllm/blob/ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8/tests/v1/e2e/spec_decode/test_spec_decode.py#L226-L284)

### 6.5 性能上大概是什么级别

按论文摘要和官方博客给出的数字，`Suffix Decoding` 在 agentic/repetitive 场景的收益很明显：

- 论文摘要：在 agentic benchmarks 上最高可达 **5.3x speedup**，并且比 EAGLE-2/3 这类 model-based 方法快，说明它在“重复长序列”上非常强。
- Snowflake 的 benchmark 表里，`Suffix Only` 在 SWE-Bench 上是 **286 tok/s**，而无 speculation 是 **75.8 tok/s**，n-gram 是 **175 tok/s**；在人类代码任务 HumanEval 上也明显优于 n-gram。
- 但 vLLM 官方方法选择表的口径更保守：它把 suffix decoding 定位为“低到中等增益，但不需要额外 draft model”的方法。这说明它不是所有 workload 的通杀答案，而是一个非常 workload-sensitive 的加速器。

参考：

- [SuffixDecoding 论文摘要](https://arxiv.org/abs/2411.04975)
- [Snowflake 博客中的对比结果](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)
- [ArcticInference 文档 benchmark 表](https://arcticinference.readthedocs.io/en/latest/suffix-decoding.html)

## 7. Suffix Decoding 深入：How

### 7.1 vLLM 视角下的执行链

从 vLLM 主线源码看，一轮 suffix decoding 的主流程大致如下：

1. 用户配置 `speculative_config={"method": "suffix", ...}`。
2. `SpeculativeConfig` 校验 suffix 参数，并确认本地已安装 `ArcticInference`。
3. vLLM 创建 `SuffixDecodingProposer`。
4. proposer 初始化一个 `SuffixDecodingCache`。
5. 每个 decode step：
   - 如果请求第一次出现，先把 prompt 建成本地 suffix tree。
   - 把刚采样出的 token 追加进 cache。
   - 取最近最多 `max_tree_depth` 个 token 作为 pattern。
   - 用 suffix cache 产生 draft token。
   - target model 一次性验证 draft。
6. 对于已经不在 batch 里的请求，停止其 local tree；其 global response cache 可能继续保留，直到 FIFO eviction。

对应代码里最重要的几行是：

- 新请求第一次出现时，`start_request(req_id, prompt_token_ids)`。
- 每轮 decode 后，`add_active_response(req_id, sampled_ids)`。
- 取 `pattern = token_ids_cpu[i, start:num_tokens]`，只看最近 `max_tree_depth` 个 token。
- `suffix_cache.speculate(...)` 同时在 local tree 和 global tree 上做 speculation。

参考：

- [vLLM `SuffixDecodingProposer` 执行主流程](https://github.com/vllm-project/vllm/blob/ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8/vllm/v1/spec_decode/suffix_decoding.py#L35-L97)

### 7.2 数据结构：为什么是两个树

`SuffixDecodingCache` 里最关键的设计就是“两棵树”：

- **local tree**
  - 每个活跃请求一棵。
  - 存 prompt 和当前请求已生成的 token。
  - 作用是抓住“当前请求内部”的重复。

- **global tree**
  - 所有历史响应共用一棵。
  - 作用是抓住“跨请求”的重复。
  - 支持按请求数上限进行 FIFO eviction。

在 `speculate()` 时，系统会分别在 local tree 和 global tree 上产生 draft，然后取 `score` 更高的那个。

这点非常重要，因为它说明它不是单纯的“当前 prompt lookup”，而是 **当前请求内复用 + 跨请求复用** 的混合体。

参考：

- [ArcticInference `SuffixDecodingCache`：global/local tree 设计](https://github.com/snowflakedb/ArcticInference/blob/fba641f8ffbaa25f6715140f4dc85692d6cf7465/arctic_inference/suffix_decoding/cache.py#L79-L118)
- [ArcticInference `speculate()`：同时比较 local/global draft](https://github.com/snowflakedb/ArcticInference/blob/fba641f8ffbaa25f6715140f4dc85692d6cf7465/arctic_inference/suffix_decoding/cache.py#L235-L310)

### 7.3 suffix tree 里到底存了什么

ArcticInference 的 C++ `SuffixTree` 不是只存 token 串本身，它还在节点上维护了对 speculation 很关键的统计量：

- `count`：有多少 suffix 经过这个节点。
- `children`：后继 token 分支。
- `head_child` / `tail_child` + sibling/group 链表：把子节点按 `count` 从高到低排好，方便快速拿到“最常见 continuation”。
- `ref_seq` / `ref_idx`：通过引用原始序列而不是复制整段 token，实现 path compression，控制内存。

这意味着它做 draft 时不是盲猜，而是在做一种“按频次排序的 continuation 检索”。

参考：

- [ArcticInference `suffix_tree.h`](https://github.com/snowflakedb/ArcticInference/blob/fba641f8ffbaa25f6715140f4dc85692d6cf7465/csrc/suffix_decoding/suffix_tree.h#L29-L177)

### 7.4 匹配和 draft 是怎么生成的

算法的关键点有三个。

#### 1. 先做 context matching

`SuffixTree::speculate()` 会遍历不同的 match length，从短到长尝试匹配当前 context 的后缀：

- 对每个 `match_len = 1..len(context)-1`
- 调用 `_match_context(...)`
- 如果匹配成功，就以这个匹配结果为起点尝试继续往后 draft

#### 2. speculate 长度是自适应的

它不是“每次固定猜 4 个/8 个”，而是：

`max_tokens = min(user_cap, floor(match_len * max_spec_factor + max_spec_offset))`

当前 vLLM 暴露的是 `max_spec_factor`，底层 ArcticInference 还支持 `max_spec_offset`，但 vLLM 主线暂时没有把这个 offset 参数暴露出来。

这背后的直觉是：

- 匹配越长，说明“历史 continuation 这次也成立”的概率更高。
- 那就可以更激进地多猜几个 token。

#### 3. 频次决定 continuation 的优先级

当前默认的 path speculation 会沿着“频次最高的 child”一直走下去：

- child 概率近似看成 `child->count / node->count`
- 如果该概率掉到 `min_token_prob` 以下，就停止继续 speculate

底层 C++ 还实现了 tree-based speculation 版本，但当前 vLLM 的 `SuffixDecodingProposer` 默认走的是 path 模式，没有显式开启 tree mode。

参考：

- [ArcticInference `SuffixTree::speculate`](https://github.com/snowflakedb/ArcticInference/blob/fba641f8ffbaa25f6715140f4dc85692d6cf7465/csrc/suffix_decoding/suffix_tree.cc#L593-L621)
- [ArcticInference `_match_context`](https://github.com/snowflakedb/ArcticInference/blob/fba641f8ffbaa25f6715140f4dc85692d6cf7465/csrc/suffix_decoding/suffix_tree.cc#L800-L823)
- [ArcticInference `_speculate_path`](https://github.com/snowflakedb/ArcticInference/blob/fba641f8ffbaa25f6715140f4dc85692d6cf7465/csrc/suffix_decoding/suffix_tree.cc#L825-L853)
- [ArcticInference `_speculate_tree`](https://github.com/snowflakedb/ArcticInference/blob/fba641f8ffbaa25f6715140f4dc85692d6cf7465/csrc/suffix_decoding/suffix_tree.cc#L873-L905)

### 7.5 一个简化后的伪代码

```text
for each decode step:
    if req is new:
        build local tree from prompt

    append newly accepted/sampled tokens into local/global cache

    pattern = last <= max_tree_depth tokens of current sequence

    local_draft  = local_tree.speculate(pattern)
    global_draft = global_tree.speculate(pattern)
    draft = choose_higher_score(local_draft, global_draft)

    verifier_output = target_model.verify(draft)
    accept longest verified prefix
```

一句话概括：  
**它用 suffix tree 从历史 token 序列里“检索 continuation”，再把检索到的 continuation 当 draft 交给 target model 验证。**

## 8. Suffix cache 和 prefix / KV cache 的区别

这部分非常容易混。

`Suffix Cache` 不是 vLLM 常说的 prefix cache，也不是 KV cache。

### Prefix cache / KV cache 的目标

- 复用已经算过的 attention KV。
- 本质是张量级缓存。
- 解决的是“相同前缀不用再 prefill 一遍”。

### Suffix cache 的目标

- 缓存历史 token 序列模式和频次。
- 本质是 token-pattern cache。
- 解决的是“基于历史重复，猜后面会生成什么 token”。

所以两者差别可以概括成：

- prefix/KV cache：**复用计算结果**
- suffix cache：**复用序列模式**

前者不改变 decoding 方式；后者属于 speculative decoding 的 proposer。

## 9. Suffix Decoding 在 vLLM 当前暴露的关键参数

当前 vLLM 主线对 suffix decoding 暴露的主要参数有四个：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `suffix_decoding_max_tree_depth` | `24` | 限制匹配长度和 speculation 深度 |
| `suffix_decoding_max_cached_requests` | `10000` | global tree 最多缓存多少历史请求；设为 `0` 等于关闭 global cache |
| `suffix_decoding_max_spec_factor` | `1.0` | speculation 长度和匹配长度的比例系数 |
| `suffix_decoding_min_token_prob` | `0.1` | continuation 概率低于阈值就停止 speculate |

另外还有一个通用参数：

- `num_speculative_tokens`
  - 对 suffix decoding 来说它只是**上限**，不是固定 draft 长度。
  - 如果用户不显式设置，当前 vLLM 源码会把它默认成 `suffix_decoding_max_tree_depth`。

参考：

- [vLLM 官方配置文档](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/)
- [vLLM suffix 配置源码](https://github.com/vllm-project/vllm/blob/ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8/vllm/config/speculative.py#L160-L179)
- [vLLM `_validate_suffix_decoding`](https://github.com/vllm-project/vllm/blob/ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8/vllm/config/speculative.py#L644-L678)

### 9.1 一个容易踩坑的点：Arctic 文档和 vLLM 文档的命名不完全一致

如果你同时看 ArcticInference 文档和 vLLM 文档，会发现参数名不完全一样：

| ArcticInference 名称 | vLLM 主线名称 | 说明 |
| --- | --- | --- |
| `suffix_cache_max_depth` | `suffix_decoding_max_tree_depth` | 树深度 |
| `suffix_cache_max_requests` | `suffix_decoding_max_cached_requests` | global cache 请求数上限 |
| `suffix_max_spec_factor` | `suffix_decoding_max_spec_factor` | 自适应长度比例 |
| `suffix_min_token_prob` | `suffix_decoding_min_token_prob` | 最小 token 概率阈值 |

另外，ArcticInference 原生还支持：

- `suffix_max_spec_offset`
- hybrid 模式：`"method": "arctic", "enable_suffix_decoding": true`

而 vLLM 直接集成的 suffix 模式目前更“收敛”，暴露的参数更少。

## 10. 从 UCM 视角补一层：vLLM KV Cache 相关实现

这一节和 `suffix_cache` 不是同一个模块，但对 UCM 同学很关键。因为 UCM 真正接入 vLLM 的位置不是 speculative decoding proposer，而是 vLLM 的 **KV cache 管理和 KV transfer connector**。

### 10.1 vLLM 的 KV cache 基本单位：block

vLLM V1 里 KV cache 的基本管理单位是 `KVCacheBlock`。从 prefix caching 角度看，只有 **完整 block** 才能被 hash、缓存和复用。官方设计文档把 block hash 描述成三类信息的组合：

- parent block hash，保证前缀链路一致。
- 当前 block 的 token ids。
- extra hashes，例如 LoRA、multi-modal placeholder、cache salt 等隔离信息。

参考：

- [vLLM prefix caching 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- [vLLM `KVCacheManager.get_computed_blocks` / `allocate_slots`](https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/v1/core/kv_cache_manager.py#L194-L428)
- [vLLM `BlockPool` 维护 free queue 和 block hash map](https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/v1/core/block_pool.py#L130-L225)

工程上可以把 `BlockPool` 理解成两张表加一个队列：

| 结构 | 作用 |
| --- | --- |
| `free_block_queue` | 维护可分配 block，也承担 prefix cache 的 eviction 顺序。 |
| `cached_block_hash_to_block` | 从 block hash 查到已经缓存的 KV block。 |
| request -> blocks | 记录每个 request 当前占用的 block table。 |

当 request 结束时，block 不一定马上“没价值”：如果它已经带有 block hash，就可以留在 prefix cache 中，直到后续被 LRU/空闲队列策略淘汰。这个设计是 UCM 外部 KV 存储可以接入的基础：vLLM 本地先判断 HBM 中有没有可复用 block，外部 connector 再告诉 vLLM “外部还有多少 token 的 KV 可以加载回来”。

### 10.2 scheduler 侧：本地 hit 和外部 hit 怎么合并

vLLM scheduler 处理一个新 request 时，大致会走下面这条路径：

1. `KVCacheManager.get_computed_blocks(request)` 查本地 prefix cache 命中了多少完整 block。
2. 如果配置了 KVConnector，调用 `connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)` 查询外部 KV cache 命中。
3. scheduler 把本地命中 token 数和外部命中 token 数一起传给 `allocate_slots(...)`。
4. 分配完 block 后，调用 `connector.update_state_after_alloc(...)`，让 connector 记录后续 worker 侧需要 load/dump 哪些 block。
5. 本轮 schedule 结束前，调用 `connector.build_connector_meta(...)` 生成 opaque metadata，传给 worker。

参考：

- [vLLM scheduler 查询本地/外部 KV hit](https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/v1/core/sched/scheduler.py#L590-L751)
- [vLLM scheduler 构建 KVConnector metadata](https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/v1/core/sched/scheduler.py#L906-L927)
- [vLLM `KVConnectorBase_V1` scheduler-side hooks](https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L453-L522)

这对 UCM 的意义是：UCM 不需要改 vLLM 的 attention kernel 才能复用外部 KV。它只要在 scheduler 侧准确回答“这个 request 的前缀有多少 token 的 KV 已在外部可用”，再在 worker 侧把这些 block 搬回 vLLM 分配好的 KV cache 地址即可。

### 10.3 worker 侧：KVConnector 的 load / save 生命周期

worker 侧的生命周期更贴近存储系统：

1. GPU worker 初始化时把 vLLM 的 KV cache tensor 注册给 connector，connector 才知道每层 KV 的地址、shape、stride。
2. forward 前调用 `start_load_kv(...)`，把外部命中的 KV block 加载到 vLLM 当前分配的 block 地址。
3. forward 中/后可以通过 `save_kv_layer(...)` 或 `wait_for_save(...)` 把新算出来的 KV block dump 到外部。
4. worker 把 load/save 完成信息、失败 block id、connector stats 回传给 scheduler。

参考：

- [vLLM GPU worker 注册 KV cache 并触发 pre/post forward](https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/v1/worker/gpu/kv_connector.py#L53-L102)
- [vLLM model runner mixin 中的 connector 生命周期](https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/v1/worker/kv_connector_model_runner_mixin.py#L92-L119)
- [vLLM `KVConnectorBase_V1` worker-side hooks](https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L251-L355)
- [vLLM `KVTransferConfig`](https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/config/kv_transfer.py#L23-L73)

### 10.4 和 suffix_cache 的关系

把两套机制放在一起看：

| 机制 | 缓存对象 | 命中后省掉什么 | 接入位置 |
| --- | --- | --- | --- |
| suffix_cache | token continuation 模式 | 一部分 decoding step；target model 一次验证多个 draft token | speculative decoding proposer |
| prefix cache | HBM 中的 KV block | 重复 prompt 的 prefill 计算 | vLLM KVCacheManager / BlockPool |
| UCM 外部 KV cache | 外部存储中的 KV block | HBM 容量压力和重复 prefill 计算 | vLLM KVConnector |

所以，suffix_cache 对 UCM 的启发不是“把 suffix tree 存进 UCM 就能复用 KV”，而是：LLM 推理加速里有一类 **retrieval-style cache** 思路，既可以检索 token pattern，也可以检索 KV block、稀疏 attention block、CacheBlend chunk。UCM 更适合承载后几类，因为它的核心能力是 KV tensor 的寻址、传输、存储和生命周期管理。

## 11. UCM 相关功能调研

### 11.1 UCM 的核心定位

UCM 全称 Unified Cache Management，它的 README 把核心原则概括为：**持久化 LLM KVCache，并通过检索机制替代冗余计算**。它不只是 prefix cache 的外部存储插件，而是围绕 KV cache 做统一管理，覆盖 prefix caching、训练无关的稀疏 attention 检索、长序列推理，以及基于存储计算分离的 PD disaggregation。

参考：

- [UCM README：核心原则与收益描述](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/README.md#L20-L28)
- [UCM README：为什么需要外部 KV 存储](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/README.md#L32-L59)
- [UCM README：功能列表](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/README.md#L63-L72)

对 UCM 组同学来说，可以把项目拆成三层：

| 层次 | 主要问题 | 代表模块 |
| --- | --- | --- |
| vLLM integration | 怎么让 vLLM 认为外部 KV 是“已计算 token” | `ucm.integration.vllm.ucm_connector` |
| Store abstraction | 怎么用统一接口查、预取、加载、保存 KV block | `UcmKVStoreBaseV1` / C++ `StoreV1` |
| Storage pipeline | KV block 如何跨 Device、Host、SSD、NFS、3FS、压缩模块流动 | `PipelineStore`、`CacheStore`、`PosixStore`、`Ds3fsStore`、`CompressStore` |

### 11.2 UCMConnector 如何接入 vLLM

UCM 的 vLLM 接入点是 `UCMConnector`，它走的是 vLLM `KVConnectorBase_V1` 这套接口。关键实现可以按下面几块读：

| 模块 | 做什么 |
| --- | --- |
| `KVCacheLayout` | 从 vLLM 注册的 KV cache tensor 中提取每层 base pointer、stride、shape、block size 等信息。 |
| `RequestHasher` / `generate_hash` | 基于模型、TP size、dtype、rank 以及 token block 做链式 hash，用于定位外部 KV block。 |
| `get_num_new_matched_tokens` | scheduler 侧查询 UCM store，返回外部命中的 token 数。 |
| `_generate_dispatch_meta` / `build_connector_meta` | 把每个 request 拆成本地已算、外部需要 load、新 block 需要 dump 三类。 |
| `start_load_kv` | worker 侧根据目标 block 地址调用 store load。 |
| `wait_for_save` | worker 侧把新产生的 KV block dump 到外部 store。 |

参考：

- [UCM `KVCacheLayout`](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/integration/vllm/ucm_connector.py#L132-L249)
- [UCM request hash 与 connector 初始化](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/integration/vllm/ucm_connector.py#L257-L404)
- [UCM 外部 KV hit 查询](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/integration/vllm/ucm_connector.py#L514-L593)
- [UCM load/save metadata 构建](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/integration/vllm/ucm_connector.py#L600-L706)
- [UCM worker 侧 load/save](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/integration/vllm/ucm_connector.py#L708-L884)

这个实现里一个很值得关注的细节是：UCM 在 scheduler 侧的命中结果必须和 worker 侧实际 load 的 KV block 对齐。也就是说，`lookup_on_prefix` 回答的是逻辑 token/block 命中，`start_load_kv` 处理的是物理 KV 地址搬运，中间靠 `build_connector_meta` 绑定起来。

### 11.3 Store 抽象：UCM 面向存储的核心接口

UCM 的 store 接口把 sparse algorithm / prefix cache 逻辑和具体外部存储解耦。Python 侧 `UcmKVStoreBaseV1` 和 C++ 侧 `StoreV1` 都围绕几类操作：

- `lookup` / `lookup_on_prefix`：查哪些 block 已经在外部可用。
- `prefetch`：提前发起异步预取。
- `load` / `load_data`：把外部 KV block 加载到给定 device 地址。
- `dump` / `dump_data`：把 device 上的 KV block 写入外部。
- `wait` / `check`：等待异步任务或检查完成状态。

参考：

- [UCM Python `UcmKVStoreBaseV1`](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/store/ucmstore_v1.py#L41-L204)
- [UCM C++ `StoreV1` interface](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/store/ucmstore_v1.h#L33-L148)
- [UCM Extending Store 开发文档](https://ucm.readthedocs.io/en/latest/developer-guide/extending_store.html)

开发文档还明确把 store component 的职责拆成四类：space management、persistence、tiered transfer、data processing。这里的 data processing 很重要，因为它意味着 UCM store 不只是“KV 文件读写”，还可以承载压缩、量化、编码转换等存储侧优化。

### 11.4 当前可见的存储后端与功能

UCM 文档里和存储直接相关的功能点主要有：

| 功能 | 作用 | 参考 |
| --- | --- | --- |
| NFS Store | 将 KV cache 从 GPU HBM offload 到 SSD/local disk/NFS，降低 HBM 压力。 | [NFS Store 文档](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/nfs_store.html) |
| PipelineStore | 通过 store chain 组合 Device->Host、Host->POSIX/3FS/其他后端的传输。 | [PipelineStore 文档](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/pipeline_store.html) |
| Ds3fs Store | 结合 Cache Store 和 3FS 挂载路径，把 KV 写入 3FS 相关存储。 | [Ds3fs Store 文档](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/ds3fs_store.html) |
| Compress Store | 基于 KVfold 对 KV tensor 做压缩/解压，减少磁盘 IO，当前文档描述主要支持 BF16 2.0x。 | [Compress Store 文档](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/compress_store.html) |
| GSA / HATA Sparse Attention | 用 hash-aware similarity 选择相关 KV block，减少 attention 计算和 HBM 使用。 | [GSA 文档](https://ucm.readthedocs.io/en/latest/user-guide/sparse-attention/gsa.html) |
| CacheBlend | 合并多个预计算 KV cache，对拼接文本只重算少量 token。 | [CacheBlend 文档](https://ucm.readthedocs.io/en/latest/user-guide/sparse-attention/cacheblend.html) |
| PD Disaggregation | 面向 prefill / decode 分离，把 KV 生产、传输、消费放到更灵活的架构里。 | [Centralized PD 文档](https://ucm.readthedocs.io/en/latest/user-guide/pd-disaggregation/centralized_pd.html) |

源码里 `PipelineStore` 的构造也很直观：`Cache|Posix`、`Cache|Ds3fs`、`Cache|Compress|Posix` 这类配置会被解析成一条 store pipeline。这样做的好处是，新增一个压缩层、缓存层或远端存储层时，不必重写 vLLM connector。

参考：

- [UCM Pipeline connector：封装 C++ PipelineStore](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/store/pipeline/connector.py#L65-L158)
- [UCM Pipeline builders](https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/store/pipeline/connector.py#L161-L214)

## 12. 面向存储的功能点展望

这一节是基于上面源码和文档的工程推演，不是 UCM 官方 roadmap。对 UCM 这类 KV cache 存储项目，我会重点关注下面几个方向。

### 12.1 多级 KV cache 与放置策略

目前 UCM 已经有 Device、Host、POSIX/NFS/3FS、压缩 pipeline 的雏形。下一步很自然是做更明确的多级缓存策略：

- HBM：只放当前 batch、热 prefix、短期会被 decode 访问的 block。
- CPU DRAM / pinned memory：做 load/save staging、热 block 缓冲、异步预取。
- Local NVMe / SSD：放中热 prefix 和跨请求复用 KV。
- NFS / 3FS / 分布式存储：放跨实例、跨服务生命周期复用的 KV。
- 元数据服务：维护 block hash、位置、版本、热度、租户、过期时间。

真正难的是 admission / eviction：不是所有 KV 都值得 dump，也不是所有外部 hit 都值得 load。可以把 token 长度、prefix 复用概率、存储带宽、当前队列压力、模型 KV size、TTFT 目标一起放进策略里。

### 12.2 block identity 和兼容性治理

KV cache 是强语义数据，错读一个 block 可能不会立刻 crash，但会污染模型输出。因此外部 KV 的 key 不应只看 token hash，还要绑定兼容性元数据：

- model name / revision / quantization。
- dtype、KV layout、block size、head dim、TP/PP/DP rank。
- RoPE scaling、sliding window、attention backend。
- LoRA adapter、multi-modal 输入、cache salt / tenant id。
- UCM store 版本、压缩算法版本、checksum。

vLLM prefix caching 设计里已经有 parent hash、block token 和 extra hashes；UCM 当前 `RequestHasher` 也会把 model、tensor parallel size、dtype、rank 等放进 metadata。后续可以考虑和 vLLM 的 prefix hash 语义进一步对齐，减少“本地 prefix cache 命中规则”和“外部 KV 命中规则”之间的隐性差异。

### 12.3 异步化、layer-wise pipeline 和 prefetch

KV 存储系统最怕把 GPU forward 卡在 I/O 上。UCM 已经有 direct connector 和 layer-wise connector 的方向，后续可以继续强化：

- scheduler 侧提前知道即将运行的 request 后，提前 `prefetch` 外部 block。
- worker 侧按 layer 粒度 overlap：load layer N、forward layer N、dump layer N-1。
- 对长 prefix 大命中场景，支持分块加载，避免必须等所有 block 到齐。
- 对失败 load 使用 vLLM 的 `kv_load_failure_policy`，在 `recompute` 和 `fail` 之间按业务选择。

这里的关键指标不是单次 load 带宽，而是 **TTFT 分解**：hash lookup 时间、metadata 时间、device-host transfer、host-storage transfer、decompress 时间、GPU wait 时间分别占多少。

### 12.4 压缩、量化和数据处理下沉

UCM Compress Store 已经把“存储前后处理”放进 pipeline，这是很适合继续挖的方向：

- 从 BF16 扩到 FP16、FP8、INT8 或更细粒度的 per-layer / per-head KV 压缩。
- 热 block 少压缩或不压缩，冷 block 高压缩，按热度动态选择。
- 解压从 CPU 多线程进一步下沉到 GPU/NPU 或 DSA/RDMA 相关加速路径。
- 压缩策略和 sparse attention 结合：不常被检索的 block 压得更狠，经常命中的局部窗口保留高精度。

这类优化的 trade-off 很清楚：省存储和 I/O，但增加解压延迟和 CPU/内存带宽压力。适合长上下文、高 prefix 命中、存储带宽瓶颈明显的场景。

### 12.5 稀疏检索和存储索引共设计

GSA / HATA 和 CacheBlend 提示了一个方向：UCM 不一定只存 KV tensor，还可以存“帮助找 KV 的索引”。

- 为每个 KV block 存 embedding/hash/signature，用于 query-aware block selection。
- 把稀疏 attention 的 block 选择结果和 store prefetch 结合起来。
- 对 CacheBlend 的 chunk hash、delta rope、重算边界建立元数据索引。
- 对超长上下文，把 lookup 从“完整 prefix 命中”扩展到“相关 block 命中”。

这和 suffix_cache 有一点精神相通：都在做历史序列信息检索。但 suffix_cache 检索的是 token continuation，UCM 稀疏检索更适合检索 KV block 或 block metadata。

### 12.6 可观测性、可靠性和多租户

如果 KV cache 进入外部存储，系统问题会从“算得慢”变成“哪里慢、哪里错、谁污染了谁”。建议重点建设：

- 指标：local hit、external hit、load bytes、dump bytes、lookup latency、load latency、decompress latency、GPU wait、TTFT breakdown。
- 可靠性：checksum、版本化、partial block 处理、后台 GC、坏块隔离。
- 失败策略：load 失败时自动 recompute，或者快速 fail 并暴露明确错误。
- 多租户：tenant namespace、cache salt、访问控制、加密、避免跨租户 timing leak。
- 调试工具：按 request_id 展示 block 从 HBM 到外部 store 的生命周期。

这部分听起来不如 kernel 优化酷，但对生产系统非常要命。KV cache 一旦变成共享存储资源，就需要数据库/缓存系统那套纪律：key schema、版本、隔离、回收、指标和故障语义。

### 12.7 和 suffix decoding 的潜在联动

短期看，suffix decoding 不太适合直接变成 UCM 的 KV 存储功能，因为 speculative draft token 在 target model 验证前没有可靠 KV 可以复用。但它仍有两个启发：

- suffix_cache 的 token 重复统计可以作为 workload 重复性的信号，帮助 UCM 决定哪些 request / prefix 更值得 dump 到外部。
- 如果 suffix proposer 预测到高概率 continuation，UCM 可以尝试提前 prefetch 相关 prefix 或上下文 block，降低后续命中时的 load 延迟。

也就是说，suffix decoding 更适合给 UCM 提供 **调度和预取信号**，而不是直接提供可复用 KV 数据。

## 13. 总结判断

如果从 mentor 汇报的角度，我会把它总结成下面这几句：

1. 投机解码的本质不是“换一种采样”，而是 **用便宜 proposer 提前猜，用 target verifier 保底校验**，以减少大模型 decode forward 的串行次数。
2. 方案选型要看 proposer 成本和 workload，而不是只看 acceptance rate；EAGLE/MTP/DFlash 更像高收益 learned proposer，n-gram/suffix/REST 更像低门槛检索式 proposer。
3. `Suffix Decoding` 是 vLLM 中一种非常有代表性的 **retrieval-style / cache-based speculative decoding**，它和 draft-model 系 speculative decoding 是两条不同思路。
4. `suffix cache` 的真正价值不在“通用聊天都更快”，而在 **高重复、长输出、agentic workflow** 里能吃到跨轮次/跨请求的重复结构。
5. `suffix cache` 和 prefix cache / KV cache 不是一回事；prefix cache 复用 KV，suffix cache 复用 token continuation 模式。
6. 从 UCM 视角看，最值得关注的是 speculative decoding 对 KV cache 系统提出的新要求：draft KV、rejected KV 回滚、tree/block draft 的分支状态、hidden state buffer、以及重复性信号如何指导外部 KV 的 dump/prefetch/admission policy。

## 14. 可直接引用的参考链接

### 官方文档

- vLLM 官方文档，Speculative Decoding 总览与 `--speculative-config`：<https://docs.vllm.ai/usage/speculative_decoding/>
- vLLM 官方文档，N-Gram Speculation：<https://docs.vllm.ai/en/v0.19.0/features/speculative_decoding/n_gram/>
- vLLM 官方文档，Suffix Decoding：<https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/>
- vLLM 官方文档，MTP：<https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/>
- vLLM 官方文档，Parallel Draft Models：<https://docs.vllm.ai/en/v0.19.1/features/speculative_decoding/parallel_draft_model/>
- vLLM 官方 API 文档，`vllm.v1.spec_decode.suffix_decoding`：<https://docs.vllm.ai/en/latest/api/vllm/v1/spec_decode/suffix_decoding/>
- vLLM 官方 API 文档，`SpeculativeConfig`：<https://docs.vllm.ai/en/latest/api/vllm/config/speculative/>
- vLLM Speculators 文档，Getting Started：<https://docs.vllm.ai/projects/speculators/en/stable/user_guide/getting_started/>
- vLLM Speculators 文档，Eagle3：<https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/eagle3/>
- vLLM Speculators 文档，DFlash：<https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/>
- vLLM 官方文档，Automatic Prefix Caching：<https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/>
- vLLM 官方设计文档，Prefix Caching：<https://docs.vllm.ai/en/latest/design/prefix_caching.html>
- ArcticInference 文档，Suffix Decoding：<https://arcticinference.readthedocs.io/en/latest/suffix-decoding.html>
- UCM 官方仓库：<https://github.com/ModelEngine-Group/unified-cache-management>
- UCM 官方文档首页：<https://ucm.readthedocs.io/en/latest/>
- UCM Extending Store 开发文档：<https://ucm.readthedocs.io/en/latest/developer-guide/extending_store.html>
- UCM PipelineStore 文档：<https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/pipeline_store.html>
- UCM NFS Store 文档：<https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/nfs_store.html>
- UCM Ds3fs Store 文档：<https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/ds3fs_store.html>
- UCM Compress Store 文档：<https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/compress_store.html>
- UCM GSA 文档：<https://ucm.readthedocs.io/en/latest/user-guide/sparse-attention/gsa.html>
- UCM CacheBlend 文档：<https://ucm.readthedocs.io/en/latest/user-guide/sparse-attention/cacheblend.html>
- UCM Centralized PD Disaggregation 文档：<https://ucm.readthedocs.io/en/latest/user-guide/pd-disaggregation/centralized_pd.html>

### 论文与博客

- Fast Inference from Transformers via Speculative Decoding：<https://arxiv.org/abs/2211.17192>
- Accelerating Large Language Model Decoding with Speculative Sampling：<https://arxiv.org/abs/2302.01318>
- SpecInfer: Tree-based Speculative Inference and Verification：<https://arxiv.org/abs/2305.09781>
- Draft & Verify: Self-Speculative Decoding：<https://arxiv.org/abs/2309.08168>
- REST: Retrieval-Based Speculative Decoding：<https://arxiv.org/abs/2311.08252>
- Medusa: Multiple Decoding Heads：<https://arxiv.org/abs/2401.10774>
- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty：<https://arxiv.org/abs/2401.15077>
- Better & Faster Large Language Models via Multi-token Prediction：<https://arxiv.org/abs/2404.19737>
- LayerSkip: Early Exit Inference and Self-Speculative Decoding：<https://arxiv.org/abs/2404.16710>
- EAGLE-2: Dynamic Draft Trees：<https://arxiv.org/abs/2406.16858>
- CREST: Compact Retrieval-Based Speculative Decoding：<https://arxiv.org/abs/2408.04678>
- DReSD: Dense Retrieval for Speculative Decoding：<https://arxiv.org/abs/2502.15572>
- PARD: Low-Cost PARallel Draft Model Adaptation：<https://arxiv.org/abs/2504.18583>
- DFlash: Block Diffusion for Flash Speculative Decoding：<https://arxiv.org/abs/2602.06036>
- SuffixDecoding 论文（arXiv 2411.04975）：<https://arxiv.org/abs/2411.04975>
- vLLM 博客，How Speculative Decoding Boosts vLLM Performance：<https://vllm.ai/blog/spec-decode>
- vLLM 博客，P-EAGLE: Faster LLM inference with Parallel Speculative Decoding in vLLM：<https://vllm.ai/blog/2026-03-13-p-eagle>
- Hugging Face 博客，Faster Text Generation with Self-Speculative Decoding：<https://huggingface.co/blog/layerskip>
- Snowflake 博客，Fastest Speculative Decoding in vLLM with Arctic Inference and Arctic Training：<https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/>
- Snowflake 博客，SuffixDecoding at Production Scale with Arctic Inference and vLLM：<https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/>

### 固定到具体 commit 的源码链接

- vLLM 当前 `SpeculativeConfig` 方法枚举与参数：<https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/config/speculative.py#L34-L185>
- vLLM GPUModelRunner 选择 drafter：<https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/worker/gpu_model_runner.py#L535-L610>
- vLLM `SpecDecodeBaseProposer.propose`：<https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/llm_base_proposer.py#L421-L638>
- vLLM `SpecDecodingStats` / metrics：<https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/metrics.py#L18-L116>
- vLLM `NgramProposer`：<https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/ngram_proposer.py#L12-L285>
- vLLM `DraftModelProposer`：<https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/draft_model.py#L17-L56>
- vLLM `EagleProposer`：<https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/eagle.py#L10-L32>
- vLLM `MedusaProposer`：<https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/medusa.py#L18-L70>
- vLLM `DFlashProposer`：<https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/dflash.py#L21-L102>
- vLLM 当前 `SuffixDecodingProposer`：<https://github.com/vllm-project/vllm/blob/07aeaf9d4df870a76d5a0dc19d6a7e74b4be5d3b/vllm/v1/spec_decode/suffix_decoding.py#L9-L97>
- vLLM `SuffixDecodingProposer`：<https://github.com/vllm-project/vllm/blob/ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8/vllm/v1/spec_decode/suffix_decoding.py#L9-L97>
- vLLM `SpeculativeConfig` 中的 suffix 参数：<https://github.com/vllm-project/vllm/blob/ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8/vllm/config/speculative.py#L160-L179>
- vLLM `SpeculativeConfig._validate_suffix_decoding`：<https://github.com/vllm-project/vllm/blob/ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8/vllm/config/speculative.py#L644-L678>
- vLLM e2e 测试（acceptance 随 cache warm-up 上升）：<https://github.com/vllm-project/vllm/blob/ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8/tests/v1/e2e/spec_decode/test_spec_decode.py#L226-L284>
- vLLM `KVCacheManager`：<https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/v1/core/kv_cache_manager.py#L194-L428>
- vLLM `BlockPool`：<https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/v1/core/block_pool.py#L130-L225>
- vLLM scheduler 的外部 KV hit 接入：<https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/v1/core/sched/scheduler.py#L590-L751>
- vLLM `KVConnectorBase_V1`：<https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L453-L522>
- vLLM worker 侧 KVConnector 生命周期：<https://github.com/vllm-project/vllm/blob/73dd2f33b7a5a8a237fe7296039cec246e4c68bd/vllm/v1/worker/gpu/kv_connector.py#L53-L102>
- ArcticInference `SuffixDecodingCache`：<https://github.com/snowflakedb/ArcticInference/blob/fba641f8ffbaa25f6715140f4dc85692d6cf7465/arctic_inference/suffix_decoding/cache.py#L59-L310>
- ArcticInference `suffix_tree.h`：<https://github.com/snowflakedb/ArcticInference/blob/fba641f8ffbaa25f6715140f4dc85692d6cf7465/csrc/suffix_decoding/suffix_tree.h#L29-L177>
- ArcticInference `suffix_tree.cc`：<https://github.com/snowflakedb/ArcticInference/blob/fba641f8ffbaa25f6715140f4dc85692d6cf7465/csrc/suffix_decoding/suffix_tree.cc#L593-L905>
- UCM README：<https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/README.md#L20-L72>
- UCM `UCMConnector`：<https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/integration/vllm/ucm_connector.py#L132-L884>
- UCM `UcmKVStoreBaseV1`：<https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/store/ucmstore_v1.py#L41-L204>
- UCM C++ `StoreV1`：<https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/store/ucmstore_v1.h#L33-L148>
- UCM Pipeline connector：<https://github.com/ModelEngine-Group/unified-cache-management/blob/36bcefa0560a71cb38aef3cabbb55f91ed656507/ucm/store/pipeline/connector.py#L65-L214>
