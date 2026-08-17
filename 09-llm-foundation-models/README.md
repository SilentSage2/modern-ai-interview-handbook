# LLMs & Foundation Models

**Status:** Priority

## Why this matters

This chapter covers the model and training stack behind modern language foundation models before adaptation and agents are introduced.

## Learning objectives

- Explain autoregressive pretraining, tokenization, sampling, and perplexity.
- Distinguish base models, SFT, RLHF, DPO, and in-context learning.
- Explain scaling, MoE, context windows, and inference constraints.

## Chapter map

- Tokenization: BPE/SentencePiece intuition
- Decoder-only causal language modeling
- Next-token prediction objective
- Scaling: parameters, tokens, compute
- Base model vs instruction-following model
- Supervised fine-tuning (SFT)
- Preference learning: RLHF and DPO
- In-context learning and prompting
- Sampling: temperature, top-k, top-p
- MoE LLMs
- Context windows, KV cache and inference bottlenecks


---

## Core concepts and theory

### 1. Autoregressive Language Modeling

For tokens \(x_1,\ldots,x_T\):

\[
p(x_{1:T})
=
\prod_{t=1}^T
p(x_t|x_{<t}).
\]

Maximum likelihood minimizes

\[
\mathcal L
=
-\sum_{t=1}^T
\log p_\theta(x_t|x_{<t}).
\]

A decoder-only Transformer implements these conditional distributions using causal attention.

---

### 2. Tokenization

Text is mapped to integer tokens:

\[
\text{text}\rightarrow (x_1,\ldots,x_T).
\]

Subword tokenization balances:
- character-level flexibility;
- word-level efficiency;
- manageable vocabulary size.

Vocabulary size affects:
- embedding/output matrix size;
- sequence length;
- multilingual/code efficiency.

---

### 3. Logits and Softmax

Model outputs logits \(z\in\mathbb R^{V}\).

\[
p_i
=
\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

Cross entropy for target token \(y\):

\[
\mathcal L=-\log p_y.
\]

---

### 4. Perplexity

Average token NLL:

\[
\mathrm{NLL}
=
-\frac1T\sum_t\log p(x_t|x_{<t}).
\]

Perplexity:

\[
\mathrm{PPL}=e^{\mathrm{NLL}}.
\]

Interpretation:
lower perplexity means the model assigns higher probability to observed text.

Do not treat perplexity as a complete measure of instruction following, factuality, reasoning, or usefulness.

---

### 5. Temperature

Given logits \(z_i\), temperature \(T\):

\[
p_i(T)
=
\frac{e^{z_i/T}}
{\sum_j e^{z_j/T}}.
\]

- \(T<1\): sharper distribution.
- \(T>1\): flatter distribution.

---

### 6. Top-k and Top-p

#### Top-k
Keep only the \(k\) highest-probability tokens.

#### Top-p / nucleus sampling
Choose the smallest set \(S\) such that

\[
\sum_{i\in S}p_i\ge p.
\]

Top-p adapts candidate-set size to model uncertainty.

---

### 7. Foundation Model Paradigm

Classical task-specific learning:

\[
D_A\to M_A,\qquad D_B\to M_B.
\]

Foundation modeling:

\[
D_{\rm broad}
\to M_0
\to
\{M_A,M_B,\ldots\}.
\]

Downstream adaptation can happen through:
- prompting;
- in-context examples;
- linear probes;
- full fine-tuning;
- PEFT/LoRA;
- retrieval augmentation.

---

### 8. SFT

Given instruction \(x\), target response \(y\):

\[
\mathcal L_{\rm SFT}
=
-\sum_{t\in \text{response}}
\log p_\theta(y_t|x,y_{<t}).
\]

Often prompt tokens are masked from the loss so training focuses on response generation.

---

### 9. Preference Data

Preference pair:

\[
(x,y_w,y_l)
\]

where \(y_w\) is preferred over \(y_l\).

This supports reward modeling or direct preference optimization.

---

### 10. RLHF Conceptual Pipeline

1. SFT policy \(\pi_{\rm ref}\).
2. Collect human preferences.
3. Train reward model \(r_\phi(x,y)\).
4. Optimize policy reward while constraining it from drifting too far from reference.

A typical objective resembles

\[
\max_\theta
\mathbb E_{y\sim\pi_\theta}
[r_\phi(x,y)]
-
\beta
D_{\rm KL}
(\pi_\theta\|\pi_{\rm ref}).
\]

The KL term stabilizes optimization and preserves language quality.

---

### 11. DPO Derivation Intuition

DPO avoids explicitly training a reward model followed by RL.

For preferred \(y_w\) and rejected \(y_l\), optimize

\[
-\log
\sigma
\left(
\beta
[
\log\pi_\theta(y_w|x)
-
\log\pi_{\rm ref}(y_w|x)
-
\log\pi_\theta(y_l|x)
+
\log\pi_{\rm ref}(y_l|x)
]
\right).
\]

Intuition:
increase the preferred answer's log-probability relative to the reference and decrease the rejected answer's relative log-probability.

---

### 12. In-Context Learning

Weights remain fixed:

\[
\theta'=\theta.
\]

Task information enters through the context:

\[
p(y|x,\text{demonstrations};\theta).
\]

This is different from gradient-based fine-tuning.

---

### 13. Scaling

Useful conceptual axes:
- model size \(N\);
- training tokens \(D\);
- compute \(C\).

The practical lesson is that performance depends on balancing all three. A model can be undertrained if parameter count grows without enough data/compute.

---

### 14. MoE

Router produces expert scores:

\[
g(x)=\mathrm{softmax}(W_rx).
\]

Select top-\(k\) experts:

\[
y
=
\sum_{i\in \mathrm{TopK}(g)}
g_i(x)E_i(x).
\]

Benefit:
large parameter capacity while activating only a subset per token.

Challenge:
- load balancing;
- communication;
- expert collapse/imbalance;
- serving complexity.

<!-- DEEP_DIVE_START -->
## Deep dive I: from text to training examples

A causal LM training sequence might be:

```text
[BOS] The cat sat on the mat [EOS]
```

Shifted training:

```text
input : [BOS] The cat sat on the mat
target: The  cat sat on  the mat [EOS]
```

The model outputs logits

\[
Z\in\mathbb R^{B\times T\times V}.
\]

Cross entropy compares each position with the next target token.

One forward pass therefore creates \(T\) next-token training examples.

---

## Deep dive II: teacher forcing

During training, position \(t\) receives the **ground-truth previous tokens**.

During inference, it receives previously generated tokens.

This creates exposure mismatch:

\[
\text{training context}\sim p_{\rm data},
\qquad
\text{generation context}\sim p_\theta.
\]

Autoregressive models can compound earlier generation errors because future predictions condition on their own outputs.

---

## Deep dive III: tokenization engineering

Vocabulary size \(V\) creates a tradeoff.

Large vocabulary:
- fewer tokens per text;
- larger embedding/output matrices.

Small vocabulary:
- smaller output space;
- longer sequences.

For code or multilingual text, tokenizer efficiency materially affects context usage and training compute.

A tokenizer is therefore not merely preprocessing; it changes the model's computational representation of data.

---

## Deep dive IV: embedding and output projection

Input token ID \(i\) selects row:

\[
e_i=E[i],
\qquad
E\in\mathbb R^{V\times d}.
\]

Final hidden state:

\[
h_t\in\mathbb R^d.
\]

Output logits:

\[
z_t=W_{\rm out}h_t.
\]

Often input embedding and output projection weights are tied:

\[
W_{\rm out}=E.
\]

This reduces parameters and encourages shared token geometry.

---

## Deep dive V: pretraining vs post-training

### Pretraining asks
“What token is likely next?”

It creates broad language/world representations.

### SFT asks
“What response should an assistant produce for this instruction?”

It changes behavioral format.

### Preference optimization asks
“Among plausible responses, which behavior should be preferred?”

These stages solve distinct problems.

A strong answer should not call all three “fine-tuning.”

---

## Deep dive VI: SFT masking example

Sequence:

```text
<user> Explain attention.
<assistant> Attention computes ...
```

You may train only on assistant tokens.

Loss mask:

```text
user tokens      → ignore
assistant tokens → compute CE
```

Formally:

\[
\mathcal L
=
-\sum_t m_t\log p(y_t|y_{<t}),
\qquad
m_t\in\{0,1\}.
\]

This prevents the model from being optimized to reproduce the prompt itself.

---

## Deep dive VII: DPO intuition with numbers

Suppose relative log-probability improvement over reference is:

\[
\Delta_w
=
\log\pi_\theta(y_w|x)-\log\pi_{\rm ref}(y_w|x)=1.2,
\]

\[
\Delta_l
=
\log\pi_\theta(y_l|x)-\log\pi_{\rm ref}(y_l|x)=0.1.
\]

Then preferred-minus-rejected margin is:

\[
1.2-0.1=1.1.
\]

DPO rewards positive margin.

If the rejected answer improves more than the preferred answer, the margin becomes negative and loss increases.

The reference model provides an anchor so optimization is relative to the starting policy rather than unconstrained probability growth.

---

## Deep dive VIII: why scaling is not only “bigger model”

Compute roughly depends on both model size and tokens.

A very large model trained on too few tokens may be undertrained.

A smaller model trained on much more appropriate data may outperform it for a fixed compute budget.

When discussing scaling, mention:
- parameter count;
- token count;
- data quality;
- compute budget;
- architecture;
- inference cost.

---

## Deep dive IX: MoE routing

For token hidden state \(x\), router:

\[
g=W_rx.
\]

Choose top-\(k\) experts:

\[
S=\mathrm{TopK}(g).
\]

Output:

\[
y=\sum_{i\in S}p_iE_i(x).
\]

The attraction is **conditional computation**.

If a dense model has \(N\) parameters, all relevant layers are active for every token. MoE can have much larger total parameters while activating only a fraction.

But experts may become imbalanced. A routing/load-balancing objective is often needed so one expert does not receive nearly all tokens.

---

## Deep dive X: prompt length and inference cost

Long prompt cost has two components:

### Prefill
Dense attention over prompt:

\[
O(T^2d)
\]

for standard attention, though optimized kernels reduce memory traffic.

### Decode
Each new token attends to cached history. Compute per step grows roughly with context length, and KV memory grows linearly with stored tokens.

Therefore “supports a long context window” is not the same as “long context is cheap.”

---

## Minimal causal-LM training sketch

```python
tokens = tokenizer(text, return_tensors="pt").input_ids
inp = tokens[:, :-1]
target = tokens[:, 1:]

logits = model(inp)                       # [B,T,V]
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    target.reshape(-1)
)
```

Real training additionally requires:
- padding/attention masks;
- packed sequences;
- distributed training;
- mixed precision;
- checkpointing;
- evaluation and data filtering.

---

## Common LLM interview traps

**“Does lower perplexity guarantee better chatbot quality?”**  
No.

**“Does instruction tuning inject new factual knowledge reliably?”**  
It can change knowledge behavior, but external/fresh knowledge is often better handled with retrieval.

**“Is chain-of-thought the same as an agent?”**  
No. An agent additionally interacts with tools/environment/state in a loop.

**“Is an LLM a world model?”**  
It may contain predictive world knowledge, but world-model roles usually emphasize explicit environment dynamics/action consequences and planning.
<!-- DEEP_DIVE_END -->

<!-- SECOND_DEEP_DIVE_START -->
## Architecture case study: modern decoder-only LLM components

A typical modern decoder-only LLM may combine:
- token embeddings;
- RMSNorm;
- RoPE;
- causal attention;
- GQA;
- gated MLP such as SwiGLU;
- repeated pre-norm residual blocks;
- final vocabulary head.

The important interview skill is to explain *why* each evolved from the original Transformer.

| Component | Motivation |
|---|---|
| RMSNorm | simpler normalization |
| RoPE | relative-position behavior in Q/K |
| GQA | reduce KV-cache cost |
| SwiGLU | stronger gated MLP |
| pre-norm | deep optimization stability |

Do not treat “Transformer” as frozen in 2017.

---

## Causal-LM loss with padding

Suppose batch has variable lengths.

Token mask:

\[
m_{bt}=
\begin{cases}
1&\text{valid target}\\
0&\text{padding/ignored}
\end{cases}.
\]

Loss:

\[
L
=
-
\frac{
\sum_{b,t}
m_{bt}
\log p_\theta(y_{bt}|x_{b,<t})
}{
\sum_{b,t}m_{bt}
}.
\]

Without masking, model would be trained to predict artificial padding symbols.

---

## Packing

Instead of padding many short examples to max length, concatenate multiple examples into one sequence block while preserving boundaries.

Goal:

\[
\text{useful tokens per batch}
\uparrow.
\]

But attention/loss masks must prevent unintended information leakage across examples when required by the formatting scheme.

Packing is an example of a data-pipeline detail that materially changes GPU utilization.

---

## Why output vocabulary projection is expensive

Logits:

\[
Z=HW_{\rm vocab}^\top,
\]

where

\[
H\in\mathbb R^{BT\times d},
\qquad
W_{\rm vocab}\in\mathbb R^{V\times d}.
\]

If \(V\) is large, this matrix multiplication and logits storage can be expensive.

Training typically needs logits for all vocabulary entries to compute softmax cross entropy.

Inference only needs enough probability information to choose/sample next tokens but still computes vocabulary logits in standard architectures.

---

## Sequence log-probability

For response

\[
y=(y_1,\ldots,y_T),
\]

conditional log probability is

\[
\log\pi_\theta(y|x)
=
\sum_t
\log\pi_\theta(y_t|x,y_{<t}).
\]

Preference methods like DPO operate on these sequence-level sums/differences.

Length can therefore affect raw log-probability comparisons; implementations need consistent formulation.

---

## Reward model formulation

Given prompt-response pair \((x,y)\), reward model outputs scalar:

\[
r_\phi(x,y).
\]

Preference training can use Bradley–Terry style probability:

\[
P(y_w\succ y_l)
=
\sigma(
r_\phi(x,y_w)-r_\phi(x,y_l)
).
\]

Loss:

\[
L
=
-\log\sigma(r_w-r_l).
\]

This teaches relative ranking rather than an absolute “true reward.”

---

## RLHF vs DPO at a systems level

### RLHF with PPO
Needs:
- policy generation;
- reward evaluation;
- value estimation;
- reference model/KL;
- PPO rollout/update machinery.

### DPO
Needs:
- preferred/rejected pairs;
- current model log-probs;
- reference model log-probs.

DPO is operationally simpler but does not make RL concepts obsolete; RL remains important for online/environmental rewards, verifiable tasks, agents, and world models.

---

## Data quality in pretraining

More tokens are not automatically better.

Important dimensions:
- duplication;
- contamination;
- language/domain balance;
- code quality;
- unsafe/low-quality content;
- synthetic-data feedback;
- temporal freshness.

Foundation-model performance is a function of data curation as much as architecture.

---

## Evaluation taxonomy

### Intrinsic
- NLL/perplexity.

### Knowledge/reasoning
- benchmark accuracy;
- exact match;
- multiple choice.

### Generation
- human preference;
- factuality;
- instruction adherence.

### Code
- unit-test success;
- pass@k.

### Deployment
- latency;
- tokens/sec;
- memory;
- cost.

A single benchmark score cannot characterize an LLM.

---

## Hallucination as probabilistic generation

An LM generates a high-probability continuation under learned distribution, not guaranteed truth.

If context does not uniquely determine factual answer, the model can still produce fluent text.

Methods that improve grounding:
- retrieval;
- tools;
- verification;
- constrained decoding;
- post-training;
- uncertainty/refusal policies.

The key mental model:

> language modeling optimizes plausibility under training distribution, not a formal truth constraint.

---

## Mini-project: SFT a small LLM

Pipeline:

```text
dataset
→ chat template
→ tokenizer
→ response loss mask
→ base model
→ LoRA
→ SFT
→ held-out evaluation
```

Compare:
- base model;
- prompted base model;
- SFT LoRA model.

Evaluate:
- task score;
- format compliance;
- general-domain regression;
- trainable params;
- GPU memory.

This project covers both LLM and fine-tuning modules.
<!-- SECOND_DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Separate pretraining, post-training, and inference.
- Base-model likelihood quality is not the same as instruction-following quality.
- During decoding, reason in terms of logits → sampling policy → token → KV-cache update.

---

## Hands-on / practice

### Level 1 — Reproduce
Implement or run a canonical example that demonstrates the central idea.

### Level 2 — Compare
Create at least one controlled comparison (baseline vs method, accuracy vs compute, or full vs efficient version).

### Level 3 — Explain
Write:
- what you changed;
- why it worked or failed;
- GPU memory / runtime where relevant;
- one figure or table;
- a 2-minute interview explanation.

## Deliverables
- [ ] runnable code
- [ ] README with commands
- [ ] experiment configuration
- [ ] quantitative result
- [ ] failure-case notes
- [ ] interview-ready project summary

---

## Interview readiness checklist

Before marking this chapter ready, make sure you can:

- explain the main idea without notes;
- write the important equations from memory;
- discuss at least one design tradeoff;
- compare the method with its nearest alternative;
- identify at least one failure mode;
- connect the theory to a real implementation or project.

For dedicated interview questions, see [`interview_qa.md`](interview_qa.md).  
For papers and official documentation, see [`references.md`](references.md).
