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

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Separate pretraining, post-training, and inference.
- Base-model likelihood quality is not the same as instruction-following quality.
- During decoding, reason in terms of logits → sampling policy → token → KV-cache update.

---

## Hands-on / practice

## Level 1 — Reproduce
Implement or run a canonical example that demonstrates the central idea.

## Level 2 — Compare
Create at least one controlled comparison (baseline vs method, accuracy vs compute, or full vs efficient version).

## Level 3 — Explain
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
