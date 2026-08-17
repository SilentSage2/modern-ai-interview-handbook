# Transformers

**Status:** Priority

## Why this matters

Transformer knowledge is the central bridge connecting LLMs, ViTs, VLMs, multimodal systems, and many world models.

## Learning objectives

- Derive scaled dot-product attention and explain Q/K/V.
- Explain MHA, causal masking, RoPE, LayerNorm/RMSNorm.
- Reason about KV cache, prefill/decode, GQA/MQA, and FlashAttention.

## Chapter map

- Token embeddings and sequence representations
- Q/K/V and scaled dot-product attention
- Multi-head attention
- Causal vs bidirectional attention
- Feed-forward networks and residual blocks
- LayerNorm / RMSNorm
- Absolute, relative and rotary positional embeddings (RoPE)
- Attention complexity O(N^2)
- KV cache, prefill and decode
- FlashAttention intuition
- Mixture-of-Experts routing


---

## Core concepts and theory

### 1. From Sequence to Matrix

Let a sequence contain \(N\) tokens with hidden dimension \(d\):

\[
X\in\mathbb R^{N\times d}.
\]

A Transformer layer should allow each token to aggregate information from other tokens.

The central operation is attention.

---

## 2. Query, Key, Value

Project each token into three spaces:

\[
Q=XW_Q,\qquad
K=XW_K,\qquad
V=XW_V.
\]

Shapes:

\[
W_Q,W_K\in\mathbb R^{d\times d_k},
\qquad
W_V\in\mathbb R^{d\times d_v}.
\]

Thus

\[
Q,K\in\mathbb R^{N\times d_k},
\qquad
V\in\mathbb R^{N\times d_v}.
\]

For token \(i\), query \(q_i\) is compared against all keys \(k_j\):

\[
s_{ij}=q_i^\top k_j.
\]

This produces

\[
S=QK^\top\in\mathbb R^{N\times N}.
\]

Interpretation:
- Query: what information does this token seek?
- Key: what information does another token advertise?
- Value: what content should be transferred if attended to?

---

## 3. Why Scale by \(\sqrt{d_k}\)?

Assume elements of \(q\) and \(k\) are independent with

\[
\mathbb E[q_l]=\mathbb E[k_l]=0,\qquad
\mathrm{Var}(q_l)=\mathrm{Var}(k_l)=1.
\]

The dot product is

\[
q^\top k=\sum_{l=1}^{d_k}q_lk_l.
\]

Each product has roughly unit variance, so

\[
\mathrm{Var}(q^\top k)\approx d_k.
\]

Therefore its standard deviation grows as

\[
\sqrt{d_k}.
\]

Large logits push softmax into saturation, causing near one-hot weights and poor gradients.

Scale:

\[
\tilde S=\frac{QK^\top}{\sqrt{d_k}}.
\]

Then the logit variance remains approximately \(O(1)\).

---

## 4. Softmax Attention

Row-wise softmax gives

\[
A_{ij}
=
\frac{\exp(\tilde S_{ij})}
{\sum_{m=1}^N\exp(\tilde S_{im})}.
\]

Each row sums to 1.

Output:

\[
Y=AV.
\]

For token \(i\):

\[
y_i=\sum_{j=1}^N A_{ij}v_j.
\]

So attention is a learned, input-dependent weighted average of value vectors.

---

## 5. Causal Masking

Autoregressive language modeling requires token \(i\) to use only tokens \(j\le i\).

Define

\[
M_{ij}
=
\begin{cases}
0,&j\le i\\
-\infty,&j>i.
\end{cases}
\]

Then

\[
A=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right).
\]

Since \(e^{-\infty}=0\), future positions receive zero probability.

---

## 6. Multi-Head Attention

A single attention map is restrictive. Use \(h\) independent projections:

\[
Q_i=XW_Q^{(i)},
\quad
K_i=XW_K^{(i)},
\quad
V_i=XW_V^{(i)}.
\]

Then

\[
H_i=
\mathrm{Attention}(Q_i,K_i,V_i).
\]

Concatenate:

\[
H=\mathrm{Concat}(H_1,\ldots,H_h).
\]

Project:

\[
Y=HW_O.
\]

If model dimension is \(d\), a common design is

\[
d_k=d_v=\frac dh.
\]

This keeps total attention width approximately fixed while allowing different representation subspaces.

---

## 7. Why Attention is \(O(N^2)\)

The score matrix

\[
QK^\top
\]

has \(N^2\) entries.

Compute complexity:

\[
O(N^2d_k).
\]

Memory for attention weights:

\[
O(N^2).
\]

This is the central bottleneck for very long contexts.

---

## 8. Transformer Block

A simplified pre-norm block:

\[
H_1=X+\mathrm{MHA}(\mathrm{LN}(X)),
\]

\[
H_2=H_1+\mathrm{FFN}(\mathrm{LN}(H_1)).
\]

FFN is token-wise:

\[
\mathrm{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2.
\]

Attention mixes information **across tokens**.

FFN mixes information **across channels/features within each token**.

This distinction is very useful in interviews.

---

## 9. LayerNorm

For hidden vector \(x\in\mathbb R^d\),

\[
\mu=\frac1d\sum_i x_i,
\]

\[
\sigma^2=\frac1d\sum_i(x_i-\mu)^2.
\]

Normalize:

\[
\hat x_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}.
\]

Affine transform:

\[
y_i=\gamma_i\hat x_i+\beta_i.
\]

LayerNorm does not depend on other samples in the batch.

---

## 10. RMSNorm

RMSNorm removes mean subtraction:

\[
\mathrm{RMS}(x)
=
\sqrt{\frac1d\sum_i x_i^2+\epsilon},
\]

\[
y_i=\gamma_i\frac{x_i}{\mathrm{RMS}(x)}.
\]

It is computationally simpler and widely used in modern LLMs.

---

## 11. Positional Information

Self-attention alone is permutation equivariant.

Without positions, reordering tokens reorders outputs but does not tell the model which order is correct.

### Sinusoidal encoding

\[
PE_{(pos,2i)}
=
\sin\left(
pos/10000^{2i/d}
\right),
\]

\[
PE_{(pos,2i+1)}
=
\cos\left(
pos/10000^{2i/d}
\right).
\]

---

## 12. RoPE Intuition

RoPE rotates query/key vector pairs by position-dependent angles.

For 2D pair:

\[
R(\theta_m)
=
\begin{bmatrix}
\cos\theta_m&-\sin\theta_m\\
\sin\theta_m&\cos\theta_m
\end{bmatrix}.
\]

At positions \(m,n\):

\[
q_m=R(\theta_m)q,
\qquad
k_n=R(\theta_n)k.
\]

Their inner product:

\[
q_m^\top k_n
=
q^\top R(\theta_m)^\top R(\theta_n)k
=
q^\top R(\theta_n-\theta_m)k.
\]

Therefore the interaction naturally depends on **relative position \(n-m\)**.

That is the key derivation to remember.

---

## 13. KV Cache

During autoregressive generation, previous K/V vectors do not change.

At step \(t\), instead of recomputing:

\[
K_{1:t},V_{1:t},
\]

store

\[
K_{1:t-1},V_{1:t-1}
\]

and compute only

\[
k_t,v_t.
\]

Then query \(q_t\) attends to the cached history.

#### Memory
For \(L\) layers, \(H_{kv}\) KV heads, head dimension \(d_h\), sequence \(T\):

\[
\mathrm{KV\ elements}
\approx
2LTH_{kv}d_h.
\]

Multiply by bytes per element and batch size.

This is why long-context LLM inference is often memory-bandwidth and memory-capacity limited.

---

## 14. Prefill vs Decode

### Prefill
Process all prompt tokens simultaneously.

Main characteristics:
- large matrix multiplications;
- parallel;
- compute intensive;
- builds KV cache.

### Decode
Generate one/few token(s) per step.

Main characteristics:
- repeatedly reads weights + KV cache;
- often memory-bandwidth limited;
- latency sensitive.

---

## 15. MQA and GQA

Standard MHA has separate K/V per attention head.

#### Multi-Query Attention
Many Q heads share one K/V head.

#### Grouped-Query Attention
Groups of Q heads share K/V heads.

Benefit:
\[
\text{KV cache memory}\downarrow
\]

while retaining more expressiveness than a single shared KV head.

---

## 16. FlashAttention

FlashAttention computes exact attention but changes the execution strategy.

Naive implementation materializes large \(N\times N\) intermediates in high-bandwidth memory.

FlashAttention:
- tiles Q/K/V;
- keeps blocks in fast on-chip memory;
- recomputes/selectively accumulates softmax statistics;
- reduces HBM traffic.

Key interview answer:

> FlashAttention is mainly an IO-aware algorithm, not an approximation to attention.

---

## 17. Common Interview Comparisons

#### Attention vs convolution
- attention: dynamic weights, global interaction;
- convolution: fixed learned kernels, local prior, translation equivariance.

#### Encoder vs decoder
- encoder attention is typically bidirectional;
- decoder-only LLM uses causal self-attention.

#### Pre-norm vs post-norm
Pre-norm puts normalization before sublayer and generally improves optimization stability for deep Transformers.

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Track shapes of Q, K, V and the attention matrix.
- Separate token mixing (attention) from channel mixing (FFN).
- For systems questions, connect long context to O(N²) attention during prefill and KV-cache growth during decode.
- Be able to implement causal attention from scratch in PyTorch without relying on a high-level Transformer class.

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
