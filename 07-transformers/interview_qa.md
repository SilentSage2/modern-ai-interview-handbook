# Transformers — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. Explain Q, K, and V and derive scaled dot-product attention.

Let hidden states be `X` with shape `[B, T, d_model]`. Learned linear projections create queries, keys, and values.

- **Query (Q):** what information this token is seeking.
- **Key (K):** what information a candidate token advertises.
- **Value (V):** what content should be transferred if that candidate receives attention.

For one head:

```math
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}

\right)V.
```

`QK^T` creates compatibility scores; softmax converts them to weights; the weighted sum of V transfers information.

**Why divide by sqrt(d_k)?** A dot product of `d_k` roughly unit-variance terms has variance proportional to `d_k`. Scaling prevents logits from growing with head dimension and saturating softmax.

**Design idea.** Attention is content-dependent routing: connection weights depend on the current input rather than being fixed after training.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Why multi-head attention instead of one large head?

A single head uses one projection space to represent all relationships. Multi-Head Attention (MHA) learns several Q/K/V projection spaces in parallel, allowing different relational patterns within the same layer.

If model width is `d_model` and there are `h` heads, each head often uses `d_head = d_model / h`. Head outputs are concatenated and projected back to model width.

**Important nuance.** Heads are allowed to specialize but are not guaranteed to have clean human-interpretable roles; some can be redundant.

**Systems connection.** Full MHA stores separate key/value states for each head. MQA and GQA reduce K/V head count to shrink inference cache memory.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. Why do Transformers need positional information, and what is RoPE?

Content-only self-attention is permutation equivariant: rearranging the input rearranges the outputs, but the model has no intrinsic notion of sequence order.

Rotary Position Embedding (RoPE) rotates query/key feature pairs by position-dependent angles. The central identity is:

```math
(R_m q)^\top(R_n k)
=
q^\top R_{n-m}k.
```

The query-key interaction therefore depends naturally on relative position difference.

**Design idea.** Instead of simply adding a positional vector to token content, RoPE changes the geometry of attention matching itself. This makes relative position information directly available in Q/K dot products.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. How can Transformer training be parallel if generation is autoregressive?

During training, the full target sequence is already known. A causal mask prevents each position from seeing future tokens, but all positions can still be evaluated in one batched matrix computation.

At inference, token `t+1` does not exist until the model samples it, so new tokens must be generated sequentially.

**Key distinction.**
- Training is parallel across known positions.
- Autoregressive inference is sequential across newly generated positions.

This explains why Transformers removed the recurrent dependency bottleneck of RNN training while LLM decoding still has a sequential latency component.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. What is the KV cache and why is it both helpful and expensive?

During autoregressive generation, keys and values for earlier tokens do not change. The Key–Value (KV) cache stores them once and appends only the new token's K/V at each step.

**Benefit:** avoids recomputing past K/V, drastically reducing repeated computation.

**Cost:** memory grows with batch size, layer count, context length, number of K/V heads, head dimension, and bytes per value:

```math
M_{\mathrm{KV}}
\propto
2
B
L
T
H_{\mathrm{KV}}
d_h
\cdot
\text{bytes}.
```

This is why Grouped-Query Attention (GQA), Multi-Query Attention (MQA), paged KV cache, and quantization are important in LLM serving.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. Prefill versus decode: why do they have different bottlenecks?

**Prefill** processes the entire prompt. It uses large matrix multiplications and dense attention over many tokens, exposing high parallelism and often substantial compute.

**Decode** adds one token per sequence per step. Each step performs relatively little new arithmetic but repeatedly reads model weights and an increasingly large KV cache. This often makes decode memory-bandwidth limited.

**Consequences.**
- FlashAttention improves attention IO and is particularly valuable for large prompt attention.
- Quantization can strongly help decode by reducing bytes read for weights.
- Batching improves decode throughput by giving the GPU more simultaneous tokens.

The two phases should be benchmarked separately.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q7. What is FlashAttention actually optimizing?

FlashAttention computes the same exact dense attention result but changes the execution schedule. A naive implementation materializes large score/probability matrices in high-bandwidth GPU memory. FlashAttention tiles Q/K/V and performs online softmax accumulation using faster on-chip memory, reducing expensive memory traffic.

**Key idea.** It is an IO-aware exact algorithm, not a sparse or low-rank approximation.

**Interview trap.** FlashAttention does not change the mathematical dense pairwise interaction from quadratic to linear complexity. It makes the same computation much more hardware efficient.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q8. What is the role of the FFN compared with attention?

Attention mixes information **across token positions**. The Feed-Forward Network (FFN) transforms feature channels **within each token**, using the same MLP at every position.

A useful mental model is:

```text
attention = token mixing
FFN       = feature/channel mixing
```

The FFN often contains a large fraction of Transformer parameters because it expands hidden width before projecting back. Modern models often use gated variants such as SwiGLU.

**Design idea.** Attention decides where information should come from; the FFN transforms the gathered information into new features.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

