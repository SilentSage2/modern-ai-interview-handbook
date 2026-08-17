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

Let a sequence contain $N$ tokens with hidden dimension $d$:


$$
X\in\mathbb R^{N\times d}.
$$


A Transformer layer should allow each token to aggregate information from other tokens.

The central operation is attention.

---

### 2. Query, Key, Value

Project each token into three spaces:


$$
Q=XW_Q,\qquad
K=XW_K,\qquad
V=XW_V.
$$


Shapes:


$$
W_Q,W_K\in\mathbb R^{d\times d_k},
\qquad
W_V\in\mathbb R^{d\times d_v}.
$$


Thus


$$
Q,K\in\mathbb R^{N\times d_k},
\qquad
V\in\mathbb R^{N\times d_v}.
$$


For token $i$, query $q_i$ is compared against all keys $k_j$:


$$
s_{ij}=q_i^\top k_j.
$$


This produces


$$
S=QK^\top\in\mathbb R^{N\times N}.
$$


Interpretation:
- Query: what information does this token seek?
- Key: what information does another token advertise?
- Value: what content should be transferred if attended to?

---

### 3. Why Scale by $\sqrt{d_k}$?

Assume elements of $q$ and $k$ are independent with


$$
\mathbb E[q_l]=\mathbb E[k_l]=0,\qquad
\mathrm{Var}(q_l)=\mathrm{Var}(k_l)=1.
$$


The dot product is


$$
q^\top k=\sum_{l=1}^{d_k}q_lk_l.
$$


Each product has roughly unit variance, so


$$
\mathrm{Var}(q^\top k)\approx d_k.
$$


Therefore its standard deviation grows as


$$
\sqrt{d_k}.
$$


Large logits push softmax into saturation, causing near one-hot weights and poor gradients.

Scale:


$$
\tilde S=\frac{QK^\top}{\sqrt{d_k}}.
$$


Then the logit variance remains approximately $O(1)$.

---

### 4. Softmax Attention

Row-wise softmax gives


$$
A_{ij}
=
\frac{\exp(\tilde S_{ij})}
{\sum_{m=1}^N\exp(\tilde S_{im})}.
$$


Each row sums to 1.

Output:


$$
Y=AV.
$$


For token $i$:


$$
y_i=\sum_{j=1}^N A_{ij}v_j.
$$


So attention is a learned, input-dependent weighted average of value vectors.

---

### 5. Causal Masking

Autoregressive language modeling requires token $i$ to use only tokens $j\le i$.

Define


$$
M_{ij}
=
\begin{cases}
0,&j\le i\\
-\infty,&j>i.
\end{cases}
$$


Then


$$
A=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right).
$$


Since $e^{-\infty}=0$, future positions receive zero probability.

---

### 6. Multi-Head Attention

A single attention map is restrictive. Use $h$ independent projections:


$$
Q_i=XW_Q^{(i)},
\quad
K_i=XW_K^{(i)},
\quad
V_i=XW_V^{(i)}.
$$


Then


$$
H_i=
\mathrm{Attention}(Q_i,K_i,V_i).
$$


Concatenate:


$$
H=\mathrm{Concat}(H_1,\ldots,H_h).
$$


Project:


$$
Y=HW_O.
$$


If model dimension is $d$, a common design is


$$
d_k=d_v=\frac dh.
$$


This keeps total attention width approximately fixed while allowing different representation subspaces.

---

### 7. Why Attention is $O(N^2)$

The score matrix


$$
QK^\top
$$


has $N^2$ entries.

Compute complexity:


$$
O(N^2d_k).
$$


Memory for attention weights:


$$
O(N^2).
$$


This is the central bottleneck for very long contexts.

---

### 8. Transformer Block

A simplified pre-norm block:


$$
H_1=X+\mathrm{MHA}(\mathrm{LN}(X)),
$$


$$
H_2=H_1+\mathrm{FFN}(\mathrm{LN}(H_1)).
$$


FFN is token-wise:


$$
\mathrm{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2.
$$


Attention mixes information **across tokens**.

FFN mixes information **across channels/features within each token**.

This distinction is very useful in interviews.

---

### 9. LayerNorm

For hidden vector $x\in\mathbb R^d$,


$$
\mu=\frac1d\sum_i x_i,
$$


$$
\sigma^2=\frac1d\sum_i(x_i-\mu)^2.
$$


Normalize:


$$
\hat x_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}.
$$


Affine transform:


$$
y_i=\gamma_i\hat x_i+\beta_i.
$$


LayerNorm does not depend on other samples in the batch.

---

### 10. RMSNorm

RMSNorm removes mean subtraction:


$$
\mathrm{RMS}(x)
=
\sqrt{\frac1d\sum_i x_i^2+\epsilon},
$$


$$
y_i=\gamma_i\frac{x_i}{\mathrm{RMS}(x)}.
$$


It is computationally simpler and widely used in modern LLMs.

---

### 11. Positional Information

Self-attention alone is permutation equivariant.

Without positions, reordering tokens reorders outputs but does not tell the model which order is correct.

### Sinusoidal encoding


$$
PE_{(pos,2i)}
=
\sin\left(
pos/10000^{2i/d}
\right),
$$


$$
PE_{(pos,2i+1)}
=
\cos\left(
pos/10000^{2i/d}
\right).
$$


---

### 12. RoPE Intuition

RoPE rotates query/key vector pairs by position-dependent angles.

For 2D pair:


$$
R(\theta_m)
=
\begin{bmatrix}
\cos\theta_m&-\sin\theta_m\\
\sin\theta_m&\cos\theta_m
\end{bmatrix}.
$$


At positions $m,n$:


$$
q_m=R(\theta_m)q,
\qquad
k_n=R(\theta_n)k.
$$


Their inner product:


$$
q_m^\top k_n
=
q^\top R(\theta_m)^\top R(\theta_n)k
=
q^\top R(\theta_n-\theta_m)k.
$$


Therefore the interaction naturally depends on **relative position $n-m$**.

That is the key derivation to remember.

---

### 13. KV Cache

During autoregressive generation, previous K/V vectors do not change.

At step $t$, instead of recomputing:


$$
K_{1:t},V_{1:t},
$$


store


$$
K_{1:t-1},V_{1:t-1}
$$


and compute only


$$
k_t,v_t.
$$


Then query $q_t$ attends to the cached history.

#### Memory
For $L$ layers, $H_{kv}$ KV heads, head dimension $d_h$, sequence $T$:


$$
\mathrm{KV\ elements}
\approx
2LTH_{kv}d_h.
$$


Multiply by bytes per element and batch size.

This is why long-context LLM inference is often memory-bandwidth and memory-capacity limited.

---

### 14. Prefill vs Decode

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

### 15. MQA and GQA

Standard MHA has separate K/V per attention head.

#### Multi-Query Attention
Many Q heads share one K/V head.

#### Grouped-Query Attention
Groups of Q heads share K/V heads.

Benefit:


$$
\text{KV cache memory}\downarrow
$$


while retaining more expressiveness than a single shared KV head.

---

### 16. FlashAttention

FlashAttention computes exact attention but changes the execution strategy.

Naive implementation materializes large $N\times N$ intermediates in high-bandwidth memory.

FlashAttention:
- tiles Q/K/V;
- keeps blocks in fast on-chip memory;
- recomputes/selectively accumulates softmax statistics;
- reduces HBM traffic.

Key interview answer:

> FlashAttention is mainly an IO-aware algorithm, not an approximation to attention.

---

### 17. Common Interview Comparisons

#### Attention vs convolution
- attention: dynamic weights, global interaction;
- convolution: fixed learned kernels, local prior, translation equivariance.

#### Encoder vs decoder
- encoder attention is typically bidirectional;
- decoder-only LLM uses causal self-attention.

#### Pre-norm vs post-norm
Pre-norm puts normalization before sublayer and generally improves optimization stability for deep Transformers.

<!-- DEEP_DIVE_START -->
## Deep dive I: a complete tensor-shape walkthrough

Suppose:


$$
X\in\mathbb R^{B\times T\times d_{\rm model}},
$$


with


$$
B=2,\quad T=128,\quad d_{\rm model}=768,
$$


and $h=12$ heads.

Then


$$
d_h=\frac{768}{12}=64.
$$


After linear projections:


$$
Q,K,V\in\mathbb R^{2\times128\times768}.
$$


Reshape into heads:


$$
Q,K,V
\rightarrow
[2,12,128,64].
$$


Attention scores:


$$
QK^\top
\rightarrow
[2,12,128,128].
$$


After softmax, multiply values:


$$
[2,12,128,128]
\times
[2,12,128,64]
\rightarrow
[2,12,128,64].
$$


Merge heads:


$$
[2,12,128,64]
\rightarrow
[2,128,768].
$$


This shape flow should become automatic.

### Minimal PyTorch attention

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False):
        B, T, D = x.shape

        qkv = self.qkv(x)                  # [B,T,3D]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k = k.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        v = v.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        # [B,H,T,Dh]

        score = q @ k.transpose(-2,-1) / math.sqrt(self.d_head)

        if causal:
            mask = torch.triu(
                torch.ones(T,T,device=x.device,dtype=torch.bool),
                diagonal=1
            )
            score = score.masked_fill(mask, float("-inf"))

        attn = score.softmax(dim=-1)
        y = attn @ v                      # [B,H,T,Dh]

        y = y.transpose(1,2).contiguous().view(B,T,D)
        return self.out(y)
```

Important implementation details:
- `transpose` can make a tensor non-contiguous;
- `contiguous()` before `view()` avoids layout assumptions;
- masking occurs **before** softmax;
- scaling uses per-head dimension, not total model dimension.

---

## Deep dive II: why attention can represent dynamic interactions

Convolution uses learned weights that are fixed after training:


$$
y_i=\sum_{\Delta}w_\Delta x_{i+\Delta}.
$$


Attention weights depend on the current input:


$$
A_{ij}(X)
=
\mathrm{softmax}_j
\left(
q_i(X)^\top k_j(X)
\right).
$$


Therefore two different sequences can induce different connectivity patterns even with identical parameters.

This is why attention is often described as **content-adaptive routing**.

---

## Deep dive III: Transformer parameter counting

For one dense attention block with model width $d$:

Q, K, V projections:


$$
3d^2.
$$


Output projection:


$$
d^2.
$$


Attention total:


$$
4d^2.
$$


If FFN hidden dimension is $d_{\rm ff}=4d$,


$$
W_1:d\rightarrow4d
\quad\Rightarrow\quad
4d^2,
$$


$$
W_2:4d\rightarrow d
\quad\Rightarrow\quad
4d^2.
$$


FFN total:


$$
8d^2.
$$


So a classic Transformer block has roughly


$$
12d^2
$$


large matrix parameters, ignoring bias/norm.

**Important systems implication:** most parameters are often in the MLP, not the attention score matrix.

---

## Deep dive IV: causal training vs autoregressive inference

Training sequence:


$$
[x_1,x_2,\ldots,x_T].
$$


A causal mask allows all next-token losses to be computed in parallel:


$$
x_1\to x_2,\quad
x_{1:2}\to x_3,\quad
\ldots
$$


within one forward pass.

Inference is different because token $x_{T+1}$ does not exist until it is sampled. Therefore generation remains sequential:


$$
x_{T+1}
\rightarrow
x_{T+2}
\rightarrow
x_{T+3}.
$$


This resolves a common confusion:

> Transformer training is highly parallel, but autoregressive decoding is inherently sequential across generated tokens.

---

## Deep dive V: KV-cache memory worked example

Suppose:
- $L=32$ layers;
- $H_{kv}=8$;
- $d_h=128$;
- $T=8192$;
- batch $B=1$;
- BF16 = 2 bytes.

KV elements:


$$
2 \times L\times H_{kv}\times T\times d_h.
$$


Approximate bytes:


$$
2(32)(8)(8192)(128)(2)
\approx 1.07\times10^9\text{ bytes}
$$


or about $1$ GB.

This is only KV cache for one sequence. Longer context, larger batch, or more KV heads increase memory linearly.

This example explains why GQA/MQA and paged KV management matter operationally.

---

## Deep dive VI: RoPE worked example

Take one 2D query component:


$$
q=
\begin{bmatrix}
q_1\\q_2
\end{bmatrix}.
$$


At position $m$:


$$
q_m=
\begin{bmatrix}
\cos m\theta&-\sin m\theta\\
\sin m\theta&\cos m\theta
\end{bmatrix}q.
$$


Similarly for $k_n$.

Because rotations compose:


$$
R(m\theta)^\top R(n\theta)
=
R((n-m)\theta),
$$


the attention dot product contains relative offset $n-m$.

In higher dimension, different coordinate pairs rotate at different frequencies so the representation covers multiple distance scales.

---

## Deep dive VII: why FFN expansion matters

Transformer MLP commonly expands width:


$$
d\rightarrow d_{\rm ff}\rightarrow d,
\qquad
d_{\rm ff}>d.
$$


The expansion creates a larger nonlinear feature space.

Modern gated MLP variants use forms such as


$$
\mathrm{SwiGLU}(x)
=
(\mathrm{Swish}(xW_1)\odot xW_2)W_3.
$$


The gate allows multiplicative feature interactions rather than only elementwise activation after one projection.

---

## Deep dive VIII: attention variants you should recognize

### MHA
Each query head has its own K/V head.

### MQA
Many Q heads share one K/V pair.

### GQA
Several Q heads share each K/V head.

Why reduce KV heads?
Because decode cache scales with $H_{kv}$, while query heads do not need to be cached.

### Sliding-window attention
Token attends only nearby positions:


$$
|i-j|\le w.
$$


Complexity becomes closer to


$$
O(Twd)
$$


instead of $O(T^2d)$, but arbitrary long-range recall is restricted.

---

## Deep dive IX: common Transformer misconceptions

**Misconception 1:** Attention is the whole Transformer.  
No. Residual paths, normalization, MLPs, embeddings, positional mechanisms, and training objective are equally central.

**Misconception 2:** Attention is always interpretable.  
Attention weights show one routing mechanism but do not uniquely explain the model's computation.

**Misconception 3:** FlashAttention makes attention linear-time.  
No. It makes exact attention substantially more IO-efficient; the dense mathematical interaction is still quadratic in sequence length.

**Misconception 4:** KV cache reduces memory.  
It reduces repeated compute but **increases stored inference state**.

---

## Worked whiteboard exercise

Given


$$
B=4,\quad T=1024,\quad d=1024,\quad h=16,
$$


answer:

1. What is $d_h$?
2. What shape is one attention score tensor?
3. How many score elements are created across the batch?
4. If context doubles, what happens to dense attention score memory?
5. Which dimensions would GQA change?

Answers:


$$
d_h=64,
$$


scores:


$$
[4,16,1024,1024].
$$


Score elements:


$$
4\times16\times1024^2.
$$


Doubling $T$ approximately quadruples score memory. GQA reduces K/V head count, not the number of Q heads.
<!-- DEEP_DIVE_END -->

<!-- SECOND_DEEP_DIVE_START -->
## Architecture case study: a decoder-only Transformer

A modern decoder-only language model can be summarized as

```text
token IDs
→ token embedding
→ [Transformer block] × L
→ final norm
→ vocabulary projection
→ logits
```

A pre-norm block often looks like

```text
x ───────────────┐
│                │
norm             │
↓                │
causal attention │
↓                │
+ ←──────────────┘
│
├────────────────┐
│                │
norm             │
↓                │
gated MLP        │
↓                │
+ ←──────────────┘
```

The model dimension $d$ stays constant across residual blocks, which is required for the addition:


$$
x+F(x).
$$


Attention and MLP may internally expand/project but return to $d$.

### Why the residual stream is useful conceptually

Think of the residual state as a shared communication channel. Each layer reads the current representation and writes a correction:


$$
x_{l+1}=x_l+\Delta x_l.
$$


This perspective helps explain why many interpretability analyses talk about information being written to or read from a residual stream.

---

## Cross-attention

Self-attention uses the same source for Q/K/V.

Cross-attention uses:


$$
Q=X_{\rm query}W_Q,
$$


$$
K=X_{\rm context}W_K,
\qquad
V=X_{\rm context}W_V.
$$


Shapes:


$$
X_q\in\mathbb R^{B\times T_q\times d},
\qquad
X_c\in\mathbb R^{B\times T_c\times d}.
$$


Score matrix:


$$
QK^\top
\in
\mathbb R^{B\times h\times T_q\times T_c}.
$$


This operation is central to:
- encoder-decoder Transformers;
- VLMs;
- multimodal fusion;
- latent resamplers.

---

## Why causal masking does not prevent parallel training

For a sequence of length $T$, all queries are present simultaneously. The mask merely zeros forbidden future edges.

Therefore we compute the whole triangular dependency graph in one batched matrix operation.

This differs from RNN training, where hidden state recurrence


$$
h_t=f(h_{t-1},x_t)
$$


creates a sequential dependency through $t$.

---

## Attention memory: training vs inference

### Training
We need activations for backward. Without memory-efficient kernels, attention intermediates can scale with


$$
B h T^2.
$$


### Autoregressive inference
For each new token, attention weights need not be stored across future steps, but K/V states are cached:


$$
O(BLT H_{kv}d_h).
$$


So:
- training long sequences is often hurt by activation memory;
- decoding long sequences is often hurt by KV-cache memory and bandwidth.

---

## Why checkpointing attention saves memory

During backward, gradients need forward intermediates.

Checkpointing discards selected intermediates and recomputes forward operations when backward reaches them.

If $f(x)$ is checkpointed:

```text
forward: keep x, discard internal activations
backward: recompute f(x), then differentiate
```

Tradeoff:


$$
\text{activation memory}\downarrow,\qquad
\text{FLOPs}\uparrow.
$$


---

## Numerical stability of softmax

Naive:


$$
\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$


can overflow if $z_i$ is large.

Subtract max:


$$
\mathrm{softmax}(z_i)
=
\frac{e^{z_i-m}}
{\sum_j e^{z_j-m}},
\qquad
m=\max_j z_j.
$$


This is mathematically identical because the common factor $e^{-m}$ cancels.

Attention implementations rely on this stable form.

---

## Why padding masks differ from causal masks

Causal mask encodes **temporal legality**:


$$
j>i \Rightarrow \text{blocked}.
$$


Padding mask encodes **invalid data positions**:

```text
real real real PAD PAD
```

A batch may need both.

For variable sequence length, attention score can combine:


$$
S + M_{\rm causal}+M_{\rm padding}.
$$


---

## Encoder-only vs decoder-only vs encoder-decoder

### Encoder-only
Bidirectional self-attention.

Good for representation tasks:
- classification;
- retrieval;
- token labeling.

### Decoder-only
Causal self-attention.

Good for open-ended autoregressive generation.

### Encoder-decoder
Encoder builds bidirectional source representation; decoder uses causal self-attention plus cross-attention.

Good for conditional sequence generation such as translation.

This taxonomy is more fundamental than memorizing model names.

---

## Implementation exercise: build one Transformer block

```python
class Block(nn.Module):
    def __init__(self, d_model, n_heads, mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mult*d_model),
            nn.GELU(),
            nn.Linear(mult*d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), causal=True)
        x = x + self.mlp(self.norm2(x))
        return x
```

Then test:
1. output shape equals input shape;
2. future tokens do not affect earlier outputs under causal mask;
3. gradients reach embedding parameters;
4. compare with `torch.nn.functional.scaled_dot_product_attention`.

---

## Debugging checklist for attention code

If output is wrong, check:

1. Did you transpose head and sequence axes correctly?
2. Is the key transposed on the last two axes?
3. Is softmax over the **key** dimension?
4. Is causal mask orientation correct?
5. Is scaling $\sqrt{d_h}$, not $\sqrt d$?
6. Are padding positions masked?
7. Did a non-contiguous tensor get reshaped incorrectly?
8. Are mixed-precision logits overflowing?

---

## Design question: when is global attention unnecessary?

For local spatial or streaming tasks, full all-to-all attention may spend compute on irrelevant pairs.

Alternatives:
- local windows;
- sparse patterns;
- hierarchical pooling;
- recurrence/state-space approaches;
- periodic global tokens.

The correct answer depends on whether long-range information is important and how often it is needed.

---

## Mini-project: tiny language model from scratch

Train a small decoder-only Transformer on a small text corpus.

Minimum goals:
- tokenizer;
- positional representation;
- causal attention;
- MLP;
- residual/norm;
- next-token cross entropy;
- sampling;
- KV cache implementation.

A good README for the project should report:
- parameter count;
- sequence length;
- training loss/perplexity;
- samples at multiple temperatures;
- tokens/sec;
- KV-cache speed comparison.

This project turns Transformer knowledge from “read about it” into implementation experience.
<!-- SECOND_DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Track shapes of Q, K, V and the attention matrix.
- Separate token mixing (attention) from channel mixing (FFN).
- For systems questions, connect long context to O(N²) attention during prefill and KV-cache growth during decode.
- Be able to implement causal attention from scratch in PyTorch without relying on a high-level Transformer class.

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
