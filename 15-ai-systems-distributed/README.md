# GPU, Distributed Training & AI Systems

**Status:** Priority

## Why this matters

Modern AI interviews increasingly test whether candidates understand where GPU memory, communication, and throughput are actually spent.

## Learning objectives

- Break training memory into parameters, gradients, optimizer states, and activations.
- Compare data/tensor/pipeline parallelism and FSDP/ZeRO.
- Reason about arithmetic intensity, batching, mixed precision, and checkpointing.

## Chapter map

- GPU memory hierarchy and bandwidth
- Training memory: parameters, gradients, optimizer states, activations
- Mixed precision: FP32, FP16, BF16, FP8 concepts
- Data parallelism
- Tensor parallelism
- Pipeline parallelism
- DDP, FSDP and ZeRO concepts
- NCCL and collective communication
- Gradient accumulation and checkpointing
- Profiling compute vs memory bottlenecks
- Throughput vs latency


---

## Core concepts and theory

### 1. Training Memory Decomposition

Rough GPU memory:

\[
M
=
M_{\rm params}
+
M_{\rm grads}
+
M_{\rm opt}
+
M_{\rm activations}
+
M_{\rm temp}.
\]

For \(N\) parameters in FP16:

\[
M_{\rm params}\approx2N\text{ bytes}.
\]

Adam may additionally keep FP32 master weights and two FP32 moments, so optimizer-related memory can dominate.

---

### 2. Activation Memory

For layer activations

\[
A_l\in\mathbb R^{B\times T\times d_l},
\]

memory scales approximately with batch \(B\), sequence length \(T\), and hidden dimension.

This is why:
- longer context;
- larger batch;
- deeper models

rapidly increase training memory.

---

### 3. Gradient Checkpointing

Without checkpointing:
store many activations from forward pass.

With checkpointing:
store selected checkpoints and recompute missing activations during backward.

Tradeoff:

\[
\text{memory}\downarrow,
\qquad
\text{compute}\uparrow.
\]

---

### 4. Data Parallelism

Each GPU holds a model replica.

Split batch:

\[
B=B_1+\cdots+B_n.
\]

Each device computes local gradient \(g_i\).

Aggregate:

\[
g
=
\frac1n\sum_i g_i.
\]

Then all replicas update identically.

Main communication:
all-reduce gradients.

---

### 5. Tensor Parallelism

Split a large matrix operation across GPUs.

For

\[
Y=XW,
\]

partition \(W\) by columns or rows.

Benefit:
individual layer can exceed one GPU memory.

Cost:
communication inside layers.

---

### 6. Pipeline Parallelism

Split layers across stages:

\[
\text{GPU1}:1\!:\!L_1,\quad
\text{GPU2}:L_1\!+\!1\!:\!L_2,\ldots
\]

Microbatches flow through pipeline.

Issue:
pipeline bubbles/idle time.

---

### 7. FSDP / ZeRO

Instead of every GPU storing all model states, shard:
- parameters;
- gradients;
- optimizer states.

Conceptual memory per GPU can approach

\[
O\left(\frac{N}{n_{\rm GPU}}\right)
\]

for sharded components.

Tradeoff:
extra communication and orchestration complexity.

---

### 8. Arithmetic Intensity

\[
\text{Arithmetic intensity}
=
\frac{\text{FLOPs}}
{\text{bytes moved}}.
\]

Low intensity:
memory-bandwidth bound.

High intensity:
compute bound.

Large matrix multiplication is compute efficient because data reuse is high.

Autoregressive decode can become bandwidth limited because each step processes small token batches while reading large weights/KV cache.

---

### 9. Batch Size

Larger batch can:
- improve GPU utilization;
- increase throughput;
- reduce gradient noise.

But:
- increases memory;
- can worsen latency;
- may require learning-rate adjustment;
- too-large batch can hurt generalization/optimization.

---

### 10. Mixed Precision

FP16/BF16 reduce:
- memory;
- bandwidth;
- tensor-core compute cost.

BF16:
- 8-bit exponent like FP32;
- lower mantissa precision.

FP16:
- more mantissa bits;
- narrower dynamic range.

This is why BF16 is often easier for large-model training.

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- For any memory question, write the memory equation first.
- For any speed question, ask whether the workload is compute-bound, bandwidth-bound, or communication-bound.
- Distinguish scaling model capacity from scaling batch throughput.

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
