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


$$
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
$$


For $N$ parameters in FP16:


$$
M_{\rm params}\approx2N\text{ bytes}.
$$


Adam may additionally keep FP32 master weights and two FP32 moments, so optimizer-related memory can dominate.

---

### 2. Activation Memory

For layer activations


$$
A_l\in\mathbb R^{B\times T\times d_l},
$$


memory scales approximately with batch $B$, sequence length $T$, and hidden dimension.

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


$$
\text{memory}\downarrow,
\qquad
\text{compute}\uparrow.
$$


---

### 4. Data Parallelism

Each GPU holds a model replica.

Split batch:


$$
B=B_1+\cdots+B_n.
$$


Each device computes local gradient $g_i$.

Aggregate:


$$
g
=
\frac1n\sum_i g_i.
$$


Then all replicas update identically.

Main communication:
all-reduce gradients.

---

### 5. Tensor Parallelism

Split a large matrix operation across GPUs.

For


$$
Y=XW,
$$


partition $W$ by columns or rows.

Benefit:
individual layer can exceed one GPU memory.

Cost:
communication inside layers.

---

### 6. Pipeline Parallelism

Split layers across stages:


$$
\text{GPU1}:1\!:\!L_1,\quad
\text{GPU2}:L_1\!+\!1\!:\!L_2,\ldots
$$


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


$$
O\left(\frac{N}{n_{\rm GPU}}\right)
$$


for sharded components.

Tradeoff:
extra communication and orchestration complexity.

---

### 8. Arithmetic Intensity


$$
\text{Arithmetic intensity}
=
\frac{\text{FLOPs}}
{\text{bytes moved}}.
$$


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

<!-- DEEP_DIVE_START -->
## Deep dive I: memory accounting for Adam training

Suppose $N$ parameters.

A simplified mixed-precision setup may store:

- BF16 parameter: $2N$ bytes;
- BF16/FP16 gradient: $2N$;
- FP32 master weight: $4N$;
- Adam first moment: $4N$;
- Adam second moment: $4N$.

Already:


$$
16N\text{ bytes}
$$


before activation memory and temporary buffers.

For $N=7$ billion:


$$
16\times7\times10^9
\approx112\text{ GB}.
$$


This rough calculation immediately explains why large-model training needs sharding/parallelism.

Exact memory varies by framework and optimizer implementation, but the accounting logic is what matters.

---

## Deep dive II: activation scaling

Transformer hidden activations scale roughly with:


$$
B\times T\times d\times L.
$$


Attention may add $T^2$-related intermediates depending on kernel implementation.

Thus doubling sequence length can increase memory dramatically even when model parameters are unchanged.

This is why long-context training is difficult even if the model “fits” at shorter context.

---

## Deep dive III: data-parallel communication

Each GPU computes local gradient $g_i$.

All-reduce conceptually produces:


$$
g=\sum_i g_i
$$


and distributes the result to all workers.

Ring all-reduce avoids sending all data through one central parameter server but communication still becomes a scaling bottleneck.

Compute/communication overlap is important for efficient DDP.

---

## Deep dive IV: tensor parallel example

Linear layer:


$$
Y=XW,
\qquad
W=[W_1,W_2].
$$


Column partition:


$$
Y_1=XW_1,\qquad Y_2=XW_2.
$$


Each GPU computes part of output features. Later operations may require gathering or communicating these partitions.

Tensor parallelism solves **single-layer/model-width memory**, not just batch throughput.

---

## Deep dive V: FSDP intuition

Ordinary data parallel:
every GPU stores all parameters.

FSDP:
parameters are sharded at rest. Around a layer's computation, required shards are gathered; after use, memory can be released/resharded.

Conceptually:

```text
sharded params
→ all-gather layer params
→ forward/backward
→ reduce-scatter gradients
→ keep local shard
```

This trades communication for memory.

---

## Deep dive VI: ZeRO stages conceptually

A useful conceptual progression:

- Stage 1: shard optimizer state.
- Stage 2: shard optimizer state + gradients.
- Stage 3: shard optimizer state + gradients + parameters.

The deeper the sharding, the lower per-device model-state memory and the greater orchestration/communication complexity.

---

## Deep dive VII: roofline intuition

Peak performance is bounded by:


$$
\mathrm{Performance}
\le
\min(
\mathrm{PeakFLOPs},
\mathrm{Bandwidth}\times\mathrm{ArithmeticIntensity}
).
$$


If arithmetic intensity is low, adding more theoretical FLOPs does not help: memory bandwidth limits the workload.

This equation is a powerful way to reason about:
- decode;
- elementwise kernels;
- small batches;
- large GEMMs.

---

## Deep dive VIII: why batching helps GEMM

Batch 1 linear layer resembles:


$$
[1,d]\times[d,k].
$$


Large batch:


$$
[B,d]\times[d,k].
$$


Larger matrix multiplication improves hardware occupancy/data reuse and amortizes launch overhead.

But online serving must trade throughput against latency.

---

## Deep dive IX: gradient accumulation

If GPU fits microbatch $b$ but desired effective batch is $B=Kb$:

```python
optimizer.zero_grad()
for _ in range(K):
    loss = model(microbatch) / K
    loss.backward()
optimizer.step()
```

This reproduces a larger effective batch approximately without storing all examples simultaneously.

It does **not** improve data-parallel compute efficiency in the same way as actually increasing instantaneous batch size.

---

## Deep dive X: profiling workflow

When code is slow, do not immediately rewrite kernels.

Ask:

1. Is GPU utilization low?
2. Is CPU/data loading starving GPU?
3. Is H2D/D2H copy dominating?
4. Are kernels too small?
5. Is memory bandwidth saturated?
6. Is multi-GPU communication dominating?
7. Is precision preventing tensor-core use?

Profile first, optimize second.

This is especially relevant when translating numerical pipelines from CPU/MATLAB to GPU: a function that is already very fast on CPU may not justify transfer and launch overhead on GPU.
<!-- DEEP_DIVE_END -->

<!-- SECOND_DEEP_DIVE_START -->
## Parallelism decision framework

Ask what does not fit or what is too slow.

### Model fits, batch throughput too low
Use data parallelism.

### Individual layer/model does not fit
Use tensor/model parallelism.

### Model depth can be staged
Pipeline parallelism.

### Model states dominate memory
FSDP/ZeRO.

Large training systems often combine multiple dimensions:


$$
\text{DP}\times\text{TP}\times\text{PP}.
$$


This is called multidimensional parallelism.

---

## Communication volume intuition

Distributed speedup is limited by:


$$
T_{\rm step}
\approx
T_{\rm compute}
+
T_{\rm communication}
-
T_{\rm overlap}.
$$


Adding GPUs helps only if added compute parallelism outweighs communication/synchronization.

Strong scaling eventually saturates.

---

## Gradient synchronization timing

DDP can launch all-reduce for a gradient bucket while backward computes earlier layers.

This overlaps communication with computation.

If bucket sizing or network is poor, GPU may stall waiting for collectives.

Profilers can show:
- compute kernels;
- NCCL collectives;
- gaps.

---

## Data-loading bottleneck

GPU can be idle even though model is efficient.

Pipeline:

```text
storage
→ CPU decode/augment
→ pinned host memory
→ H2D copy
→ GPU
```

Optimizations:
- multiple workers;
- prefetch;
- pinned memory;
- asynchronous H2D;
- caching;
- move augmentation to GPU when appropriate.

Always check input pipeline before blaming model kernels.

---

## Host-to-device transfer

PCIe/NVLink bandwidth is much lower than on-GPU memory bandwidth.

Repeated CPU↔GPU transfers inside a loop can destroy acceleration.

Best pattern:
- move batch once;
- keep intermediate arrays on GPU;
- copy back only required outputs.

This principle applies equally to custom CuPy numerical pipelines and neural networks.

---

## Kernel launch overhead

A tiny GPU kernel may take microseconds of launch/scheduling overhead comparable to useful compute.

Thousands of small operations can be slower than fewer fused operations.

This motivates:
- vectorization;
- fusion;
- compiled graphs;
- larger batches.

---

## Complex precision tradeoff

For numerical scientific workloads:
- FP64/complex128 may be needed for consistency;
- FP32/complex64 may be much faster on many GPUs.

A correct benchmark should compare:
- runtime;
- numerical error;
- downstream result.

The fastest dtype is useless if error violates scientific tolerance.

This is the same principle behind AI quantization benchmarks.

---

## Multi-GPU scaling efficiency

If one GPU throughput is $P_1$, $n$-GPU throughput $P_n$:


$$
\eta_n
=
\frac{P_n}{nP_1}.
$$


Perfect scaling:


$$
\eta_n=1.
$$


Real scaling drops due to:
- communication;
- load imbalance;
- synchronization;
- input bottlenecks.

Always report scaling efficiency rather than only “4 GPUs are faster.”

---

## Hardware-aware reasoning

Important GPU resources:
- tensor/compute cores;
- HBM capacity;
- HBM bandwidth;
- shared memory/cache;
- interconnect bandwidth.

Different workloads stress different resources.

This is why a theoretical FLOP count alone cannot predict wall-clock speed.

---

## Mini-project: profile a Transformer training step

Measure:
1. baseline eager;
2. mixed precision;
3. larger batch;
4. gradient checkpointing;
5. multi-GPU DDP.

Record:
- max memory;
- samples/sec;
- step time;
- GPU utilization.

Explain every change in terms of:
- memory;
- compute;
- communication;
- arithmetic intensity.
<!-- SECOND_DEEP_DIVE_END -->

<!-- THIRD_DEEP_DIVE_START -->
## Scaling metrics

### Throughput scaling


$$
S_n
=
\frac{P_n}{P_1}.
$$


### Parallel efficiency


$$
E_n
=
\frac{S_n}{n}.
$$


If 4 GPUs produce 3.2× throughput:


$$
E_4=\frac{3.2}{4}=0.8.
$$


80% parallel efficiency is more informative than simply reporting “3.2× faster.”

---

## Strong vs weak scaling

### Strong scaling
Fixed total problem size, more GPUs.

Goal:
reduce time to solution.

### Weak scaling
Problem size grows with number of GPUs.

Goal:
maintain time while processing proportionally more work.

Training-system benchmarks should state which kind of scaling is measured.

---

## Communication topology

GPU connectivity matters.

Possible links:
- PCIe;
- NVLink/NVSwitch;
- inter-node network.

Tensor-parallel communication across slow inter-node links can be far more expensive than within a high-bandwidth node.

Thus parallelism mapping should consider physical topology.

---

## Why sequence parallelism appears

Some activations scale with sequence length $T$.

Sequence parallelism shards selected token/sequence-dimension activations across devices, reducing replicated activation memory.

It is often combined with tensor parallelism in large Transformer systems.

You do not need implementation details for every framework, but recognize the motivation: distribute activation/state dimensions that remain replicated under other schemes.

---

## Optimizer state sharding vs activation checkpointing

They attack different memory terms.

Optimizer sharding:


$$
M_{\rm optimizer}\downarrow.
$$


Checkpointing:


$$
M_{\rm activations}\downarrow.
$$


Quantization:


$$
M_{\rm weights}\downarrow
$$


(mainly inference or frozen-base training, depending on method).

A good memory strategy starts by identifying which term dominates.

---

## Debugging an OOM

Ask in order:

1. Did batch/sequence/input resolution change?
2. Is graph retaining tensors accidentally?
3. Are gradients being accumulated without clearing?
4. Is validation accidentally tracking gradients?
5. Are optimizer states larger than expected?
6. Is fragmentation/workspace causing peak?
7. Can checkpointing/sharding/precision help?

This is practical ML-engineering knowledge interviewers value.
<!-- THIRD_DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- For any memory question, write the memory equation first.
- For any speed question, ask whether the workload is compute-bound, bandwidth-bound, or communication-bound.
- Distinguish scaling model capacity from scaling batch throughput.

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
