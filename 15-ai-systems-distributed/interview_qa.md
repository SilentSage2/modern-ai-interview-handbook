# GPU, Distributed Training & AI Systems — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. How do you estimate training memory for a large model?

Break memory into terms:

```math
M
=
M_{\mathrm{params}}
+
M_{\mathrm{grads}}
+
M_{\mathrm{optimizer}}
+
M_{\mathrm{activations}}
+
M_{\mathrm{temporary}}.
```

For Adam-style training, optimizer state can dominate because each trainable parameter may have two FP32 moment buffers and possibly additional copies. Activations scale with batch size, sequence length/resolution, hidden width, and depth.

**Key idea.** Identify the dominant memory term before choosing an optimization:
- checkpointing reduces activations;
- FSDP/ZeRO reduce replicated model state;
- PEFT reduces trainable gradient/optimizer state;
- quantization reduces weight storage;
- smaller batch/sequence reduces activations.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Data parallelism, tensor parallelism, and pipeline parallelism: how do they differ?

**Data Parallelism (DP):** each GPU has a model replica and processes different examples; gradients are synchronized.

**Tensor Parallelism (TP):** split individual layer matrices/operations across GPUs. Use this when model/layer width does not fit one device or when model-level compute must be parallelized.

**Pipeline Parallelism (PP):** place different layer groups on different devices and stream microbatches through stages.

**Design principle.** They solve different bottlenecks. Large systems often combine DP × TP × PP.

**Cost.** Every form introduces communication or synchronization, so adding GPUs can eventually reduce parallel efficiency.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. What do FSDP and ZeRO actually save?

Ordinary data parallelism replicates parameters, gradients, and optimizer states on every worker.

Fully Sharded Data Parallel (FSDP) and Zero Redundancy Optimizer (ZeRO) strategies shard some or all of those states across workers. Workers gather the pieces needed for computation and reshard or reduce them afterward.

**Benefit:** much lower per-device memory.

**Cost:** more communication, orchestration, and sensitivity to interconnect topology.

These are primarily memory-scalability methods, but because they change communication schedules they also affect throughput.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. What is arithmetic intensity and why is it useful?

Arithmetic intensity is FLOPs performed per byte moved:

```math
I
=
\frac{\text{FLOPs}}
{\text{bytes transferred}}.
```

High-intensity workloads can become compute-bound; low-intensity workloads are often memory-bandwidth bound.

Large matrix multiplications reuse data heavily and often have high intensity. LLM decode at small batch repeatedly reads large weights and KV cache for very few new tokens, so it can be bandwidth-limited.

**Why this matters.** It explains why quantization can speed decode even when the GPU has unused theoretical FLOPs: fewer bytes must be moved.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. Why can larger batches improve GPU throughput?

GPUs are efficient when many parallel operations are available. Larger batches create larger matrix operations, improve occupancy/data reuse, and amortize kernel-launch overhead.

But larger batches:
- consume more memory;
- can increase online request latency;
- reduce gradient noise during training;
- may require learning-rate retuning.

Therefore training batch size and inference-serving batch size are different optimization problems. Throughput should always be balanced against memory and latency constraints.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. BF16 versus FP16: why is BF16 common in large-model training?

Both use 16 bits, but BF16 keeps an FP32-like exponent range with fewer mantissa bits. FP16 has more mantissa precision but a much narrower exponent range.

Large models produce values across a wide dynamic range, so BF16 reduces overflow and underflow risk and often avoids explicit loss scaling.

**Systems benefit.** Both formats reduce memory traffic and can use accelerator tensor cores relative to FP32.

**Caveat.** Sensitive reductions and optimizer accumulators may still use higher precision. Mixed precision means choosing precision per operation, not forcing everything to BF16.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q7. Why can multi-GPU scaling saturate?

If one GPU throughput is `P1` and n-GPU throughput is `Pn`, parallel efficiency is:

```math
E_n
=
\frac{P_n}{nP_1}.
```

Efficiency drops as communication, synchronization, load imbalance, and input bottlenecks become significant relative to useful compute.

Tensor parallelism may communicate inside layers; data parallelism synchronizes gradients; pipeline parallelism has bubbles.

A strong benchmark reports both speedup and parallel efficiency and profiles communication rather than only stating that more GPUs were used.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q8. How do you debug a slow GPU workload?

Profile before rewriting kernels.

Ask:
1. Is the GPU idle waiting for data?
2. Are CPU→GPU or GPU→CPU copies frequent?
3. Are kernels too small?
4. Is HBM bandwidth saturated?
5. Is compute saturated?
6. Is multi-GPU communication dominating?
7. Is precision preventing fast tensor-core kernels?
8. Is synchronization serializing work?

This methodology applies equally to PyTorch and custom CuPy/scientific pipelines. A function already fast on CPU may not justify GPU transfer and launch overhead.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

