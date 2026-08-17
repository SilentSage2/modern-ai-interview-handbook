# Inference, TensorRT & Deployment — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. What does TensorRT do that eager PyTorch does not?

NVIDIA TensorRT is an inference compiler/runtime. Eager PyTorch must remain flexible for Python execution and many model behaviors; TensorRT can specialize an evaluation graph for target hardware, chosen precision, and expected shape ranges.

Typical optimizations include:
- operator/kernel fusion;
- tactic or kernel selection;
- tensor-layout choices;
- memory planning;
- lower precision;
- constant folding.

**Key idea.** TensorRT is not another training framework. It trades general execution flexibility for highly specialized inference efficiency.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Why does operator fusion improve inference?

Separate operations such as linear → bias → activation may launch separate GPU kernels and write/read intermediate tensors through high-bandwidth memory.

A fused kernel can keep intermediate values closer to registers/shared/on-chip memory and reduce launch overhead.

The mathematical FLOPs may barely change. The speedup often comes from **less memory traffic and fewer kernel launches**.

This is a general systems lesson: wall-clock performance depends on data movement and scheduling, not only the number of arithmetic operations.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. PTQ versus QAT: what is the difference?

Post-Training Quantization (PTQ) quantizes an already trained model, usually using representative calibration data to estimate ranges/scales. It is inexpensive but may hurt accuracy in sensitive networks.

Quantization-Aware Training (QAT) simulates quantization effects during training so parameters adapt to the approximation. It costs extra training but can recover more accuracy.

**Design decision.** Try PTQ first when the model is robust; use QAT if accuracy degradation is unacceptable.

Always report downstream task metrics as well as tensor-level numerical error.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. Why can weight-only quantization help LLM decode so much?

Autoregressive decode often has low arithmetic intensity: each step processes only a few new tokens while reading very large weight matrices and KV cache.

If weights are stored at fewer bits, fewer bytes must be read from memory for the same logical computation. This can reduce bandwidth pressure and free GPU memory for larger batches or longer contexts.

**Key idea.** The benefit is not merely a smaller model file. In a bandwidth-bound workload, reducing memory traffic directly increases tokens per second.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. What are dynamic shapes and why do they complicate optimization?

Production requests can vary in batch size, image resolution, or sequence length. TensorRT can use optimization profiles specifying minimum, optimal, and maximum shapes.

Static shapes let the compiler specialize aggressively. Broad dynamic ranges require kernels and memory plans that work across many possibilities and may reduce specialization.

A deployment benchmark should therefore state shape profiles and test the relevant min/opt/max cases instead of measuring only one convenient input shape.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. What are paged KV cache and continuous batching, and how are they different?

**Paged KV cache** is a memory-management technique. It stores each sequence's key/value states in fixed-size blocks rather than requiring one large contiguous allocation, reducing fragmentation and supporting dynamic growth.

**Continuous batching** is a scheduling technique. When some sequences finish decoding, their execution slots can be reused by newly arrived requests while others continue.

They solve different problems:
- paging → memory allocation and fragmentation;
- continuous batching → utilization under variable sequence lengths.

Together they are central to high-throughput LLM serving.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q7. Latency versus throughput: how do you benchmark correctly?

Latency is time per request; throughput is work completed per unit time. Larger batches often improve throughput but can increase queueing and per-request latency.

For GPU timing:
- warm up first;
- synchronize around timed regions;
- fix precision and input shape;
- run enough iterations;
- report p50/p95/p99 for online systems.

For LLMs also distinguish:
- Time To First Token (TTFT);
- inter-token latency;
- total generation time;
- tokens per second.

A fair PyTorch-versus-TensorRT comparison must use the same workload and correctness tolerance.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q8. How do you validate correctness after conversion or quantization?

Measure both numerical agreement and task-level behavior.

One useful metric is relative L2 error:

```math
e_{\mathrm{rel}}
=
\frac{
\|y_{\mathrm{optimized}}-y_{\mathrm{baseline}}\|_2
}{
\|y_{\mathrm{baseline}}\|_2+\epsilon
}.
```

Also report max absolute error.

But final correctness should be application-specific: classification agreement, Dice score, reconstruction error, generated-token quality, or another downstream metric.

**Key idea.** Faster wrong output is not optimization. Conversion and quantization must be validated under the same input distribution and shape ranges used in deployment.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

