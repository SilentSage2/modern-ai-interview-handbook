# Inference, TensorRT & Deployment

**Status:** Priority + Hands-on

## Why this matters

Deployment experience requires understanding inference graphs, quantization, batching, KV cache, TensorRT, and performance measurement—not only calling model.forward().

## Learning objectives

- Explain TensorRT-style graph/runtime optimization.
- Understand PTQ/QAT, weight-only quantization, dynamic shapes, and operator fusion.
- Reason about LLM serving, paged KV cache, continuous batching, latency, and throughput.

## Chapter map

- Inference graph optimization
- PyTorch eager vs compiled/deployment engines
- ONNX concepts
- TensorRT engine building
- Operator/kernel fusion
- Precision and quantization: FP16/BF16/FP8/INT8/INT4
- Dynamic shapes and batching
- TensorRT-LLM concepts
- Paged KV cache and continuous/in-flight batching
- Serving latency, throughput and concurrency
- Benchmark methodology and numerical validation


---

## Core concepts and theory

### 1. Latency and Throughput

Latency:

\[
L=\text{time per request}.
\]

Throughput:

\[
Q=\frac{\text{requests or tokens}}{\text{second}}.
\]

Increasing batch size often raises throughput while potentially increasing queueing and per-request latency.

---

### 2. Inference Optimization Pipeline

Conceptually:

\[
\text{trained graph}
\rightarrow
\text{graph simplification}
\rightarrow
\text{kernel/tactic selection}
\rightarrow
\text{precision optimization}
\rightarrow
\text{memory planning}
\rightarrow
\text{engine}.
\]

TensorRT is an optimized inference compiler/runtime for NVIDIA GPUs.

---

### 3. Operator Fusion

Suppose:

\[
y=\mathrm{ReLU}(\mathrm{BN}(\mathrm{Conv}(x))).
\]

Naive execution may:
- launch multiple kernels;
- write/read intermediate tensors to global memory.

Fusion executes combined operations with fewer memory round-trips and kernel launches.

Benefit is often from reduced memory traffic and launch overhead, not fewer mathematical operations alone.

---

### 4. Quantization

Map high-precision value \(x\) to integer representation \(q\):

\[
q
=
\mathrm{round}
\left(
\frac{x}{s}
\right)
+z.
\]

Dequantize:

\[
\hat x=s(q-z).
\]

Where:
- \(s\): scale;
- \(z\): zero point.

Quantization error:

\[
e=x-\hat x.
\]

Goal:
reduce memory/bandwidth and exploit low-precision hardware while keeping \(e\) acceptable.

---

### 5. PTQ vs QAT

#### Post-Training Quantization
Quantize after training, often using calibration data.

Pros:
cheap.

Cons:
accuracy may drop.

#### Quantization-Aware Training
Simulate quantization effects during training.

Pros:
better recovery.

Cons:
training complexity.

---

### 6. Weight-Only Quantization

Quantize model weights but keep activations in higher precision.

Especially useful for LLM decode because reading model weights is expensive.

---

### 7. Dynamic Shapes

Production inputs vary:
- image resolution;
- batch size;
- sequence length.

Inference engines may need optimization profiles covering shape ranges.

Static shapes enable more aggressive optimization but less flexibility.

---

### 8. LLM Serving

Important phases:

#### Prefill
parallel processing of prompt tokens.

#### Decode
autoregressive generation.

Important systems:
- KV cache;
- continuous/in-flight batching;
- paged cache management;
- quantized weights;
- tensor/pipeline parallelism.

---

### 9. Paged KV Cache

Instead of allocating one contiguous KV region per request, store cache in fixed-size blocks/pages.

Benefits:
- less fragmentation;
- easier dynamic allocation;
- better sharing/scheduling of variable-length sequences.

---

### 10. Continuous Batching

Static batching waits for a batch to finish together.

Continuous batching dynamically inserts new requests as others finish.

Benefit:
higher utilization under variable request lengths.

---

### 11. Benchmarking Correctly

Always record:
- hardware;
- software versions;
- precision;
- batch size;
- input shapes;
- warm-up;
- synchronization;
- number of runs.

GPU timing should synchronize because kernels are asynchronous.

For latency distributions, report:
- mean/median;
- p90/p95/p99 when appropriate.

For model conversion, also report numerical agreement:

\[
\mathrm{error}
=
\frac{\|y_{\rm optimized}-y_{\rm baseline}\|}
{\|y_{\rm baseline}\|}.
\]

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Benchmark the baseline and optimized path under identical shapes, precision, warm-up, and synchronization.
- Validate numerical agreement after conversion or quantization.
- Separate model-level optimization, runtime scheduling, and serving-system optimization.

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
