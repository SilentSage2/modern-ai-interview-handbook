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


```math
L=\text{time per request}.
```


Throughput:


```math
Q=\frac{\text{requests or tokens}}{\text{second}}.
```


Increasing batch size often raises throughput while potentially increasing queueing and per-request latency.

---

### 2. Inference Optimization Pipeline

Conceptually:


```math
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
```


TensorRT is an optimized inference compiler/runtime for NVIDIA GPUs.

---

### 3. Operator Fusion

Suppose:


```math
y=\mathrm{ReLU}(\mathrm{BN}(\mathrm{Conv}(x))).
```


Naive execution may:
- launch multiple kernels;
- write/read intermediate tensors to global memory.

Fusion executes combined operations with fewer memory round-trips and kernel launches.

Benefit is often from reduced memory traffic and launch overhead, not fewer mathematical operations alone.

---

### 4. Quantization

Map high-precision value $x$ to integer representation $q$:


```math
q
=
\mathrm{round}
\left(
\frac{x}{s}
\right)
+z.
```


Dequantize:


```math
\hat x=s(q-z).
```


Where:
- $s$: scale;
- $z$: zero point.

Quantization error:


```math
e=x-\hat x.
```


Goal:
reduce memory/bandwidth and exploit low-precision hardware while keeping $e$ acceptable.

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


```math
\mathrm{error}
=
\frac{\|y_{\rm optimized}-y_{\rm baseline}\|}
{\|y_{\rm baseline}\|}.
```


<!-- DEEP_DIVE_START -->
## Deep dive I: training graph vs inference graph

Training requires:
- gradient graph;
- dropout/randomness;
- optimizer state;
- intermediate activations for backward.

Inference needs only forward computation.

Therefore deployment can:
- remove training-only nodes;
- fold constants;
- fuse operations;
- select hardware-specific kernels;
- reduce precision.

This creates optimization opportunities unavailable during general eager training.

---

## Deep dive II: TensorRT builder and runtime mental model

Think in two phases.

### Build


```math
\text{model graph}
\rightarrow
\text{optimized engine}.
```


The builder considers compatible kernel implementations/tactics, tensor layouts, precision, dynamic-shape profiles, and memory planning.

### Runtime

Load serialized engine and execute with concrete inputs.

This distinction explains why engine build can take time even though inference is fast.

---

## Deep dive III: ONNX path

Common path:

```text
PyTorch
→ export
→ ONNX graph
→ TensorRT parser/builder
→ serialized engine
→ TensorRT runtime
```

Potential failure sources:
- unsupported operator;
- dynamic control flow;
- incompatible shape assumptions;
- custom op;
- numerical mismatch.

Custom plugins can implement unsupported operations but increase maintenance burden.

---

## Deep dive IV: operator fusion worked example

Separate kernels:

```text
linear
→ bias add
→ activation
```

Each may write intermediate output to HBM.

Fused kernel:

```text
linear + bias + activation
```

can keep intermediate values closer to registers/shared/on-chip memory.

Performance benefit can come from:


```math
\text{memory traffic}\downarrow
+
\text{kernel launches}\downarrow.
```


---

## Deep dive V: quantization derivation

For symmetric signed $b$-bit quantization, integer range approximately:


```math
[-(2^{b-1}-1),2^{b-1}-1].
```


Choose scale based on max range:


```math
s=
\frac{\max |x|}
{2^{b-1}-1}.
```


Then:


```math
q=\mathrm{clip}
\left(
\mathrm{round}(x/s)
\right),
\qquad
\hat x=sq.
```


If one outlier makes $\max|x|$ huge, most ordinary values use only a small part of integer range, increasing quantization error. This motivates calibration and per-channel/group scaling.

---

## Deep dive VI: per-tensor vs per-channel

Per-tensor:


```math
x\rightarrow s
```


one scale for whole tensor.

Per-channel:


```math
x_c\rightarrow s_c.
```


Per-channel better adapts to differing channel ranges but stores more scales and may complicate kernels.

For weight matrices, per-channel/group quantization is often attractive.

---

## Deep dive VII: static vs dynamic shape optimization

Static shape gives compiler maximum certainty:


```math
[B,C,H,W]=[1,3,224,224].
```


Dynamic deployment may require ranges:

```text
min shape
opt shape
max shape
```

Broader ranges improve flexibility but can constrain specialization and require more engine/profile planning.

---

## Deep dive VIII: LLM serving memory budget

GPU memory contains:
- model weights;
- KV cache;
- runtime workspaces;
- temporary activations.

Quantizing weights frees memory that can be used for:
- larger model;
- more concurrent requests;
- longer KV cache.

Thus quantization can increase serving throughput even beyond raw kernel speed.

---

## Deep dive IX: paged KV cache

Naive contiguous allocation for each request can waste memory because sequence lengths vary and grow dynamically.

Paged design divides cache into blocks:

```text
request A → pages 3, 9, 10
request B → pages 1, 7
request C → pages 2, 4, 5, 11
```

Logical sequence order is decoupled from physical contiguous memory.

This reduces fragmentation and supports dynamic scheduling.

---

## Deep dive X: continuous batching timeline

Static:

```text
A: token token token DONE
B: token token token token token DONE
[new request waits]
```

Continuous:

```text
A finishes → slot immediately reused by C
B continues
```

This increases utilization when request lengths differ.

---

## Deep dive XI: latency decomposition

End-to-end latency can include:


```math
L_{\rm total}
=
L_{\rm queue}
+
L_{\rm preprocess}
+
L_{\rm H2D}
+
L_{\rm compute}
+
L_{\rm D2H}
+
L_{\rm postprocess}.
```


Optimizing only kernel time may not improve user-visible latency if queueing or data movement dominates.

For LLMs, also distinguish:
- time to first token;
- inter-token latency;
- total generation latency.

---

## Deep dive XII: benchmark methodology

Bad benchmark:

```python
t0 = time.time()
y = model(x)
t1 = time.time()
```

CUDA is asynchronous, so CPU timer may stop before GPU work completes.

Better:

```python
torch.cuda.synchronize()
t0 = time.perf_counter()

for _ in range(N):
    y = model(x)

torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
```

Also:
- warm up first;
- report batch/input shapes;
- fix precision;
- avoid including compilation unless intended;
- report latency distribution.

---

## Deployment experiment template

### Baseline
PyTorch eager, FP32/BF16/FP16.

### Optimized
TensorRT engine.

### Optional
INT8 or other supported quantization.

Measure:
- peak GPU memory;
- mean/p50/p95 latency;
- throughput;
- numerical error;
- build time;
- engine size.

A convincing deployment project should report both **speed** and **correctness**.

---

## Current TensorRT concepts to know

Modern TensorRT documentation organizes the workflow around:
- a builder that compiles/optimizes a trained model into an engine;
- a runtime that executes that engine;
- precision control;
- dynamic shapes;
- quantization;
- transformer-specific optimization.

The precise APIs evolve, so interviews should focus first on these stable system concepts and then on the version-specific API you actually used.
<!-- DEEP_DIVE_END -->

<!-- SECOND_DEEP_DIVE_START -->
## Deployment architecture case study

A production inference service often looks like:

```text
client
→ request queue
→ preprocessing
→ scheduler/batcher
→ GPU runtime
→ postprocessing
→ response
```

TensorRT optimizes the GPU execution part, but system latency also depends on surrounding layers.

This is why “my TensorRT kernel is 2× faster” may not mean user request is 2× faster.

---

## Throughput model

If average service time per batch is $t_B$ and batch contains $B$ items:


```math
\mathrm{throughput}
\approx
\frac{B}{t_B}.
```


If $t_B$ grows sublinearly with $B$, throughput improves.

But request latency includes waiting to form a batch:


```math
L_{\rm request}
=
L_{\rm queue}+t_B+\cdots.
```


Offline batch inference and online low-latency serving therefore optimize different objectives.

---

## Time to first token vs tokens/sec

For LLM:

### TTFT
Includes:
- queue;
- prompt preprocessing;
- prefill.

### Inter-token latency
Mostly decode loop.

A system can have:
- fast TTFT but slow generation;
- slow TTFT for long prompts but high decode throughput.

Report both where relevant.

---

## Quantization error and sensitive layers

Not all tensors tolerate low precision equally.

A deployment workflow may:
1. quantize most layers;
2. identify accuracy-sensitive operations;
3. keep those in higher precision.

This mixed-precision strategy trades performance for accuracy selectively.

---

## Calibration data

PTQ calibration should represent deployment distribution.

If calibration only contains narrow easy samples, observed activation ranges may be misleading.

Calibration is therefore a small-data estimation problem:


```math
\text{estimate quantization scale from representative activations}.
```


---

## Engine portability

Highly optimized engines may depend on:
- GPU architecture;
- TensorRT version;
- precision support;
- optimization profile.

Deployment should treat build environment/version as part of artifact metadata.

Do not assume an engine binary is universally portable across arbitrary machines.

---

## Dynamic batching vs continuous batching

### Dynamic batching
Server groups independent requests into batches before inference.

Common for vision/classification.

### Continuous batching
For autoregressive LLM generation, scheduling occurs repeatedly across decode steps. Finished sequences leave and new ones can join.

This finer-grained scheduling addresses variable generation lengths.

---

## Speculative decoding concept

Use a smaller/faster draft model to propose several tokens:


```math
y_{t:t+k}^{\rm draft}.
```


Large target model verifies them in fewer expensive sequential steps.

If many proposed tokens are accepted, generation latency improves while preserving target-model distribution under the chosen algorithm.

Conceptually this attacks the sequential decode bottleneck with **prediction + verification**.

---

## Model compilation vs serving

Separate these concerns.

### Compilation/runtime optimization
- TensorRT;
- fused kernels;
- quantization;
- graph optimization.

### Serving
- request scheduling;
- batching;
- caching;
- replication;
- load balancing;
- autoscaling;
- API.

One optimized model can still be poorly served, and one excellent scheduler cannot fix a very slow kernel.

---

## TensorRT conversion checklist

Before conversion:
- put model in eval mode;
- identify input shapes;
- define tolerance;
- save representative inputs;
- benchmark baseline.

After conversion:
- compare output numerically;
- benchmark warm and steady-state latency;
- test min/opt/max dynamic shapes;
- test failure inputs;
- record engine/version metadata.

---

## Numerical comparison

For outputs $y$ and $\hat y$, use more than one metric:

Absolute max:


```math
e_\infty
=
\max_i|y_i-\hat y_i|.
```


Relative L2:


```math
e_{\rm rel}
=
\frac{\|y-\hat y\|_2}{\|y\|_2+\epsilon}.
```


Task metric:
- classification agreement;
- Dice;
- reconstruction error;
- downstream quantitative output.

Small tensor error may or may not matter to task.

---

## Mini-project: TensorRT end-to-end

Choose a model you understand well, e.g. ViT or U-Net.

### Step 1
Benchmark PyTorch baseline.

### Step 2
Export/compile.

### Step 3
Run TensorRT FP16/BF16 if supported.

### Step 4
Optional quantization.

### Step 5
Compare across batch sizes.

Create plots:
- latency vs batch;
- throughput vs batch;
- memory vs batch;
- numerical error vs precision.

### Interview summary
Explain not only the speedup but **why** it happened:
- fusion?
- lower precision?
- better kernels?
- larger effective batch?
- reduced overhead?

---

## Common deployment mistakes

**Benchmarking first run only:** includes warmup/compilation.

**No synchronization:** asynchronous GPU timing is wrong.

**Ignoring preprocessing:** end-to-end latency remains high.

**Reporting only average latency:** hides tail behavior.

**No correctness test:** faster wrong output is not optimization.

**Huge batch for “best throughput”:** may violate application latency target.
<!-- SECOND_DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Benchmark the baseline and optimized path under identical shapes, precision, warm-up, and synchronization.
- Validate numerical agreement after conversion or quantization.
- Separate model-level optimization, runtime scheduling, and serving-system optimization.

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
