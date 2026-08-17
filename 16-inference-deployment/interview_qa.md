# Inference, TensorRT & Deployment — Interview Q&A

## Q1. Latency vs throughput?

**Short answer:** Latency measures time per request; throughput measures work completed per unit time. Batching often improves throughput while potentially increasing per-request latency.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Q2. What does TensorRT optimize?

**Short answer:** It compiles and optimizes inference graphs using tactics such as kernel selection/fusion, precision choices, memory planning, and hardware-specific execution.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Practice rule

For each answer prepare three versions:
- **30 sec:** concise definition + core intuition.
- **2 min:** equation/architecture + tradeoff.
- **5 min:** implementation, failure cases, and comparison with alternatives.

## Extended Interview Q&A

### E1. Why can TensorRT be faster than eager PyTorch?

**Answer:** It can compile a fixed/known graph, fuse operators, choose optimized kernels/tactics, plan memory, and exploit lower precision more aggressively.

### E2. PTQ vs QAT?

**Answer:** PTQ quantizes after training using calibration; QAT exposes the model to simulated quantization during training for better robustness.

### E3. Why does weight-only quantization help LLM decode?

**Answer:** Decode repeatedly streams large weights for few new tokens, so reducing weight bytes can relieve memory-bandwidth pressure.

### E4. What is continuous batching?

**Answer:** The server dynamically adds/removes requests at token-step boundaries instead of forcing a fixed batch to finish together.

### E5. Why use paged KV cache?

**Answer:** Paged allocation reduces fragmentation and makes variable-length, dynamically arriving sequences easier to manage efficiently.

### E6. How should GPU inference latency be measured?

**Answer:** Warm up, synchronize around timed regions, fix shapes/precision, run many iterations, and report distribution statistics plus hardware/software context.


## Whiteboard / drill questions

- Why can an optimized engine differ numerically from eager output?
- How can dynamic shapes reduce optimization opportunities?
- Why does weight quantization help decode throughput?
- How do paged KV cache and continuous batching solve different problems?
- Design a fair PyTorch vs TensorRT benchmark.


<!-- ADVANCED_QA_START -->
## Advanced / system follow-ups

### A1. Why can lower precision improve latency beyond memory savings?

**Answer:** It reduces data movement and may unlock specialized tensor-core instructions with higher throughput. Actual benefit depends on kernel support and workload shape.

### A2. What is tail latency?

**Answer:** High-percentile latency such as p95/p99. Online systems care because a small fraction of very slow requests can dominate user experience or SLA violations.

### A3. Why should engine build time be reported separately from inference time?

**Answer:** Compilation/tactic search is usually an offline cost. Mixing it into steady-state inference obscures actual serving performance.

### A4. When is INT8 PTQ likely to fail?

**Answer:** When activation/weight distributions are difficult to represent with chosen scales or calibration data are unrepresentative, causing large quantization error in sensitive layers.

### A5. Why can larger batches hurt an online service despite higher throughput?

**Answer:** Requests may wait longer to form or complete a batch, increasing queueing and tail latency. Throughput and latency objectives conflict.

<!-- ADVANCED_QA_END -->
