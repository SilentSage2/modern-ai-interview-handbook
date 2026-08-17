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

