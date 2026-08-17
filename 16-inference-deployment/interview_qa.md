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
