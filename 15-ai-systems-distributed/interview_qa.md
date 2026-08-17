# GPU, Distributed Training & AI Systems — Interview Q&A

## Q1. Explain: GPU memory hierarchy and bandwidth

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q2. Explain: Training memory: parameters, gradients, optimizer states, activations

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q3. Explain: Mixed precision: FP32, FP16, BF16, FP8 concepts

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q4. Explain: Data parallelism

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q5. Explain: Tensor parallelism

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q6. Explain: Pipeline parallelism

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q7. Explain: DDP, FSDP and ZeRO concepts

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q8. Explain: NCCL and collective communication

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Practice rule

For each answer prepare three versions:
- **30 sec:** concise definition + core intuition.
- **2 min:** equation/architecture + tradeoff.
- **5 min:** implementation, failure cases, and comparison with alternatives.

## Extended Interview Q&A

### E1. Data parallel vs tensor parallel?

**Answer:** Data parallel replicates the model and splits examples; tensor parallel splits operations/weights of the same layer across devices.

### E2. Why does FSDP save memory?

**Answer:** It shards model states across workers instead of storing complete parameter/gradient/optimizer copies on every GPU.

### E3. What is arithmetic intensity?

**Answer:** FLOPs per byte moved. High intensity tends to be compute-bound; low intensity tends to be memory-bandwidth bound.

### E4. Why can bigger batch improve GPU throughput?

**Answer:** It exposes more parallel work and amortizes kernel launch/overhead, but costs memory and may increase latency.

### E5. Why is BF16 often preferred over FP16?

**Answer:** BF16 has a much wider exponent range, reducing overflow/underflow risk in large-model training, while still using 16 bits.

