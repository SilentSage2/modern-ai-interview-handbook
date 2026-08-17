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


## Whiteboard / drill questions

- Estimate Adam training-state memory for a 7B model.
- Why does FSDP save memory but add communication?
- When do you choose tensor parallel rather than data parallel?
- Why might a GPU show low utilization despite enough memory?
- Explain arithmetic intensity using LLM prefill vs decode.


<!-- ADVANCED_QA_START -->
## Advanced / system follow-ups

### A1. Why can more GPUs make training slower?

**Answer:** Communication, synchronization, input bottlenecks, or smaller per-GPU work can outweigh parallel compute gains. Scaling should be measured rather than assumed.

### A2. What memory does gradient checkpointing not reduce?

**Answer:** It primarily reduces stored activations. It does not inherently shrink parameter, optimizer-state, or gradient storage.

### A3. Why does tensor parallelism have more frequent communication than data parallelism?

**Answer:** Tensor-parallel devices cooperate inside individual layer operations, whereas data-parallel replicas can compute a larger local portion of the step before synchronizing gradients.

### A4. Why can CPU preprocessing limit a fast GPU?

**Answer:** If batches are not prepared and transferred fast enough, the GPU has idle gaps. End-to-end throughput is limited by the slowest pipeline stage.

### A5. What is a good first step when GPU utilization is low?

**Answer:** Profile the timeline and check input pipeline, host-device copies, kernel sizes, synchronization, and communication before changing the algorithm.

<!-- ADVANCED_QA_END -->
