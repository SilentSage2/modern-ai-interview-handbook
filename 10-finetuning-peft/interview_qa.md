# Fine-Tuning, LoRA & PEFT — Interview Q&A

## Q1. Full fine-tuning vs LoRA?

**Short answer:** Full FT updates all parameters; LoRA freezes base weights and learns low-rank update matrices, reducing trainable parameters, memory, and adapter storage.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Q2. Why can low-rank adaptation work?

**Short answer:** A useful hypothesis is that task-specific parameter updates lie in a much lower-dimensional subspace than the full weight matrix.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Practice rule

For each answer prepare three versions:
- **30 sec:** concise definition + core intuition.
- **2 min:** equation/architecture + tradeoff.
- **5 min:** implementation, failure cases, and comparison with alternatives.

## Extended Interview Q&A

### E1. Why does LoRA reduce optimizer memory?

**Answer:** The frozen base parameters do not need trainable gradients or optimizer states; only the small A/B adapter matrices are optimized.

### E2. Does LoRA reduce inference compute automatically?

**Answer:** Not necessarily. If the adapter branch is evaluated separately it adds some work. If merged into the base weights for a fixed adapter, inference can use the merged matrix.

### E3. What does LoRA rank control?

**Answer:** The rank bounds the dimensionality of the weight update. Higher rank increases adaptation capacity and parameter cost.

### E4. Why initialize one LoRA factor to zero?

**Answer:** It ensures the initial update is exactly zero, so training begins from the original pretrained function.

### E5. When would full fine-tuning beat LoRA?

**Answer:** When the downstream distribution/behavior differs strongly and sufficient data/compute are available, full fine-tuning can exploit more adaptation capacity.

### E6. When is RAG preferable to fine-tuning?

**Answer:** When the main need is fresh, proprietary, or frequently changing knowledge rather than changing the model's core behavior.


## Whiteboard / drill questions

- Calculate LoRA parameters for a 4096×4096 matrix at ranks 8, 16, and 64.
- Why can QLoRA fit a larger base model than LoRA alone?
- When should you target MLP projections in addition to attention?
- What evaluation proves fine-tuning did not destroy general capability?
- Design a full-FT vs LoRA comparison that is scientifically fair.


<!-- ADVANCED_QA_START -->
## Advanced / system follow-ups

### A1. Why can LoRA rank be different across layers?

**Answer:** Different layers may require different adaptation capacity. A fixed global rank is convenient but not theoretically necessary; rank patterns can allocate capacity where updates are more complex.

### A2. Does freezing parameters mean no activation memory is needed through them?

**Answer:** No. If gradients must flow through a frozen layer to reach trainable adapters later/inside the network, activations or recomputation may still be needed. Freezing mainly removes parameter gradients and optimizer state for those weights.

### A3. How do you detect catastrophic forgetting?

**Answer:** Evaluate both the new target task and representative pretraining/general capabilities before and after adaptation. A target-task gain with broad regression is a forgetting signal.

### A4. Why might LoRA outperform full FT on small data?

**Answer:** The low-rank constraint can act as regularization and preserve pretrained structure, while full FT has enough degrees of freedom to overfit. This is data-regime dependent, not universal.

### A5. What is the difference between PEFT and quantization?

**Answer:** PEFT reduces or structures the trainable parameter set; quantization reduces numerical precision/storage. QLoRA combines both: quantized frozen base plus trainable low-rank adapters.

<!-- ADVANCED_QA_END -->
