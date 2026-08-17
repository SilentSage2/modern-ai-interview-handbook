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

