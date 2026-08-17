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
