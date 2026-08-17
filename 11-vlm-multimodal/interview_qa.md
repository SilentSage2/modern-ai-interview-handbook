# VLM & Multimodal Foundation Models — Interview Q&A

## Q1. What does CLIP learn?

**Short answer:** A joint image-text embedding space in which matched image/text pairs have high similarity and mismatched pairs have lower similarity.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Q2. How can visual information enter an LLM?

**Short answer:** A vision encoder produces visual features/tokens, then a projector, resampler, or cross-attention mechanism maps/fuses them into representations consumable by the language model.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Practice rule

For each answer prepare three versions:
- **30 sec:** concise definition + core intuition.
- **2 min:** equation/architecture + tradeoff.
- **5 min:** implementation, failure cases, and comparison with alternatives.
