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

## Extended Interview Q&A

### E1. Why normalize CLIP embeddings?

**Answer:** Normalization makes similarity depend primarily on direction/cosine rather than arbitrary embedding magnitude, stabilizing contrastive alignment.

### E2. Why is a projector needed between ViT and LLM?

**Answer:** Vision features and LLM token states have different dimensions and representation distributions; the projector learns a compatibility interface.

### E3. Projector concatenation vs cross-attention?

**Answer:** Concatenation turns visual embeddings into token-like context processed by self-attention. Cross-attention keeps modalities more separate and lets text explicitly query visual features.

### E4. Why freeze the vision encoder first?

**Answer:** It reduces compute and protects a strong pretrained representation while the new multimodal interface learns alignment.

### E5. How can VLM hallucination happen?

**Answer:** The language prior may dominate weak visual grounding, the visual encoder/projector may discard detail, or instruction data may reward fluent but insufficiently grounded answers.

