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


## Whiteboard / drill questions

- Why is CLIP naturally suited to retrieval but not free-form generation?
- What is lost when compressing 1024 visual tokens to 32 latent tokens?
- How can you prove a VLM is actually using its image?
- Projector vs cross-attention: when would you prefer each?
- How would you evaluate visual hallucination?


<!-- ADVANCED_QA_START -->
## Advanced / system follow-ups

### A1. How would you test whether a VLM is ignoring the image?

**Answer:** Compare normal image, shuffled image, blank image, and text-only conditions. If outputs barely change, language priors may dominate.

### A2. Why is OCR a special VLM challenge?

**Answer:** Small characters can be lost by image resizing/patchification before language reasoning starts. The bottleneck may be visual resolution, not LLM capacity.

### A3. What is the difference between grounding and recognition?

**Answer:** Recognition identifies what is present; grounding links a concept/phrase to a specific visual region or evidence. A model can recognize correctly without precise localization.

### A4. Why can a dual encoder scale retrieval well?

**Answer:** Images and texts can be embedded independently offline, then compared with fast vector similarity. Cross-attention models require joint computation for each pair and are much more expensive for large retrieval corpora.

### A5. When would you use a reranker after CLIP-like retrieval?

**Answer:** Use the dual encoder for high-recall candidate retrieval, then a more expensive cross-modal model to assess fine-grained relevance on a small candidate set.

<!-- ADVANCED_QA_END -->
