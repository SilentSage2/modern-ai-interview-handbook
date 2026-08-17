# ViT & Vision Foundation Models — Interview Q&A

## Q1. How does ViT convert an image into Transformer input?

**Short answer:** Split the image into patches, flatten/project each patch into an embedding, add positional information, and process the token sequence with Transformer encoder blocks.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Q2. Why can CNNs be more data-efficient than vanilla ViTs?

**Short answer:** CNNs impose locality and translation-equivariance biases, while ViTs learn more of the spatial structure from data.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Practice rule

For each answer prepare three versions:
- **30 sec:** concise definition + core intuition.
- **2 min:** equation/architecture + tradeoff.
- **5 min:** implementation, failure cases, and comparison with alternatives.
