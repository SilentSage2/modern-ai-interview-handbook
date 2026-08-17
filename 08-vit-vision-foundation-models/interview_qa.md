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

## Extended Interview Q&A

### E1. Why can patch embedding be implemented with a convolution?

**Answer:** A Conv2D with kernel size equal to patch size and stride equal to patch size applies one learned linear projection to each non-overlapping patch.

### E2. What happens when patch size decreases?

**Answer:** Token count grows quadratically with inverse patch size, increasing spatial detail but making self-attention much more expensive.

### E3. Why does MAE use a high masking ratio?

**Answer:** Natural images are redundant. Heavy masking prevents trivial local copying and encourages the encoder to learn broader semantic/contextual structure.

### E4. How do you fine-tune ViT at a new image resolution?

**Answer:** The number of patch positions changes, so learned positional embeddings often need interpolation; the classifier head may also be replaced.

### E5. ViT vs Swin?

**Answer:** Vanilla ViT uses global attention over all patches; Swin uses shifted local windows to reduce complexity and build a hierarchical representation more suitable for dense vision tasks.


## Whiteboard / drill questions

- Compute the number of tokens for 384×384 images with 16×16 patches.
- Why can smaller patches become dramatically more expensive?
- Why can MAE drop masked patches from the encoder?
- What exactly does a linear probe measure?
- How would you adapt a pretrained ViT to 3D medical images?


<!-- ADVANCED_QA_START -->
## Advanced / system follow-ups

### A1. Why is CLS token not strictly necessary?

**Answer:** A global representation can also be formed by mean pooling patch tokens or another learned pooling mechanism. CLS is one design for collecting global information, not a mathematical requirement.

### A2. How would you adapt a 2D pretrained ViT to 3D data?

**Answer:** Options include slice-wise processing, inflating/learning 3D patch embedding, factorized spatial-depth attention, or using the 2D backbone as an encoder while adding depth aggregation. The choice depends on memory and whether through-plane structure matters.

### A3. What does a linear probe tell you that full fine-tuning does not?

**Answer:** It measures how linearly accessible downstream information already is in the frozen representation. Full fine-tuning mixes representation quality with adaptation capacity.

### A4. Why can high-resolution ViT inference become expensive faster than CNN inference?

**Answer:** Token count scales with image area divided by patch area, and global attention scales quadratically in token count. CNN cost generally scales more nearly linearly with pixel count for fixed kernels.

### A5. What would make a vision model a foundation model rather than just a pretrained ViT?

**Answer:** Broad pretraining plus evidence that the representation can adapt across multiple downstream tasks/domains/interfaces. Architecture alone does not make it a foundation model.

<!-- ADVANCED_QA_END -->
