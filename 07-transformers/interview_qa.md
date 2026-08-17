# Transformers — Interview Q&A

## Q1. Why divide attention logits by sqrt(d_k)?

**Short answer:** To keep dot-product variance controlled as head dimension grows, preventing softmax from becoming excessively saturated.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Q2. Why is self-attention O(N^2) in sequence length?

**Short answer:** The attention score matrix compares every query token with every key token, producing an N×N matrix.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Q3. What is the difference between prefill and decode?

**Short answer:** Prefill processes the prompt in parallel and builds KV cache; decode generates new tokens autoregressively and repeatedly reads the cache.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Practice rule

For each answer prepare three versions:
- **30 sec:** concise definition + core intuition.
- **2 min:** equation/architecture + tradeoff.
- **5 min:** implementation, failure cases, and comparison with alternatives.

## Extended Interview Q&A

### E1. Why do we need Q, K and V instead of only one embedding?

**Answer:** Separate projections let the model learn one representation for matching/selection and another for transmitted content. The query-key dot product decides where to look; the value determines what information is aggregated.

### E2. Why does multi-head attention help?

**Answer:** It gives multiple learned projection subspaces and attention patterns in parallel. Different heads can specialize in different relations while keeping total model width manageable.

### E3. Attention vs FFN in a Transformer block?

**Answer:** Attention mixes information across token positions; the FFN independently transforms the feature channels of each token.

### E4. Why are Transformers permutation equivariant without positional encoding?

**Answer:** All token interactions depend only on pairwise content projections. Permuting input rows produces the same permutation in outputs unless position information is injected.

### E5. What is RoPE's main mathematical advantage?

**Answer:** After position-dependent rotations, the query-key inner product depends on the relative position difference, so relative distance is naturally encoded in attention.

### E6. Why is decode often memory-bandwidth limited?

**Answer:** Each decode step handles very few new tokens but must repeatedly read large model weights and KV cache, giving lower arithmetic intensity than prompt prefill.

### E7. What does KV cache save?

**Answer:** It avoids recomputing past keys and values at every autoregressive step. It reduces compute but adds memory proportional to layers × sequence length × KV heads × head dimension.

### E8. What is GQA?

**Answer:** Grouped-query attention uses more query heads than key/value heads. Multiple query heads share each KV head, reducing KV cache memory while preserving more flexibility than single-head MQA.

### E9. What does FlashAttention change?

**Answer:** It changes how exact attention is scheduled and tiled on GPU to reduce high-bandwidth-memory traffic. It does not fundamentally change the mathematical attention output.


## Whiteboard / drill questions

- Derive the scaling factor in attention from a variance argument.
- Why does KV cache reduce compute but increase memory?
- Why can prefill be compute-bound while decode is bandwidth-bound?
- How do MHA, MQA and GQA change KV memory?
- Why is FlashAttention exact rather than approximate?
- How would attention shapes change under cross-attention?


<!-- ADVANCED_QA_START -->
## Advanced / system follow-ups

### A1. Why might attention heads become redundant?

**Answer:** Multiple heads are not guaranteed to learn distinct functions. Optimization can produce correlated heads, and pruning studies often find some heads have limited marginal contribution. Multi-head structure provides capacity, not guaranteed semantic specialization.

### A2. What changes in cross-attention complexity?

**Answer:** If query length is Tq and context length is Tc, score complexity is O(Tq·Tc·d) rather than O(T²d). This matters in VLMs and encoder-decoder models where the two modalities/sequences have different lengths.

### A3. Why can long context hurt even if it fits in memory?

**Answer:** More tokens increase compute and may dilute retrieval/attention. The model must also have been trained to use long-range information effectively; nominal context length is not equivalent to reliable long-context reasoning.

### A4. How would you verify a causal mask implementation?

**Answer:** Create two sequences identical up to position t but different afterward. Outputs at positions ≤t should be unchanged. This is a stronger test than visually inspecting the triangular mask.

### A5. Why can GQA preserve quality better than MQA?

**Answer:** GQA retains multiple K/V groups rather than forcing all query heads to share a single K/V representation, offering a middle point between full MHA expressiveness and MQA cache savings.

<!-- ADVANCED_QA_END -->
