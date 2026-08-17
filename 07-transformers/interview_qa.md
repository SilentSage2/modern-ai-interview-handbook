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
