# LLMs & Foundation Models — Interview Q&A

## Q1. Explain: Tokenization: BPE/SentencePiece intuition

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q2. Explain: Decoder-only causal language modeling

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q3. Explain: Next-token prediction objective

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q4. Explain: Scaling: parameters, tokens, compute

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q5. Explain: Base model vs instruction-following model

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q6. Explain: Supervised fine-tuning (SFT)

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q7. Explain: Preference learning: RLHF and DPO

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Q8. Explain: In-context learning and prompting

**Answer outline:** definition → intuition → key equation/architecture → tradeoff → example.

## Practice rule

For each answer prepare three versions:
- **30 sec:** concise definition + core intuition.
- **2 min:** equation/architecture + tradeoff.
- **5 min:** implementation, failure cases, and comparison with alternatives.

## Extended Interview Q&A

### E1. Base model vs instruction model?

**Answer:** A base model is optimized for next-token prediction over broad pretraining data. An instruction model is post-trained on instruction/response behavior and often preference data.

### E2. Why is next-token prediction enough to learn broad representations?

**Answer:** Predicting the next token across diverse data requires modeling syntax, semantics, world regularities, discourse, and task patterns. Scale turns a simple local objective into broad representation learning.

### E3. Perplexity vs task accuracy?

**Answer:** Perplexity measures predictive likelihood on text, not directly instruction following, factuality, reasoning, safety, or downstream utility.

### E4. SFT vs DPO?

**Answer:** SFT imitates target responses. DPO uses preference pairs to push preferred responses up relative to rejected responses and a reference model.

### E5. What is in-context learning?

**Answer:** The model changes behavior based on examples/instructions in the prompt without updating weights.

### E6. Temperature vs top-p?

**Answer:** Temperature globally reshapes the probability distribution; top-p truncates to a variable-size high-probability nucleus before sampling.

### E7. Why use MoE?

**Answer:** MoE increases total parameter capacity while activating only a subset of experts for each token, reducing compute relative to a dense model of equal parameter count.

