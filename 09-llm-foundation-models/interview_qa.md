# LLMs & Foundation Models — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. What exactly is an LLM trained to do during pretraining?

A decoder-only Large Language Model (LLM) factorizes sequence probability into next-token conditionals and maximizes their likelihood.

```math
p(x_{1:T})
=
\prod_{t=1}^{T}
p(x_t|x_{1:t-1}).
```

Training therefore becomes cross-entropy prediction of the next token at every position.

**Key idea.** The objective is simple, but solving it across enormous heterogeneous corpora requires modeling grammar, semantics, factual associations, code, discourse, and many task patterns. Broad capability emerges because all of those structures help predict what comes next.

**Important distinction.** Pretraining teaches continuation. It does not by itself guarantee helpful instruction-following behavior, which is why post-training such as SFT and preference optimization exists.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. What is the difference between a base model and an instruction model?

A **base model** is mainly pretrained for next-token prediction. If prompted with a question, it may continue the text rather than consistently behave like an assistant.

An **instruction model** undergoes post-training such as Supervised Fine-Tuning (SFT) on instruction–response examples and often preference optimization.

**Design idea.** Pretraining creates broad capability; post-training shapes how that capability is exposed and which behaviors are preferred.

A strong answer separates:
- pretraining;
- SFT;
- preference optimization such as RLHF/DPO;
- inference prompting.

Calling all of these simply “fine-tuning” hides important differences in objective and data.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. SFT, RLHF, and DPO: what does each stage optimize?

**SFT (Supervised Fine-Tuning)** imitates target responses token by token.

**RLHF (Reinforcement Learning from Human Feedback)** typically learns a reward or preference signal and optimizes the policy while constraining it from drifting too far from a reference model.

**DPO (Direct Preference Optimization)** uses preferred/rejected response pairs and directly increases the preferred response relative to the rejected response and the reference model, without an explicit PPO loop.

**Key difference in supervision.**
- SFT says: “produce this response.”
- Preference learning says: “response A is better than response B.”

DPO is operationally simpler than classical RLHF, but reinforcement learning remains important when rewards come from interactive environments, tools, verifiers, or long-horizon behavior.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. What do temperature, top-k, and top-p sampling change?

The model outputs logits over the vocabulary. Sampling controls how those probabilities are converted into the next token.

**Temperature** rescales all logits. Lower temperature sharpens the distribution; higher temperature increases diversity.

**Top-k** keeps only the k highest-probability candidates.

**Top-p (nucleus sampling)** keeps the smallest set whose cumulative probability exceeds a threshold p, so candidate-set size adapts to model uncertainty.

**Design tradeoff.** Low-randomness decoding is stable and useful for deterministic tasks but can become repetitive. Higher randomness increases diversity but also the probability of unsupported or inconsistent generations.

The correct decoding policy depends on the application rather than one universally best setting.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. Why is perplexity not enough to evaluate an LLM?

Perplexity is the exponentiated average token Negative Log-Likelihood (NLL). It measures how well the model predicts held-out text under its tokenizer.

It does **not** directly measure:
- instruction following;
- factual accuracy;
- safety;
- reasoning;
- code correctness;
- tool-use reliability;
- grounding.

Two models can have similar perplexity but very different post-training behavior.

**Evaluation principle.** Match the metric to the desired behavior. Code should be evaluated with tests; RAG with grounding and retrieval; agents with task success/tool correctness; serving with latency and throughput.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. What is Mixture of Experts and why use it?

A Mixture-of-Experts (MoE) layer contains multiple expert networks and a learned router. Each token is sent only to a small top-k subset.

```math
y
=
\sum_{i\in\operatorname{TopK}(g(x))}
g_i(x)E_i(x).
```

**Key idea:** conditional computation. The model can have many total parameters while activating only part of them for each token.

**Benefit:** more capacity without proportional FLOPs.

**Costs:** routing imbalance, expert communication across devices, larger total parameter storage, and serving complexity. MoE therefore improves the parameter-to-compute ratio but is not “free capacity.”

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q7. Why does context length affect prefill and decode differently?

During **prefill**, all prompt tokens are processed together. Dense attention over a prompt of length T has pairwise interaction scaling roughly with T², although optimized kernels reduce memory traffic.

During **decode**, each new token attends to the cached history. The KV cache grows linearly with context length, and each step repeatedly reads large weights plus cache state.

Therefore long context creates different bottlenecks in the two phases:
- prefill: large parallel attention/compute;
- decode: sequential generation, memory traffic, and cache capacity.

A nominal long context window does not imply that using the maximum context is cheap or equally reliable for retrieval/reasoning.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q8. Why do LLMs hallucinate?

An LLM is optimized to produce probable continuations under its learned distribution, not to satisfy a formal truth constraint. If the prompt does not contain enough evidence, a linguistically plausible statement can still have high probability.

Hallucination can be reduced through retrieval, tools, verifiers, improved post-training, grounding, and uncertainty/refusal behavior.

**Key idea.** Hallucination is not one isolated softmax bug. It reflects a mismatch between the training objective—predict plausible tokens—and the application objective—produce statements that are verified and supported by evidence.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

