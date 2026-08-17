# RAG, AI Agents & Coding Agents — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. What is dense retrieval and what is the key idea behind the equations?

Dense retrieval converts a query and each document/chunk into learned vectors:

```math
z_q=f_q(q),
\qquad
z_d=f_d(d).
```

It then scores semantic compatibility using a dot product or cosine similarity:

```math
s(q,d)=z_q^\top z_d.
```

The system retrieves the top `k` documents with the largest scores; here **k simply means the number of candidates returned**, not an acronym.

**Key idea.** The equation is not the important innovation. The encoders learn a **semantic geometry** in which relevant query/document pairs become close. Once that geometry is learned, fast vector search can retrieve semantically related content without exact keyword overlap.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. How is a dense retriever trained contrastively?

For a query with positive document `d+`, the retriever should assign the positive a larger similarity than negative candidates:

```math
L
=
-\log
\frac{
\exp(s(q,d^+)/\tau)
}{
\exp(s(q,d^+)/\tau)
+
\sum_j
\exp(s(q,d_j^-)/\tau)
}.
```

This is a softmax classification problem: identify the positive document among candidates.

**Design choices that matter more than the formula:** how positives are defined, how hard negatives are sampled, whether the query/document encoders share weights, and whether semantically valid examples become false negatives.

The same contrastive principle appears in CLIP and self-supervised representation learning.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. Why do RAG systems use chunking and reranking?

Documents are usually too long to embed and retrieve as one unit, so they are divided into chunks.

- Small chunks give precise retrieval but can lose surrounding context.
- Large chunks preserve context but dilute relevance and consume more LLM tokens.

A common two-stage architecture is:
1. fast retriever returns many candidates with high recall;
2. slower reranker scores the small candidate set more precisely;
3. only the best few chunks enter the LLM context.

**Key idea.** RAG is a recall–precision–context-budget problem, not merely “put documents in a vector database.”

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. RAG versus fine-tuning: what problem does each solve?

Retrieval-Augmented Generation (RAG) retrieves external information at request time and inserts it into context. Fine-tuning changes model parameters.

Use RAG for:
- fresh facts;
- proprietary documents;
- citation and grounding;
- frequently changing knowledge.

Use fine-tuning for:
- behavior and style;
- domain-specific mappings;
- output formats;
- task competence.

They can be combined. A scientific assistant can be LoRA-tuned to follow a workflow while retrieving the latest papers or experiment notes externally.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. What makes an LLM system an agent?

An agent has a **closed action–observation loop**:

```text
goal / state
→ model chooses tool or action
→ runtime executes it
→ observation/tool result
→ state update
→ model chooses next action
```

A long prompt or chain-of-thought alone does not make an agent.

Production agent systems often need tool schemas, memory/state, planning, retries, permissions, human approvals, tracing, and stopping conditions.

**Key idea.** Agent quality is a systems property created by the model plus its environment and control loop, not simply by the language model's fluency.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. What is ReAct?

ReAct means **Reason + Act**. The model alternates between reasoning/planning, an external action, and a new observation.

The value is that the model can update its next decision using real evidence from search, a database, code execution, files, or other tools instead of relying only on pretrained memory.

**Failure modes** include wrong tool selection, malformed arguments, loops, stale plans, and misinterpreted tool output.

Therefore a production agent needs explicit state, verification, and stopping rules in addition to a ReAct-style prompting pattern.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q7. When should I use an agent versus a deterministic workflow?

Use a deterministic workflow when the sequence of steps is known and stable:

```text
A → B → C
```

Use an agent when the next step depends on ambiguous, unstructured context and the system must choose dynamically among tools or strategies.

A strong production design is often hybrid: deterministic outer control with agentic decision points.

**Why this matters.** Agents add latency, cost, variability, and failure modes. Making every pipeline “agentic” is not automatically better engineering.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q8. How should an agent be evaluated?

Do not evaluate only final-answer fluency.

Measure:
- end-task success;
- retrieval recall/grounding;
- tool selection accuracy;
- argument correctness;
- execution failures and recovery;
- step count;
- latency/token cost;
- unsafe or unauthorized action attempts;
- human-approval behavior.

For coding agents, add test pass rate and diff correctness.

**Key idea.** Tracing every model/tool step makes failures diagnosable. Agent evaluation should decompose where the pipeline failed rather than assigning one opaque quality score.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q9. What makes a coding agent different from code completion?

Code completion predicts the next code tokens. A coding agent interacts with repository state over multiple steps:

```text
inspect repository
→ search files
→ understand tests/contracts
→ edit
→ run tests
→ inspect failures
→ revise
→ review diff
```

The defining feature is the feedback loop with tools and the development environment.

Experience with Codex-style workflows should therefore include task specification, repository search, test execution, diff review, and failure recovery—not simply asking an AI to write a standalone function.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

