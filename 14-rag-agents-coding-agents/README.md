# RAG, AI Agents & Coding Agents

**Status:** Priority + Hands-on

## Why this matters

Agent roles require more than prompting: retrieval, tool use, memory, planning, evaluation, safety, and multi-step execution all matter.

## Learning objectives

- Derive dense-retrieval and contrastive-retriever objectives.
- Understand tool calling, ReAct, memory, planning, and agent evaluation.
- Explain coding-agent workflows and failure modes.

## Chapter map

- Embeddings and semantic retrieval
- Chunking, indexing, vector search, reranking
- RAG vs fine-tuning
- Tool/function calling
- ReAct: reason–act–observe loops
- Agent memory and state
- Planning and task decomposition
- Multi-agent orchestration
- Guardrails, approvals and tool permissions
- Agent evaluation and tracing
- Coding agents / Codex-style repository workflows


---

## Core concepts and theory

### 1. Retrieval-Augmented Generation

Given query \(q\), retriever finds documents:

\[
D_k(q)=\{d_1,\ldots,d_k\}.
\]

Generator produces:

\[
p(y|q,d_1,\ldots,d_k).
\]

RAG separates:
- parametric memory: model weights;
- non-parametric memory: external store.

---

### 2. Dense Retrieval

Embed query and document:

\[
z_q=f_q(q),
\qquad
z_d=f_d(d).
\]

Similarity:

\[
s(q,d)=z_q^\top z_d
\]

or cosine similarity.

Retrieve top-\(k\):

\[
D_k(q)
=
\mathrm{TopK}_d\,s(q,d).
\]

---

### 3. Contrastive Retriever Training

For positive document \(d^+\):

\[
\mathcal L
=
-\log
\frac{
e^{s(q,d^+)/\tau}
}{
e^{s(q,d^+)/\tau}
+
\sum_j e^{s(q,d_j^-)/\tau}
}.
\]

This is the same broad contrastive-learning principle used in CLIP.

---

### 4. Chunking Tradeoff

Small chunks:
- precise retrieval;
- less unrelated content;
- may lose context.

Large chunks:
- better context;
- lower retrieval granularity;
- consume more context window.

This is an engineering hyperparameter, not a cosmetic preprocessing choice.

---

### 5. Reranking

Two-stage retrieval:

\[
\text{fast retriever}
\rightarrow
\text{top-}K
\rightarrow
\text{expensive reranker}
\rightarrow
\text{top-}k.
\]

Retriever maximizes recall; reranker improves precision.

---

### 6. Tool Calling

Model receives a tool schema:

\[
T_i=(name_i,\text{arguments}_i,\text{description}_i).
\]

Instead of free-form text, it can emit structured action

\[
a_t=(T_i,\text{args}).
\]

Runtime executes action and returns observation \(o_{t+1}\).

Important:
the LLM chooses the action; the application/runtime performs it.

---

### 7. Agent as Closed Loop

A useful abstraction:

\[
s_t
\overset{\pi_{\rm LLM}}{\longrightarrow}
a_t
\overset{\rm environment}{\longrightarrow}
o_{t+1}
\rightarrow
s_{t+1}.
\]

State may include:
- user goal;
- conversation;
- tool outputs;
- retrieved documents;
- memory;
- plan;
- execution status.

This makes agent systems conceptually close to sequential decision processes.

---

### 8. ReAct

Alternates reasoning and external interaction:

\[
\text{reason}
\rightarrow
\text{act}
\rightarrow
\text{observe}
\rightarrow
\text{reason}.
\]

Why useful:
the model can update its beliefs using external evidence rather than relying entirely on pretrained memory.

---

### 9. Planning

Explicit decomposition:

\[
G
\rightarrow
(g_1,g_2,\ldots,g_n).
\]

But planning adds risks:
- plans can become stale;
- unnecessary steps increase latency/cost;
- model may overcommit to an early mistaken plan.

Good agents often mix planning and replanning.

---

### 10. Memory

#### Working memory
Current context/state.

#### Episodic memory
Past interaction/task events.

#### Semantic memory
Retrieved facts/knowledge.

Implementation may use:
- vector stores;
- relational databases;
- summaries;
- structured state.

Memory is largely a system design problem, not magical internal persistence.

---

### 11. Agent Evaluation

Evaluate layers separately:

#### Model quality
Does it reason/choose appropriately?

#### Retrieval
Recall@k, precision, grounding.

#### Tool use
Correct tool and arguments.

#### End task
Success rate.

#### Efficiency
Latency, steps, tokens, cost.

#### Safety
Permission violations, unsafe actions, approval bypasses.

---

### 12. Coding Agents

Repository coding agent loop:

\[
\text{issue}
\rightarrow
\text{search/read code}
\rightarrow
\text{edit}
\rightarrow
\text{run tests}
\rightarrow
\text{inspect failures}
\rightarrow
\text{iterate}.
\]

The hard parts are:
- repository-scale context selection;
- safe edits;
- test-driven verification;
- tool execution;
- maintaining state across multi-step tasks.

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Evaluate retrieval separately from generation.
- Treat tool schemas, permissions, and execution state as part of the system.
- Agent success should be measured at the task level, not only by how fluent intermediate reasoning appears.

---

## Hands-on / practice

## Level 1 — Reproduce
Implement or run a canonical example that demonstrates the central idea.

## Level 2 — Compare
Create at least one controlled comparison (baseline vs method, accuracy vs compute, or full vs efficient version).

## Level 3 — Explain
Write:
- what you changed;
- why it worked or failed;
- GPU memory / runtime where relevant;
- one figure or table;
- a 2-minute interview explanation.

## Deliverables
- [ ] runnable code
- [ ] README with commands
- [ ] experiment configuration
- [ ] quantitative result
- [ ] failure-case notes
- [ ] interview-ready project summary

---

## Interview readiness checklist

Before marking this chapter ready, make sure you can:

- explain the main idea without notes;
- write the important equations from memory;
- discuss at least one design tradeoff;
- compare the method with its nearest alternative;
- identify at least one failure mode;
- connect the theory to a real implementation or project.

For dedicated interview questions, see [`interview_qa.md`](interview_qa.md).  
For papers and official documentation, see [`references.md`](references.md).
