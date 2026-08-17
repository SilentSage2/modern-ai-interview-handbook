# RAG, AI Agents & Coding Agents

**Status:** Priority + Hands-on

## Why this matters

Agent roles require more than prompting: retrieval, tool use, memory, planning, evaluation, safety, and multi-step execution all matter.

## Learning objectives

- Derive dense-retrieval and contrastive-retriever objectives.
- Understand tool calling, ReAct, memory, planning, and agent evaluation.
- Explain coding-agent workflows and failure modes.

## Terminology and abbreviations

Do not memorize an abbreviation before you understand what object it refers to.

| Term | Full name | Role in this chapter |
|---|---|---|
| **RAG** | Retrieval-Augmented Generation | Retrieves evidence before generation. |
| **ANN** | Approximate Nearest Neighbor | Fast vector retrieval in large embedding stores. |
| **ReAct** | Reason + Act | Alternates reasoning, tool action, and observation. |
| **API** | Application Programming Interface | Programmatic tool/service interface. |
| **SDK** | Software Development Kit | Libraries for building applications against a platform. |

For the repository-wide list, see [`../GLOSSARY.md`](../GLOSSARY.md).


## Big picture and design philosophy

### RAG separates knowledge storage from model parameters

Retrieval supplies external evidence at request time, so knowledge can change without retraining the LLM.

### Dense retrieval learns a semantic coordinate system

Query/document encoders learn vectors where relevant pairs are close. Dot product is only the scoring mechanism; contrastive training creates the geometry.

### An agent is a closed-loop system

The model observes state, chooses a tool/action, receives a result, updates state, and acts again. Reliability therefore depends on control logic, permissions, memory, verification, and stopping conditions.

> **How to read the equations below:** first identify the problem, what each variable represents, why this formulation was chosen, and what tradeoff it introduces. The equation is the precise implementation of the idea—not the idea itself.


## Chapter map

- Embeddings and semantic retrieval
- Chunking, indexing, vector search, reranking
- Retrieval-Augmented Generation (RAG) versus fine-tuning
- Tool/function calling
- ReAct (Reason + Act): reason–act–observe loops
- Agent memory and state
- Planning and task decomposition
- Multi-agent orchestration
- Guardrails, approvals and tool permissions
- Agent evaluation and tracing
- Coding agents / Codex-style repository workflows


---

## Core concepts and theory

### 1. Retrieval-Augmented Generation

Given query $q$, retriever finds documents:


```math
D_k(q)=\{d_1,\ldots,d_k\}.
```


Generator produces:


```math
p(y|q,d_1,\ldots,d_k).
```


RAG separates:
- parametric memory: model weights;
- non-parametric memory: external store.

---

### 2. Dense Retrieval

> **Key idea: learned semantic search.** Traditional keyword search matches words. Dense retrieval instead learns an embedding space in which a query and a relevant document occupy nearby positions even if they do not use exactly the same vocabulary. The important modeling object is therefore the **geometry of the embedding space**; the dot product below is simply the fast scoring rule used after that geometry has been learned.

Let:
- $q$ = the user query;
- $d$ = a document or document chunk;
- $f_q$ = query encoder;
- $f_d$ = document encoder;
- $z_q$ and $z_d$ = learned embedding vectors.

Encode query and document:

```math
z_q=f_q(q),
\qquad
z_d=f_d(d).
```

A simple relevance score is the dot product:

```math
s(q,d)=z_q^\top z_d.
```

If vectors are normalized, this is equivalent to cosine similarity up to scale. Larger similarity should mean “more semantically relevant.”

Retrieve the top $k$ documents:

```math
D_k(q)
=
\operatorname{TopK}_{d}\;s(q,d).
```

Here **$k$ is simply the number of retrieved candidates**. For example, `top-k = 20` means the retriever returns the 20 chunks with the highest relevance scores.

Why not send the whole database to the LLM? Because context is limited and expensive. Retrieval is a **candidate-selection stage** that tries to maximize the chance that the needed evidence is present in a small context budget.

---

### 3. Contrastive Retriever Training

> **Key idea: turn retrieval into a classification problem.** The embedding space does not become semantic automatically. It is trained so a query scores its relevant document above competing documents.

For one query, let:
- $d^+$ = a relevant/positive document;
- $d_j^-$ = negative candidate documents;
- $\tau$ = temperature controlling how sharp the competition is.

A common contrastive objective is:

```math
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
```

The numerator is the positive document's score. The denominator contains the positive plus competing negatives. Minimizing the loss increases the positive's relative probability.

The **design choices around the equation** are often more important than the equation itself:

1. **Positive construction:** what counts as truly relevant?
2. **Negative sampling:** random negatives are often too easy; hard negatives teach fine distinctions.
3. **False negatives:** two documents may both be relevant even if only one is labeled positive.
4. **Encoder sharing:** query and document encoders may share weights or specialize.
5. **Temperature $\tau$:** smaller values make score differences matter more strongly.

This is the same broad contrastive-learning principle used in CLIP and self-supervised representation learning: learn a space in which the intended correspondences are geometrically easy to retrieve.


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


```math
\text{fast retriever}
\rightarrow
\text{top-}K
\rightarrow
\text{expensive reranker}
\rightarrow
\text{top-}k.
```


Retriever maximizes recall; reranker improves precision.

---

### 6. Tool Calling

Model receives a tool schema:


```math
T_i=(name_i,\text{arguments}_i,\text{description}_i).
```


Instead of free-form text, it can emit structured action


```math
a_t=(T_i,\text{args}).
```


Runtime executes action and returns observation $o_{t+1}$.

Important:
the LLM chooses the action; the application/runtime performs it.

---

### 7. Agent as Closed Loop

A useful abstraction:


```math
s_t
\overset{\pi_{\rm LLM}}{\longrightarrow}
a_t
\overset{\rm environment}{\longrightarrow}
o_{t+1}
\rightarrow
s_{t+1}.
```


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


```math
\text{reason}
\rightarrow
\text{act}
\rightarrow
\text{observe}
\rightarrow
\text{reason}.
```


Why useful:
the model can update its beliefs using external evidence rather than relying entirely on pretrained memory.

---

### 9. Planning

Explicit decomposition:


```math
G
\rightarrow
(g_1,g_2,\ldots,g_n).
```


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


```math
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
```


The hard parts are:
- repository-scale context selection;
- safe edits;
- test-driven verification;
- tool execution;
- maintaining state across multi-step tasks.

<!-- DEEP_DIVE_START -->
## Deep dive I: build a RAG pipeline from first principles

### Offline indexing

1. ingest documents;
2. clean/parse;
3. chunk;
4. embed each chunk;
5. store embedding + metadata.


```math
d_i
\rightarrow
z_i=f(d_i).
```


### Online query


```math
q\rightarrow z_q=f(q).
```


Retrieve:


```math
i_1,\ldots,i_k
=
\operatorname{TopK}
(z_q^\top z_i).
```


Optionally rerank and place selected text into the model context.

The most important insight:

> RAG is a pipeline; model quality depends on every stage before generation.

---

## Deep dive II: retrieval metrics

If relevant set is $R_q$ and retrieved top-$k$ set is $A_k$:

Recall@k:


```math
\frac{|A_k\cap R_q|}{|R_q|}.
```


Precision@k:


```math
\frac{|A_k\cap R_q|}{k}.
```


For RAG, recall is often critical. If supporting evidence is not retrieved, the generator cannot use it.

---

## Deep dive III: hybrid retrieval

Dense embeddings capture semantic similarity.

Sparse lexical methods capture exact words, identifiers, rare technical strings.

Hybrid score:


```math
s(d,q)
=
\lambda s_{\rm dense}
+
(1-\lambda)s_{\rm sparse}.
```


This is often valuable for:
- code;
- product IDs;
- paper acronyms;
- error messages.

---

## Deep dive IV: tool calling as constrained action space

Without tools, LLM outputs text tokens.

With tool schemas, action space includes structured calls:

```json
{
  "tool": "search_logs",
  "arguments": {"experiment_id": 37}
}
```

Runtime validates schema and executes.

This separates:
- **decision**: model;
- **execution**: external runtime.

That boundary is essential for reliability and security.

---

## Deep dive V: state machine view of an agent

A production agent can be modeled as states:

```text
RECEIVE
→ PLAN
→ CALL_TOOL
→ OBSERVE
→ (CALL_TOOL | ANSWER | REQUEST_APPROVAL | FAIL)
```

Thinking in explicit states helps prevent accidental infinite loops and unclear termination.

Useful termination conditions:
- goal satisfied;
- maximum steps;
- unrecoverable tool error;
- approval denied;
- insufficient information.

---

## Deep dive VI: memory vs context

Context window is transient model input.

Memory is an external mechanism deciding what prior information should be restored into context.

Long-term memory pipeline:


```math
\text{event}
\rightarrow
\text{store}
\rightarrow
\text{retrieve later}
\rightarrow
\text{inject into context}.
```


This is architecturally similar to RAG but personalized/task-state oriented.

---

## Deep dive VII: agent evaluation by decomposition

Suppose end-task success is poor. Diagnose:

### Retrieval failure?
Relevant file never found.

### Reasoning/routing failure?
Correct tool available but model chose wrong action.

### Argument failure?
Correct tool but malformed inputs.

### Tool failure?
External API errored.

### State failure?
Result was obtained but later forgotten/misinterpreted.

### Verification failure?
Agent edited code but did not run tests.

This decomposition is much more actionable than a single “agent quality” score.

---

## Deep dive VIII: human approval design

High-impact action:

```text
agent proposes action
→ render exact planned side effect
→ human approves/rejects
→ runtime executes
```

Approval should occur **after** arguments are concrete enough to inspect but **before** irreversible execution.

Examples:
- send email;
- merge PR;
- delete cloud data;
- purchase item;
- deploy production change.

---

## Deep dive IX: multi-agent patterns

### Manager pattern

```text
manager
├→ search specialist
├→ coding specialist
└→ reviewer
```

Manager keeps control and calls specialists as tools.

### Handoff pattern

One agent transfers control to another specialist.

Tradeoff:
- manager gives centralized orchestration;
- handoff can simplify specialized conversations;
- both increase context/routing complexity.

---

## Deep dive X: coding agent workflow

A robust code agent should not jump directly from issue to edit.

```text
1. inspect repository
2. identify relevant files
3. understand tests/contracts
4. make minimal patch
5. run targeted tests
6. inspect failure
7. iterate
8. run broader tests
9. summarize diff
```

This resembles a software engineer's feedback loop.

### Why tests matter

LLM-generated code can look locally plausible while violating repository behavior. Tests provide an external objective signal.

Coding agent performance therefore depends heavily on:
- shell access;
- repository search;
- test execution;
- patch inspection;
- version-control awareness.

---

## Scientific experiment agent example

User asks:

> Why did experiment 37 have worse validation loss?

Agent loop:

1. retrieve config 37;
2. retrieve nearest successful experiments;
3. parse training logs;
4. compare LR, batch, dataset version;
5. run Python plot/statistics;
6. identify likely divergence point;
7. cite evidence;
8. recommend one controlled next experiment.

This one project demonstrates:
- RAG;
- tool use;
- structured state;
- Python execution;
- agent evaluation.

---

## Minimal tool-loop pseudocode

```python
state = initial_user_request

for step in range(MAX_STEPS):
    action = model.decide(state, tools)

    if action.type == "final":
        return action.answer

    if action.type == "tool":
        result = execute(action.tool, action.args)
        state = update(state, action, result)

raise RuntimeError("step limit reached")
```

Production layers add:
- permission checks;
- schemas;
- retries;
- tracing;
- cost limits;
- human approval;
- tool-output sanitization.

---

## Current platform concepts worth recognizing

Modern agent SDKs commonly expose primitives such as:
- tools;
- handoffs / agents-as-tools;
- sessions/state;
- guardrails;
- human approval;
- tracing.

Do not memorize one framework API. Understand the architecture these abstractions represent.
<!-- DEEP_DIVE_END -->

<!-- SECOND_DEEP_DIVE_START -->
## RAG architecture case study: scientific literature assistant

### Corpus
PDFs/papers/notes.

### Index
Each chunk stores:
- text;
- title;
- section;
- publication metadata;
- embedding.

### Retrieval
Query embedding → top candidates.

### Reranking
Cross-encoder or LLM evaluates relevance.

### Generation
Prompt includes evidence with source labels.

### Verification
Check whether answer claims are supported by retrieved passages.

This is a much stronger architecture than “put PDFs in a vector database.”

---

## Chunk overlap

Suppose chunk length $L$, overlap $O$.

Stride:


```math
S=L-O.
```


Overlap reduces risk that a concept split at boundary loses context, but increases:
- index size;
- duplicate retrieval;
- token redundancy.

Again, chunking is a bias/compute tradeoff.

---

## Query rewriting

User query may be underspecified.

Retriever query can be transformed:


```math
q\rightarrow \tilde q.
```


Examples:
- expand acronym;
- include entity names;
- generate multiple retrieval queries.

Multi-query retrieval:


```math
D=
\bigcup_{j=1}^m
\mathrm{Retrieve}(\tilde q_j).
```


This can improve recall but increases cost and duplicate handling.

---

## Tool-selection failure example

Available tools:
- `search_web`
- `query_database`
- `run_python`

User asks for average of internal experiment scores.

Wrong path:
model uses web search.

Correct routing:
database → fetch values → Python calculation.

Agent evaluation should explicitly score **tool routing**, not just final text.

---

## Structured tool schemas

Tool definition:

```python
def get_experiment(
    experiment_id: int,
    include_logs: bool = False
) -> Experiment:
    ...
```

Schema helps the model understand:
- required fields;
- types;
- semantics.

Runtime should validate:
- argument types;
- authorization;
- allowed values.

Never rely on natural-language compliance alone for dangerous side effects.

---

## Retry policy

Tool errors may be:
- transient;
- malformed argument;
- permission denied;
- missing resource.

Treat differently.

Example:

```text
timeout → retry with backoff
bad argument → model repairs call
permission denied → stop/request access
not found → search alternative
```

Blind retry loops waste cost and can amplify side effects.

---

## Agent observability

For each run, log a trace:

```text
user input
→ model turn
→ tool call
→ tool result
→ model turn
→ handoff
→ final output
```

Useful metrics:
- model latency;
- tool latency;
- token usage;
- tool failure rate;
- step count;
- approval rate;
- end-task success.

This is necessary for debugging long multi-step behavior.

---

## Guardrail placement

Different checks belong at different boundaries.

### Input
Is the request allowed/well-formed?

### Pre-tool
Is this action permitted and safe?

### Post-tool
Did the tool return sensitive/malformed content?

### Final output
Does response satisfy policy/format/grounding requirements?

A single final-output filter cannot protect against a dangerous tool that already executed.

---

## Agent vs workflow

If task is deterministic:

```text
A → B → C
```

ordinary code/workflow is often better.

Use an agent when the system must choose among uncertain next actions based on unstructured context.

Good design often combines both:

```text
deterministic outer workflow
+ agent at ambiguous decision points.
```

This is a strong answer to “when should you not use an agent?”

---

## Codex/coding-agent practice tasks

Use an existing repository and ask the coding agent to:

1. explain architecture;
2. locate a bug;
3. write a regression test;
4. implement minimal fix;
5. run tests;
6. refactor repeated code;
7. benchmark performance;
8. summarize diff.

Your learning goal is not to outsource coding. It is to learn:
- effective task specification;
- repository-level context;
- verification;
- review of AI-generated changes.

---

## Mini-project evaluation table

For a scientific agent, create tasks such as:

| Task | Expected evidence/tool | Success criterion |
|---|---|---|
| Compare run 21 vs 37 | logs + config | correct parameter differences |
| Explain failure | logs + plot tool | identifies real divergence |
| Find related method | paper retrieval | relevant cited source |
| Propose next run | state + evidence | one controlled change |

Report both task success and failure taxonomy.
<!-- SECOND_DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Evaluate retrieval separately from generation.
- Treat tool schemas, permissions, and execution state as part of the system.
- Agent success should be measured at the task level, not only by how fluent intermediate reasoning appears.

---

## Hands-on / practice

### Level 1 — Reproduce
Implement or run a canonical example that demonstrates the central idea.

### Level 2 — Compare
Create at least one controlled comparison (baseline vs method, accuracy vs compute, or full vs efficient version).

### Level 3 — Explain
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
