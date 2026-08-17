# RAG, AI Agents & Coding Agents — Interview Q&A

## Q1. RAG vs fine-tuning?

**Short answer:** RAG changes model context with retrieved external information; fine-tuning changes model parameters/behavior. They solve overlapping but distinct problems and can be combined.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Q2. What makes an LLM system agentic?

**Short answer:** A closed loop in which the model observes state, chooses actions/tools, receives new observations, updates state, and continues until a goal or stopping condition.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Practice rule

For each answer prepare three versions:
- **30 sec:** concise definition + core intuition.
- **2 min:** equation/architecture + tradeoff.
- **5 min:** implementation, failure cases, and comparison with alternatives.

## Extended Interview Q&A

### E1. Why add a reranker after vector search?

**Answer:** Fast embedding retrieval aims for high recall; a more expensive cross-encoder/reranker can improve relevance precision on a smaller candidate set.

### E2. What is the main failure mode of naive RAG?

**Answer:** Retrieval failure: the generator cannot ground on evidence it never receives. Generation quality cannot compensate for missing relevant context.

### E3. Why are agent systems harder than chatbots?

**Answer:** Errors compound over multiple decisions, tools mutate external state, tool results may be noisy, and long-horizon tasks require state and recovery.

### E4. When should an agent ask for human approval?

**Answer:** Before high-impact, irreversible, privileged, or ambiguous actions such as sending messages, deleting data, spending money, or changing production systems.

### E5. How do you evaluate tool use?

**Answer:** Measure tool selection accuracy, argument correctness, execution success, recovery from errors, and end-task success—not only final-answer fluency.

### E6. What makes coding agents agentic?

**Answer:** They repeatedly inspect repository state, edit files, run tests/tools, observe results, and revise rather than generating one static code completion.


## Whiteboard / drill questions

- Design an evaluation that separates retrieval failure from generation failure.
- Why should approval happen after tool arguments are generated but before execution?
- Manager-agent vs handoff architecture: compare them.
- What makes a coding agent different from code completion?
- How do you prevent an agent from looping indefinitely?


<!-- ADVANCED_QA_START -->
## Advanced / system follow-ups

### A1. Why is final-answer accuracy not enough for agent evaluation?

**Answer:** Two agents can reach the same answer with very different tool errors, costs, unsafe attempts, or brittle paths. Step-level and end-task metrics are both needed.

### A2. What is the difference between an agent and a deterministic workflow?

**Answer:** A workflow has predefined transitions; an agent selects actions dynamically from state/context. Good systems often use deterministic control around agentic decision points.

### A3. Why can more tools reduce agent reliability?

**Answer:** A larger action space makes routing harder, schemas can overlap, and irrelevant tools increase confusion. Tool sets should be minimal and well described.

### A4. How should a coding agent verify a patch?

**Answer:** Run targeted tests, inspect the diff, then run broader relevant checks. Verification must be external to the model's own confidence.

### A5. Why is tracing important in production agents?

**Answer:** Multi-step failures are otherwise difficult to localize. Tracing shows model calls, tool calls, latency, errors, handoffs, and state transitions needed for debugging/evaluation.

<!-- ADVANCED_QA_END -->
