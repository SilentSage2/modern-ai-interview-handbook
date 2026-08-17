# Modern AI Interview Handbook

A book-style study and interview reference for modern AI/ML roles.

The repository keeps **complete coverage** of common job requirements—including topics already strong in your background—while going deeper on Transformers, ViTs, foundation models, fine-tuning, VLMs, RL, world models, agents, and AI systems.

## How the repository is organized

Each technical module contains only three files:

```text
module/
├── README.md          # Complete chapter: concepts + derivations + intuition + implementation + practice
├── interview_qa.md    # Interview questions and model answers
└── references.md      # Papers, books, and official documentation
```

The idea is simple: **open `README.md` and read the chapter continuously**, rather than jumping among many small files.

## Curriculum

| # | Module | Status |
|---|---|---|
| 01 | [Machine Learning Foundations](01-ml-foundations/) | Strong / Review |
| 02 | [Deep Learning Core](02-deep-learning-core/) | Strong / Review |
| 03 | [CNN & Computer Vision](03-computer-vision/) | Strong / Review |
| 04 | [Representation & Self-Supervised Learning](04-representation-learning/) | Strong + Fill Gaps |
| 05 | [Generative Models](05-generative-models/) | Strong / Review |
| 06 | [Diffusion & Score-Based Models](06-diffusion-score-models/) | Strong |
| 07 | [Transformers](07-transformers/) | Priority |
| 08 | [ViT & Vision Foundation Models](08-vit-vision-foundation-models/) | Priority |
| 09 | [LLMs & Foundation Models](09-llm-foundation-models/) | Priority |
| 10 | [Fine-Tuning, LoRA & PEFT](10-finetuning-peft/) | Priority + Hands-on |
| 11 | [VLM & Multimodal Foundation Models](11-vlm-multimodal/) | Priority + Hands-on |
| 12 | [Reinforcement Learning](12-reinforcement-learning/) | Priority |
| 13 | [World Models](13-world-models/) | Priority + Hands-on |
| 14 | [RAG, AI Agents & Coding Agents](14-rag-agents-coding-agents/) | Priority + Hands-on |
| 15 | [GPU, Distributed Training & AI Systems](15-ai-systems-distributed/) | Priority |
| 16 | [Inference, TensorRT & Deployment](16-inference-deployment/) | Priority + Hands-on |

## Suggested study path

### Part I — Core ML / DL
01–06 are primarily rapid-review chapters:
- ML foundations
- deep learning
- computer vision
- representation learning
- generative models
- diffusion / score models

### Part II — Foundation-model core
07–11 deserve deeper study:
- Transformers
- ViT / vision foundation models
- LLMs / foundation models
- fine-tuning / LoRA / PEFT
- VLM / multimodal AI

### Part III — Decision and agentic AI
12–14:
- reinforcement learning
- world models
- RAG / agents / coding agents

### Part IV — AI systems
15–16:
- distributed/GPU systems
- inference / TensorRT / deployment

## Supporting material

- [`00-roadmap/`](00-roadmap/) — study order and progress tracking
- [`cheatsheets/`](cheatsheets/) — rapid equation/system review
- [`projects/`](projects/) — hands-on portfolio projects
- [`labs/`](labs/) — runnable code and experiments
- [`notes/`](notes/) — personal notes and interview mistakes

## Definition of “interview ready”

A topic is ready when you can do all four:

1. **Concept:** explain what problem it solves.
2. **Theory:** derive or justify its important equations.
3. **Implementation:** describe or code a representative implementation.
4. **Systems/tradeoff:** explain memory, compute, data, latency, robustness, or scaling consequences.


## How deep each chapter goes

The chapter README is intended to function like a compact textbook chapter, not a glossary. Priority chapters include:

- formal definitions and assumptions;
- derivations of central equations;
- tensor shapes and parameter/memory scaling;
- worked numerical examples;
- minimal PyTorch-style implementation walkthroughs;
- architecture comparisons;
- common misconceptions and failure modes;
- system implications;
- hands-on experiments and interview framing.

A good study loop is:

\[
\text{read}
\rightarrow
\text{derive}
\rightarrow
\text{implement}
\rightarrow
\text{explain aloud}
\rightarrow
\text{answer follow-up questions}.
\]
