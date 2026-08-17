# Modern AI Interview Handbook

A structured study, interview, and hands-on reference for modern AI/ML roles.

This repository is designed around a **complete job-skill map**, not only topics that are currently unfamiliar. Strong areas such as ML, generative modeling, and diffusion stay in the curriculum as fast-review/interview material, while newer areas receive deeper hands-on work.

## Goals

1. Refresh core ML/DL knowledge for interviews.
2. Build rigorous understanding of Transformers, ViTs, LLMs, VLMs, RL, world models, and agents.
3. Convert "familiar with" into **hands-on experience** through small reproducible projects.
4. Cover AI engineering: GPU systems, distributed training, TensorRT, serving, profiling, latency and throughput.
5. Maintain reusable interview Q&A and primary references.

## Repository Map

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

## Recommended Study Order

**Fast review first:** 01–06  
**Core gap-filling:** 07–11  
**Decision/intelligence:** 12–14  
**Systems/deployment:** 15–16

For every module use the same loop:

> Concepts → equations/intuition → interview Q&A → hands-on → explain it aloud → update notes.

## Hands-on portfolio targets

See [`projects/`](projects/):

1. **ViT/VLM + LoRA Fine-Tuning**
2. **Scientific RAG + Tool-Using Agent**
3. **RL + Latent World Model**
4. **PyTorch → TensorRT Inference Benchmark**

## Suggested skill labels

- **Strong / Review** — already used in research; review for interview language and breadth.
- **Priority** — know parts, but build systematic understanding.
- **Priority + Hands-on** — do not stop at theory; produce a reproducible project.

## How to use this repo

- Put personal notes under `notes/`.
- Add solved interview questions to each module's `interview_qa.md`.
- Add implementation snippets and experiments under `labs/`.
- Keep references primary whenever possible: original papers, official docs, and canonical books.
- Track progress in [`00-roadmap/progress.md`](00-roadmap/progress.md).
