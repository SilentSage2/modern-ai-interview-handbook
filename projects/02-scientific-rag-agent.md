# Project 2 — Scientific RAG + Tool-Using Agent

## Goal
Build an agent that can inspect experiment logs/configs/papers and answer research workflow questions.

## Architecture
User query → retrieval → LLM → tool decision → Python/log/config tool → observation → next action → final answer.

## Minimum tools
- search experiment metadata;
- read training logs;
- run a Python analysis;
- retrieve a paper/note.

## Evaluation
Task success, retrieval accuracy, tool-call correctness, latency/cost, common failure modes.
