# v6 — Textbook Completion Pass

This version addresses the four issues identified during manual review.

## 1. Formula reliability
- Uses GitHub fenced `math` blocks for display equations.
- Replaces fragile history shorthand with explicit index ranges such as `o_{0:t}` and `a_{0:t-1}`.
- Adds automated checks for fence closure, brace balance, legacy delimiters, and hidden control characters.

## 2. Abbreviations and notation
- Adds a repository-wide `GLOSSARY.md`.
- Adds `Terminology and abbreviations` to every technical chapter.
- Expands common acronyms in chapter maps where they first appear.

## 3. Interview Q&A
- Replaces generic answer outlines with full answers in all 16 modules.
- Answers include core idea, mechanism/equation, design rationale, tradeoffs, and common follow-ups.

## 4. Explanation before equations
- Adds `Big picture and design philosophy` to every chapter.
- Rewrites dense retrieval / contrastive retriever training as a worked example of idea → notation → equation → design choices.
- Establishes the handbook rule that equations should formalize an idea that has already been explained.
