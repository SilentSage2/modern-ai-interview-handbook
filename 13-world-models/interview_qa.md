# World Models — Interview Q&A

## Q1. World model vs generative model?

**Short answer:** A world model specifically captures environment dynamics and consequences—often action-conditioned—so it can support prediction and planning; a generic generative model need not model interventions or dynamics.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Q2. Why use latent dynamics?

**Short answer:** High-dimensional observations contain irrelevant detail; latent dynamics can focus capacity on predictive, decision-relevant state.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Practice rule

For each answer prepare three versions:
- **30 sec:** concise definition + core intuition.
- **2 min:** equation/architecture + tradeoff.
- **5 min:** implementation, failure cases, and comparison with alternatives.
