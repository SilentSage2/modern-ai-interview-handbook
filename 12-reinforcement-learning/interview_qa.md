# Reinforcement Learning — Interview Q&A

## Q1. What is the advantage function?

**Short answer:** A(s,a)=Q(s,a)-V(s); it measures how much better an action is than the state's baseline value.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Q2. Why does PPO clip the policy ratio?

**Short answer:** To discourage overly large policy updates and improve optimization stability while retaining a simple first-order objective.

**Follow-ups to prepare:** Why? When does this fail? What is the computational cost? What alternative would you use?

## Practice rule

For each answer prepare three versions:
- **30 sec:** concise definition + core intuition.
- **2 min:** equation/architecture + tradeoff.
- **5 min:** implementation, failure cases, and comparison with alternatives.
