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

## Extended Interview Q&A

### E1. Why can subtracting a baseline reduce variance without bias?

**Answer:** Because the expectation of score-function gradient times an action-independent baseline is zero: E_a[∇logπ(a|s)b(s)] = b(s)∇Σ_aπ(a|s)=0.

### E2. What is the difference between Q and advantage?

**Answer:** Q measures expected return after taking an action; advantage subtracts the state's average/baseline value and measures relative benefit.

### E3. Why use a target network in DQN?

**Answer:** If the same rapidly changing network defines both prediction and target, the regression target moves too quickly and destabilizes learning.

### E4. Why use replay buffer?

**Answer:** It reuses experience and breaks temporal correlation between consecutive samples, making optimization more data-efficient and closer to i.i.d.

### E5. What does GAE lambda do?

**Answer:** It interpolates between low-variance one-step TD estimates and higher-variance longer-horizon return estimates.

### E6. Why does PPO need old policy probabilities?

**Answer:** The probability ratio measures how much the current policy changed for sampled actions relative to the behavior policy that generated the data.

