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


## Whiteboard / drill questions

- Derive policy gradient from trajectory probability.
- Why does an action-independent baseline preserve unbiasedness?
- Monte Carlo vs TD: explain bias and variance.
- Why is PPO on-policy?
- What failure happens if PPO reuses old rollouts too long?


<!-- ADVANCED_QA_START -->
## Advanced / system follow-ups

### A1. Why is the Markov assumption important for Bellman equations?

**Answer:** Bellman recursion assumes the current state is sufficient to predict future return distribution under action/policy. If relevant history is missing, value functions over the observation alone may be inconsistent.

### A2. Why can reward shaping be dangerous?

**Answer:** The agent optimizes the specified numerical reward, which may contain loopholes. A shaped proxy can be easier to exploit than the intended task objective.

### A3. Why does off-policy learning need care under distribution shift?

**Answer:** The learner evaluates or improves actions/states that may be weakly represented under behavior data. Estimation errors can be amplified, especially through bootstrapping/max operations.

### A4. What is the role of the critic in actor-critic?

**Answer:** The critic estimates value/advantage to reduce variance and provide a learning signal for the actor. It does not choose the action directly in the standard actor-critic decomposition.

### A5. Why are multiple seeds important in RL?

**Answer:** Training and environment interaction are stochastic and return variance can be high. A single seed can give a misleading conclusion about algorithm quality.

<!-- ADVANCED_QA_END -->
