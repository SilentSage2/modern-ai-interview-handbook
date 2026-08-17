# Reinforcement Learning — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. What is an MDP and why is the Markov property important?

A Markov Decision Process (MDP) contains states, actions, transition dynamics, rewards, and a discount factor. The Markov property says the current state contains all history needed to predict the next-state distribution under an action:

```math
p(s_{t+1}|s_{0:t},a_{0:t})
=
p(s_{t+1}|s_t,a_t).
```

**Why it matters.** Bellman recursion assumes the state is sufficient. If the raw observation is only partial, the policy or world model may need history or a learned latent state such as:

```math
z_t
=
f(o_{0:t},a_{0:t-1}).
```

This explicit range notation is used in the handbook because it is both mathematically unambiguous and more robust in GitHub rendering.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. V, Q, and advantage: what does each quantity mean?

State value is expected return from state `s` under policy `pi`:

```math
V^\pi(s)
=
\mathbb E_\pi[G_t|s_t=s].
```

Action value asks what happens if action `a` is chosen first:

```math
Q^\pi(s,a)
=
\mathbb E_\pi[G_t|s_t=s,a_t=a].
```

Advantage compares that action with the policy's normal baseline:

```math
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s).
```

**Key idea.** Advantage tells the actor whether the sampled action was better or worse than expected for that state. Subtracting the state baseline also reduces variance in policy-gradient estimates.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. Explain Bellman recursion intuitively.

The return decomposes into immediate reward plus discounted future return:

```math
G_t
=
r_t+\gamma G_{t+1}.
```

Taking expectation gives a Bellman equation:

```math
V^\pi(s)
=
\mathbb E[
r_t+\gamma V^\pi(s_{t+1})
].
```

**Key idea.** A long-horizon decision problem can be written recursively as a one-step reward plus the value of what remains.

Dynamic programming, Temporal-Difference learning, Q-learning, and critic training all exploit this recursion. The formula is the mathematical reason bootstrapping from a next-state value estimate is possible.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. Q-learning versus policy gradient: what is fundamentally different?

Q-learning estimates how good state–action pairs are. The policy can be derived by choosing actions with the largest Q value. It is naturally off-policy because the target can use a greedy max even when data were collected by another behavior policy.

Policy-gradient methods directly parameterize an action distribution `pi_theta(a|s)` and differentiate expected return with respect to the policy parameters.

**Tradeoff.**
- Value-based methods reuse data efficiently and are natural for discrete actions.
- Policy gradients handle stochastic and continuous actions naturally but often have high-variance estimates and use on-policy data.

Actor–critic combines a policy actor with a learned value critic.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. Derive the policy-gradient idea.

Trajectory probability depends on the policy's action probabilities. Using the log-derivative trick gives:

```math
\nabla_\theta J
=
\mathbb E
\left[
\sum_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
G_t
\right].
```

Subtracting an action-independent baseline such as `V(s_t)` preserves the expectation but reduces variance, producing an advantage-weighted gradient.

**Intuition.**
- Positive advantage: increase the sampled action's log probability.
- Negative advantage: decrease it.

This turns delayed reward into a gradient signal over action probabilities without differentiating through the environment transition itself.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. What is GAE and why use it?

Generalized Advantage Estimation (GAE) combines multiple Temporal-Difference residuals:

```math
\hat A_t
=
\sum_{\ell=0}^{\infty}
(\gamma\lambda)^\ell
\delta_{t+\ell}.
```

The parameter `lambda` controls a bias–variance tradeoff.

- Smaller lambda relies more on bootstrapped value estimates: lower variance, more bias.
- Larger lambda approaches longer Monte Carlo returns: lower bias, higher variance.

GAE is popular with PPO because policy-gradient optimization benefits from a low-variance but reasonably accurate advantage signal.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q7. What is PPO clipping actually doing?

Proximal Policy Optimization (PPO) compares the probability of a sampled action under the new and old policies:

```math
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t|s_t)
}.
```

The clipped objective removes additional incentive once this ratio moves too far from one.

**Intuition.**
- Positive advantage: increase action probability, but not excessively.
- Negative advantage: decrease it, but not excessively.

This helps avoid destructive policy updates and keeps recently collected on-policy data useful for a few minibatch epochs. PPO is a practical update-control mechanism, not a perfect trust-region guarantee.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q8. On-policy versus off-policy: why does it matter?

On-policy algorithms learn from trajectories generated by the current or very recent policy. PPO is primarily on-policy.

Off-policy algorithms such as Q-learning can learn from data generated by another behavior policy and therefore reuse replay more aggressively.

**Tradeoff.**
- Off-policy learning is data efficient but faces behavior/target distribution mismatch.
- On-policy learning avoids severe mismatch but requires fresh interaction.

This distinction matters in RLHF, offline RL, robotics, and world-model settings where collecting new data can be expensive.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

