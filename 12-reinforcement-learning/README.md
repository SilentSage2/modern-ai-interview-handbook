# Reinforcement Learning

**Status:** Priority

## Why this matters

RL is the theoretical foundation for PPO/RLHF, agents, world models, and embodied decision-making.

## Learning objectives

- Derive Bellman equations and distinguish V, Q, and advantage.
- Derive policy gradients and actor–critic intuition.
- Explain GAE, PPO clipping, and model-free vs model-based RL.

## Chapter map

- MDP: state, action, transition, reward, discount
- Return, value V(s), action-value Q(s,a), advantage A(s,a)
- Bellman equations
- Q-learning and DQN
- Policy gradients and REINFORCE
- Actor–critic
- Generalized Advantage Estimation (GAE)
- PPO and clipped objectives
- Model-free vs model-based RL
- Offline RL basics
- Connection from PPO to RLHF


---

## Core concepts and theory

### 1. Markov Decision Process

An MDP is

\[
(\mathcal S,\mathcal A,P,R,\gamma).
\]

Transition:

\[
P(s'|s,a).
\]

Policy:

\[
\pi(a|s).
\]

Discounted return:

\[
G_t
=
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}.
\]

Goal:

\[
\max_\pi
J(\pi)
=
\mathbb E_\pi[G_0].
\]

---

### 2. Value Functions

State value:

\[
V^\pi(s)
=
\mathbb E_\pi[G_t|s_t=s].
\]

Action value:

\[
Q^\pi(s,a)
=
\mathbb E_\pi[G_t|s_t=s,a_t=a].
\]

Advantage:

\[
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s).
\]

---

### 3. Bellman Expectation Equation

Start with

\[
G_t=r_t+\gamma G_{t+1}.
\]

Therefore

\[
V^\pi(s)
=
\mathbb E_{a\sim\pi,s'\sim P}
[
r(s,a)+\gamma V^\pi(s')
].
\]

Similarly:

\[
Q^\pi(s,a)
=
r(s,a)
+
\gamma
\mathbb E_{s',a'}
[
Q^\pi(s',a')
].
\]

---

### 4. Bellman Optimality

Optimal action value:

\[
Q^*(s,a)
=
r(s,a)
+
\gamma
\mathbb E_{s'}
\left[
\max_{a'}Q^*(s',a')
\right].
\]

Optimal policy:

\[
\pi^*(s)=\arg\max_a Q^*(s,a).
\]

---

### 5. Q-Learning

TD target:

\[
y_t
=
r_t
+
\gamma\max_{a'}Q_{\theta^-}(s_{t+1},a').
\]

Loss:

\[
\mathcal L
=
(Q_\theta(s_t,a_t)-y_t)^2.
\]

DQN adds:
- neural Q-function;
- replay buffer;
- target network.

---

### 6. Policy Gradient

Objective:

\[
J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}[R(\tau)].
\]

Trajectory probability:

\[
p_\theta(\tau)
=
p(s_0)
\prod_t
\pi_\theta(a_t|s_t)
P(s_{t+1}|s_t,a_t).
\]

Use log-derivative trick:

\[
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau).
\]

Environment dynamics do not depend on \(\theta\), so:

\[
\nabla_\theta\log p_\theta(\tau)
=
\sum_t
\nabla_\theta\log\pi_\theta(a_t|s_t).
\]

Thus:

\[
\nabla_\theta J
=
\mathbb E
\left[
\sum_t
\nabla_\theta\log\pi_\theta(a_t|s_t)
G_t
\right].
\]

This is the foundation of REINFORCE.

---

### 7. Baseline and Advantage

Subtract a baseline \(b(s_t)\):

\[
\nabla J
=
\mathbb E
[
\nabla\log\pi(a_t|s_t)
(G_t-b(s_t))
].
\]

If \(b\) does not depend on action, the estimator remains unbiased.

Choose

\[
b(s)=V^\pi(s)
\]

to obtain advantage-style policy gradients.

---

### 8. Actor–Critic

Actor:
\[
\pi_\theta(a|s).
\]

Critic:
\[
V_\phi(s).
\]

TD error:

\[
\delta_t
=
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
\]

Use \(\delta_t\) as a low-variance estimate of advantage.

---

### 9. GAE

Temporal-difference residual:

\[
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t).
\]

Generalized advantage estimate:

\[
\hat A_t^{\rm GAE}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l\delta_{t+l}.
\]

\(\lambda\) trades bias for variance.

---

### 10. PPO

Probability ratio:

\[
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\rm old}}(a_t|s_t)
}.
\]

Unclipped surrogate:

\[
r_t(\theta)\hat A_t.
\]

Clipped objective:

\[
L^{\rm CLIP}
=
\mathbb E
[
\min(
r_t\hat A_t,
\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t
)
].
\]

#### Why clipping works intuitively

If advantage is positive, increasing action probability helps, but beyond \(1+\epsilon\) PPO removes extra incentive.

If advantage is negative, decreasing probability helps, but beyond \(1-\epsilon\) PPO removes extra incentive.

This limits destructive policy updates.

---

### 11. Model-Free vs Model-Based RL

Model-free:
learn value/policy directly from experience.

Model-based:
learn or use transition model

\[
\hat P(s'|s,a)
\]

and plan through predicted futures.

World models are a major learned model-based RL approach.

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- When deriving policy gradients, explicitly use the log-derivative trick.
- Distinguish on-policy vs off-policy data.
- Keep policy, value estimator, advantage estimator, and environment dynamics conceptually separate.

---

## Hands-on / practice

## Level 1 — Reproduce
Implement or run a canonical example that demonstrates the central idea.

## Level 2 — Compare
Create at least one controlled comparison (baseline vs method, accuracy vs compute, or full vs efficient version).

## Level 3 — Explain
Write:
- what you changed;
- why it worked or failed;
- GPU memory / runtime where relevant;
- one figure or table;
- a 2-minute interview explanation.

## Deliverables
- [ ] runnable code
- [ ] README with commands
- [ ] experiment configuration
- [ ] quantitative result
- [ ] failure-case notes
- [ ] interview-ready project summary

---

## Interview readiness checklist

Before marking this chapter ready, make sure you can:

- explain the main idea without notes;
- write the important equations from memory;
- discuss at least one design tradeoff;
- compare the method with its nearest alternative;
- identify at least one failure mode;
- connect the theory to a real implementation or project.

For dedicated interview questions, see [`interview_qa.md`](interview_qa.md).  
For papers and official documentation, see [`references.md`](references.md).
