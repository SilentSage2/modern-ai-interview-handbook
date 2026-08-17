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

<!-- DEEP_DIVE_START -->
## Deep dive I: the Markov property

A state \(s_t\) is Markov if:

\[
p(s_{t+1}|s_{0:t},a_{0:t})
=
p(s_{t+1}|s_t,a_t).
\]

The current state must summarize all relevant history for future evolution.

In partially observable environments, observation \(o_t\) may not be Markov. An agent may need history or a learned latent state:

\[
z_t=f(o_{\le t},a_{<t}).
\]

This directly motivates recurrent state estimators and world models.

---

## Deep dive II: Monte Carlo vs temporal difference

### Monte Carlo target

\[
G_t
=
r_t+\gamma r_{t+1}+\cdots.
\]

Unbiased for on-policy returns but high variance and requires episode/future completion.

### TD(0) target

\[
r_t+\gamma V(s_{t+1}).
\]

Bootstraps from current estimate.

Lower variance but introduces bias from inaccurate \(V\).

Many RL algorithms navigate this bias-variance tradeoff.

---

## Deep dive III: why Q-learning is off-policy

Update target uses:

\[
\max_{a'}Q(s',a')
\]

regardless of which action the behavior policy actually took next.

Therefore data can come from another exploratory behavior policy while learning the greedy target policy.

This supports replay buffers.

---

## Deep dive IV: policy-gradient derivation carefully

Objective:

\[
J(\theta)=\sum_\tau p_\theta(\tau)R(\tau).
\]

Differentiate:

\[
\nabla J
=
\sum_\tau \nabla p_\theta(\tau)R(\tau).
\]

Multiply/divide by \(p_\theta(\tau)\):

\[
\nabla J
=
\sum_\tau
p_\theta(\tau)
\nabla\log p_\theta(\tau)
R(\tau).
\]

Thus:

\[
\nabla J
=
\mathbb E_{\tau\sim p_\theta}
[
\nabla\log p_\theta(\tau)R(\tau)
].
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

Only policy depends on \(\theta\):

\[
\nabla\log p_\theta(\tau)
=
\sum_t\nabla\log\pi_\theta(a_t|s_t).
\]

Therefore:

\[
\nabla J
=
\mathbb E
\left[
\sum_t
\nabla\log\pi_\theta(a_t|s_t)R(\tau)
\right].
\]

Then causality lets us replace total trajectory reward with reward-to-go.

---

## Deep dive V: entropy bonus

Policy loss may include:

\[
+\beta H(\pi(\cdot|s)).
\]

Entropy:

\[
H(\pi)
=
-\sum_a\pi(a|s)\log\pi(a|s).
\]

Encourages exploration and prevents premature collapse to deterministic policies.

Too much entropy can prevent exploitation/convergence.

---

## Deep dive VI: PPO clipping cases

Let

\[
r=\frac{\pi_\theta(a|s)}{\pi_{\rm old}(a|s)}.
\]

### Positive advantage
Action was better than baseline. We want \(r>1\), but cap benefit beyond \(1+\epsilon\).

### Negative advantage
Action was worse. We want \(r<1\), but cap benefit beyond \(1-\epsilon\).

PPO does not literally guarantee a strict trust region; clipping is a practical surrogate that discourages excessive updates.

---

## Deep dive VII: actor-critic training losses

Typical combined objective includes:

Policy:
\[
L_\pi
=
-\mathbb E[L^{CLIP}].
\]

Value:
\[
L_V
=
\mathbb E[(V_\phi(s_t)-\hat V_t)^2].
\]

Entropy:
\[
L_H=-H(\pi).
\]

Total:

\[
L
=
L_\pi
+
c_vL_V
+
c_hL_H.
\]

This is why PPO implementation is more than one clipped equation.

---

## Minimal PPO data flow

```text
policy(old)
→ collect trajectories
→ rewards + values
→ compute returns / GAE
→ several minibatch epochs
→ update policy and value networks
→ discard on-policy rollout
→ collect new rollout
```

Key point:
PPO data become stale as policy changes. Reusing them indefinitely violates its on-policy assumptions.

---

## Deep dive VIII: RLHF mapping

LLM state:

\[
s_t=(\text{prompt},y_{<t}).
\]

Action:

\[
a_t=y_t.
\]

Policy:

\[
\pi_\theta(a_t|s_t)
\]

is next-token distribution.

A sequence-level reward can be assigned after generating a response.

Then policy optimization resembles RL, often with a KL penalty to a reference model.

This mapping makes RLHF much easier to understand after classical RL.

---

## Common RL misconceptions

**“Reward is a loss.”**  
Reward is an environment/task signal; the optimization algorithm constructs losses/surrogates from it.

**“Q-learning and policy gradient are the same.”**  
Q-learning learns value estimates and chooses actions from them; policy gradient directly differentiates a parameterized policy objective.

**“World models are just RL policies.”**  
A world model predicts environment dynamics; policy/planning decides actions.
<!-- DEEP_DIVE_END -->

<!-- SECOND_DEEP_DIVE_START -->
## Tabular Q-learning worked example

Suppose state \(s\), action \(a\), reward \(r=1\), next state has:

\[
Q(s',a_1)=3,
\qquad
Q(s',a_2)=5.
\]

With \(\gamma=0.9\), target:

\[
y=1+0.9(5)=5.5.
\]

If current

\[
Q(s,a)=4,
\]

TD error:

\[
\delta=5.5-4=1.5.
\]

Update with \(\alpha=0.1\):

\[
Q(s,a)\leftarrow4+0.1(1.5)=4.15.
\]

This simple numeric example is worth being able to do instantly.

---

## Exploration: epsilon-greedy

For value-based method:

\[
a=
\begin{cases}
\text{random},&\text{probability }\epsilon\\
\arg\max_aQ(s,a),&\text{otherwise}.
\end{cases}
\]

As \(\epsilon\) decreases, policy shifts from exploration toward exploitation.

Deep RL has many more sophisticated exploration strategies, but this is the baseline intuition.

---

## Policy-gradient sign intuition

Loss often implemented as:

\[
L_\pi
=
-\log\pi_\theta(a|s)A.
\]

If \(A>0\):
minimizing loss increases \(\log\pi(a|s)\), so action becomes more likely.

If \(A<0\):
it decreases action probability.

This is a useful sanity check for code.

---

## Advantage normalization

In PPO implementations, advantages are often normalized within batch:

\[
\hat A
\leftarrow
\frac{\hat A-\mu_A}{\sigma_A+\epsilon}.
\]

This changes scale/conditioning of gradient updates and can stabilize training.

It is an implementation technique, not part of the fundamental policy-gradient theorem.

---

## PPO pseudocode

```python
with torch.no_grad():
    rollout = collect(env, policy_old)
    adv, ret = compute_gae(rollout)

for epoch in range(K):
    for batch in minibatches(rollout):
        logp = policy.log_prob(batch.s, batch.a)
        ratio = (logp - batch.logp_old).exp()

        obj1 = ratio * batch.adv
        obj2 = ratio.clamp(1-eps, 1+eps) * batch.adv
        policy_loss = -torch.min(obj1, obj2).mean()

        value_loss = F.mse_loss(value(batch.s), batch.ret)
        entropy = policy.entropy(batch.s).mean()

        loss = policy_loss + c1*value_loss - c2*entropy
        optimize(loss)
```

Know what `logp_old` means: probability under the policy that generated the rollout.

---

## Continuous actions

Policy may be Gaussian:

\[
a\sim
\mathcal N(
\mu_\theta(s),
\sigma_\theta^2(s)
).
\]

Log probability is differentiable with respect to policy parameters.

This is one reason stochastic policies fit continuous-control policy-gradient methods naturally.

---

## Offline RL concept

Offline RL learns from a fixed dataset:

\[
D=\{(s,a,r,s')\}
\]

without new environment interaction.

Core challenge:
policy may choose actions poorly represented in dataset. Value estimates for out-of-distribution actions can be unreliable.

This connects to world-model planning exploitation of model error: both involve extrapolation beyond data support.

---

## RL evaluation pitfalls

Returns can have high variance.

Report:
- multiple random seeds;
- learning curves;
- mean + dispersion;
- sample efficiency;
- environment steps;
- evaluation policy without exploration noise when appropriate.

One lucky seed is not convincing evidence.

---

## Mini-project progression

1. tabular Q-learning on small environment;
2. DQN;
3. policy gradient;
4. PPO;
5. replace real transition reasoning with learned world model.

This sequence creates a natural bridge from classical RL to world models rather than learning PPO as an isolated formula.
<!-- SECOND_DEEP_DIVE_END -->

<!-- THIRD_DEEP_DIVE_START -->
## Credit assignment

A reward may occur long after the action that caused it.

Return:

\[
G_t
=
r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.
\]

The algorithm must determine which earlier actions deserve credit.

Discount \(\gamma\) controls effective horizon:

\[
H_{\rm effective}\approx\frac{1}{1-\gamma}.
\]

For \(\gamma=0.99\), effective horizon is on the order of 100 steps.

Long horizons make credit assignment and variance harder.

---

## Reward shaping

One can add intermediate rewards to make learning easier.

Risk:
agent learns to optimize shaped reward rather than true objective.

This is an instance of specification gaming.

Whenever discussing RL systems, ask:

> Does maximizing the numerical reward really correspond to desired behavior?

This question is highly relevant to RLHF and agents.

---

## On-policy vs off-policy

### On-policy
Learn from data generated by current/recent policy.

Examples:
- REINFORCE;
- PPO.

### Off-policy
Can learn from behavior generated by another policy.

Examples:
- Q-learning;
- DQN.

Off-policy methods can reuse data more efficiently but require correction/algorithms that handle distribution mismatch.

---

## Importance sampling connection

If data come from behavior \(\mu\) but target is \(\pi\):

\[
\mathbb E_{a\sim\pi}[f(a)]
=
\mathbb E_{a\sim\mu}
\left[
\frac{\pi(a)}{\mu(a)}f(a)
\right].
\]

PPO ratio

\[
r_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\rm old}(a_t|s_t)}
\]

is related to this change-of-distribution logic.

Large ratios lead to high variance/unstable updates, motivating restrictions.

---

## Partial observability

If current observation is insufficient, policy may need memory:

\[
a_t\sim
\pi(
a_t|
o_t,h_t
),
\]

\[
h_t=f(h_{t-1},o_t,a_{t-1}).
\]

This brings RL close to sequence modeling and latent world-state estimation.

---

## RL interview answer hierarchy

For any algorithm, state:

1. What is learned: value, policy, model?
2. Is it on-policy or off-policy?
3. What target/objective?
4. How is exploration handled?
5. Where does bias/variance enter?
6. What stabilizes training?
7. What data can be reused?
8. What are failure modes?
<!-- THIRD_DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- When deriving policy gradients, explicitly use the log-derivative trick.
- Distinguish on-policy vs off-policy data.
- Keep policy, value estimator, advantage estimator, and environment dynamics conceptually separate.

---

## Hands-on / practice

### Level 1 — Reproduce
Implement or run a canonical example that demonstrates the central idea.

### Level 2 — Compare
Create at least one controlled comparison (baseline vs method, accuracy vs compute, or full vs efficient version).

### Level 3 — Explain
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
