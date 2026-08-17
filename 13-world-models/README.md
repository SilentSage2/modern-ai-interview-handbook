# World Models

**Status:** Priority + Hands-on

## Why this matters

World models extend representation and generative modeling into action-conditioned temporal prediction and planning.

## Learning objectives

- Formulate deterministic and stochastic latent dynamics.
- Explain multi-step rollout error and imagined planning.
- Connect MPC, Dreamer, JEPA-style prediction, and generative world models.

## Chapter map

- Learned dynamics p(s_{t+1}|s_t,a_t)
- Latent-state models
- Observation encoder, transition model, reward/value model
- Model-based RL
- Imagined rollouts
- Planning and model predictive control
- Dreamer-style latent imagination
- Predicting pixels vs representations
- JEPA-style predictive representations
- Generative world models
- Uncertainty and compounding model error


---

## Core concepts and theory

### 1. Core Definition

A world model learns how an environment evolves.

A basic action-conditioned model:

\[
p_\theta(s_{t+1}|s_t,a_t).
\]

For partially observed environments, observations \(o_t\) may not be Markov, so use latent state \(z_t\).

---

### 2. Latent State Model

Encoder:

\[
z_t=E_\phi(o_t).
\]

Dynamics:

\[
p_\theta(z_{t+1}|z_t,a_t).
\]

Reward model:

\[
\hat r_t=R_\psi(z_t,a_t).
\]

Optional decoder:

\[
\hat o_t=D_\omega(z_t).
\]

---

### 3. Why Latent Dynamics?

Pixels contain:
- textures;
- lighting;
- irrelevant background;
- redundant local detail.

Planning needs predictive state.

Goal:

\[
z_t
\approx
\text{sufficient information for future prediction/control}.
\]

This is representation learning with a temporal/causal purpose.

---

### 4. Deterministic Dynamics

Simple model:

\[
\hat z_{t+1}
=
F_\theta(z_t,a_t).
\]

Loss:

\[
\mathcal L_{\rm dyn}
=
\|z_{t+1}-\hat z_{t+1}\|^2.
\]

Problem:
future may be stochastic or multimodal.

---

### 5. Stochastic Latent Dynamics

Model:

\[
p_\theta(z_{t+1}|z_t,a_t).
\]

For Gaussian transition:

\[
p_\theta
=
\mathcal N(
\mu_\theta(z_t,a_t),
\Sigma_\theta(z_t,a_t)
).
\]

NLL training:

\[
\mathcal L
=
-\log p_\theta(z_{t+1}|z_t,a_t).
\]

This represents uncertainty.

---

### 6. Multi-Step Rollout Error

One-step model may be accurate:

\[
\hat z_{t+1}=F(z_t,a_t),
\]

but rollout uses predictions recursively:

\[
\hat z_{t+2}
=
F(\hat z_{t+1},a_{t+1}).
\]

Small errors compound.

This creates distribution shift:
training sees true states, planning sees model-generated states.

Important mitigation ideas:
- multi-step losses;
- stochastic models;
- uncertainty penalties;
- data aggregation;
- latent regularization.

---

### 7. Planning with a World Model

Given current latent \(z_t\), evaluate candidate action sequence

\[
a_{t:t+H-1}.
\]

Roll out:

\[
\hat z_{t+k+1}
=
F(\hat z_{t+k},a_{t+k}).
\]

Predict rewards:

\[
\hat r_{t+k}
=
R(\hat z_{t+k},a_{t+k}).
\]

Score:

\[
J(a_{t:t+H-1})
=
\sum_{k=0}^{H-1}
\gamma^k\hat r_{t+k}.
\]

Choose:

\[
a^*_{t:t+H-1}
=
\arg\max J.
\]

Execute only first action, observe real environment, and replan. This is model predictive control.

---

### 8. Dreamer-Style Imagination

Instead of planning every action sequence explicitly, learn actor and critic from trajectories imagined inside latent dynamics.

Conceptual loop:

\[
\text{real data}
\rightarrow
\text{world model}
\rightarrow
\text{imagined latent rollouts}
\rightarrow
\text{actor/critic update}.
\]

This reduces dependence on expensive real interaction.

---

### 9. Pixel Prediction vs Representation Prediction

Pixel prediction objective:

\[
\|\hat o_{t+1}-o_{t+1}\|.
\]

Representation prediction:

\[
\|\hat z_{t+1}-z_{t+1}\|.
\]

Representation prediction can ignore unpredictable or task-irrelevant pixel details.

But it risks learning a representation that hides information needed for control unless training objectives are carefully designed.

---

### 10. World Model vs Generative Model

Generic conditional generator:

\[
p(x|c).
\]

World model:

\[
p(s_{t+1}|s_t,a_t)
\]

with explicit temporal dynamics and intervention/action semantics.

A diffusion model can be used inside a world model, but diffusion itself is not automatically a world model.

---

### 11. World Model vs Simulator

Simulator:
transition equations/rules are known or manually designed.

World model:
transition function is learned from data.

Hybrid systems can combine both.

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Evaluate both one-step prediction and long-horizon rollout quality.
- Planning can exploit model errors; uncertainty and out-of-distribution behavior matter.
- Distinguish representation learning, dynamics learning, reward/value prediction, and policy/planner.

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
