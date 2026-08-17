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


$$
p_\theta(s_{t+1}|s_t,a_t).
$$


For partially observed environments, observations $o_t$ may not be Markov, so use latent state $z_t$.

---

### 2. Latent State Model

Encoder:


$$
z_t=E_\phi(o_t).
$$


Dynamics:


$$
p_\theta(z_{t+1}|z_t,a_t).
$$


Reward model:


$$
\hat r_t=R_\psi(z_t,a_t).
$$


Optional decoder:


$$
\hat o_t=D_\omega(z_t).
$$


---

### 3. Why Latent Dynamics?

Pixels contain:
- textures;
- lighting;
- irrelevant background;
- redundant local detail.

Planning needs predictive state.

Goal:


$$
z_t
\approx
\text{sufficient information for future prediction/control}.
$$


This is representation learning with a temporal/causal purpose.

---

### 4. Deterministic Dynamics

Simple model:


$$
\hat z_{t+1}
=
F_\theta(z_t,a_t).
$$


Loss:


$$
\mathcal L_{\rm dyn}
=
\|z_{t+1}-\hat z_{t+1}\|^2.
$$


Problem:
future may be stochastic or multimodal.

---

### 5. Stochastic Latent Dynamics

Model:


$$
p_\theta(z_{t+1}|z_t,a_t).
$$


For Gaussian transition:


$$
p_\theta
=
\mathcal N(
\mu_\theta(z_t,a_t),
\Sigma_\theta(z_t,a_t)
).
$$


NLL training:


$$
\mathcal L
=
-\log p_\theta(z_{t+1}|z_t,a_t).
$$


This represents uncertainty.

---

### 6. Multi-Step Rollout Error

One-step model may be accurate:


$$
\hat z_{t+1}=F(z_t,a_t),
$$


but rollout uses predictions recursively:


$$
\hat z_{t+2}
=
F(\hat z_{t+1},a_{t+1}).
$$


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

Given current latent $z_t$, evaluate candidate action sequence


$$
a_{t:t+H-1}.
$$


Roll out:


$$
\hat z_{t+k+1}
=
F(\hat z_{t+k},a_{t+k}).
$$


Predict rewards:


$$
\hat r_{t+k}
=
R(\hat z_{t+k},a_{t+k}).
$$


Score:


$$
J(a_{t:t+H-1})
=
\sum_{k=0}^{H-1}
\gamma^k\hat r_{t+k}.
$$


Choose:


$$
a^*_{t:t+H-1}
=
\arg\max J.
$$


Execute only first action, observe real environment, and replan. This is model predictive control.

---

### 8. Dreamer-Style Imagination

Instead of planning every action sequence explicitly, learn actor and critic from trajectories imagined inside latent dynamics.

Conceptual loop:


$$
\text{real data}
\rightarrow
\text{world model}
\rightarrow
\text{imagined latent rollouts}
\rightarrow
\text{actor/critic update}.
$$


This reduces dependence on expensive real interaction.

---

### 9. Pixel Prediction vs Representation Prediction

Pixel prediction objective:


$$
\|\hat o_{t+1}-o_{t+1}\|.
$$


Representation prediction:


$$
\|\hat z_{t+1}-z_{t+1}\|.
$$


Representation prediction can ignore unpredictable or task-irrelevant pixel details.

But it risks learning a representation that hides information needed for control unless training objectives are carefully designed.

---

### 10. World Model vs Generative Model

Generic conditional generator:


$$
p(x|c).
$$


World model:


$$
p(s_{t+1}|s_t,a_t)
$$


with explicit temporal dynamics and intervention/action semantics.

A diffusion model can be used inside a world model, but diffusion itself is not automatically a world model.

---

### 11. World Model vs Simulator

Simulator:
transition equations/rules are known or manually designed.

World model:
transition function is learned from data.

Hybrid systems can combine both.

<!-- DEEP_DIVE_START -->
## Deep dive I: state-space modeling view

A useful generic latent state-space model is:


$$
z_t\sim p_\theta(z_t|z_{t-1},a_{t-1}),
$$


$$
o_t\sim p_\theta(o_t|z_t).
$$


Inference estimates:


$$
q_\phi(z_t|o_{\le t},a_{<t}).
$$


This resembles a learned nonlinear probabilistic state-space model.

World-model research combines:
- representation learning;
- temporal modeling;
- generative modeling;
- control.

---

## Deep dive II: deterministic + stochastic state

One useful design separates:
- deterministic recurrent state $h_t$;
- stochastic latent $z_t$.

Example:


$$
h_t=f(h_{t-1},z_{t-1},a_{t-1}),
$$


prior:


$$
p(z_t|h_t),
$$


posterior:


$$
q(z_t|h_t,o_t).
$$


Why both?
- $h_t$ can summarize predictable history;
- $z_t$ can represent uncertainty/multimodal variation.

---

## Deep dive III: learning objective

A latent world model may combine:

### Reconstruction / observation prediction


$$
L_o=-\log p(o_t|z_t,h_t).
$$


### Reward prediction


$$
L_r=-\log p(r_t|z_t,h_t).
$$


### Dynamics regularization


$$
L_{\rm KL}
=
D_{\rm KL}
(
q(z_t|h_t,o_t)
\|
p(z_t|h_t)
).
$$


This is conceptually VAE-like but extended through time.

---

## Deep dive IV: why open-loop rollout is the real test

One-step evaluation feeds true state every time:


$$
\hat z_{t+1}=F(z_t,a_t).
$$


Open-loop rollout feeds prediction:


$$
\hat z_{t+2}=F(\hat z_{t+1},a_{t+1}).
$$


The second setting matches planning usage.

A model can have excellent one-step MSE but fail catastrophically after 20 recursive steps.

Therefore evaluate error vs rollout horizon:


$$
E(k)=
\|z_{t+k}-\hat z_{t+k}\|.
$$


---

## Deep dive V: planner exploiting model error

Suppose the learned model incorrectly predicts a large reward in an unseen region.

An optimizer searching action sequences may intentionally drive into that error because it maximizes predicted reward.

This is analogous to adversarial exploitation of the learned model.

Mitigations:
- uncertainty penalties;
- conservative planning;
- ensemble disagreement;
- restrict planning to data support;
- online data collection.

---

## Deep dive VI: model predictive control example

At each real time step:

1. sample candidate action sequences;
2. roll them out in model;
3. score predicted return;
4. execute first action from best sequence;
5. observe real next state;
6. repeat.

Why execute only first action?
Because new real observation corrects accumulated model error.

---

## Deep dive VII: CEM planning

Cross-Entropy Method can optimize continuous action sequences.

Initialize distribution:


$$
a_{0:H-1}\sim\mathcal N(\mu,\Sigma).
$$


Loop:
1. sample many sequences;
2. evaluate model-predicted returns;
3. keep top elite fraction;
4. refit $\mu,\Sigma$;
5. repeat.

CEM is derivative-free and commonly used with learned dynamics.

---

## Deep dive VIII: Dreamer conceptual decomposition

```text
real observations
→ encoder/posterior
→ latent dynamics model
→ imagined future trajectories
→ critic estimates long-term return
→ actor improves actions in latent imagination
```

The main efficiency idea is that many policy-learning trajectories occur **inside the learned model** rather than the expensive real environment.

---

## Deep dive IX: JEPA-style representation prediction

Instead of reconstructing exact pixels:


$$
\hat o_{t+1}\approx o_{t+1},
$$


predict latent target:


$$
\hat z_{t+1}\approx z_{t+1}.
$$


Why useful?
Future pixels contain inherently unpredictable details. Predicting representation can focus on stable semantic/physical structure.

But target representation design becomes critical. If it discards action-relevant information, planning suffers.

---

## Deep dive X: diffusion world model

One-step future may be multimodal:


$$
p(o_{t+1}|o_t,a_t).
$$


A diffusion model can represent this conditional distribution.

Thus:

```text
current observation + action
→ conditional diffusion
→ sample plausible future
```

This is a world model **if** it is trained as environment dynamics and used for prediction/planning.

---

## Minimal latent-world-model pseudocode

```python
z = encoder(obs)
z_next_pred = dynamics(z, action)
reward_pred = reward_head(z, action)

loss_dyn = F.mse_loss(z_next_pred, z_next.detach())
loss_reward = F.mse_loss(reward_pred, reward)

loss = loss_dyn + lam * loss_reward
```

A serious version must address:
- representation collapse;
- multi-step error;
- stochasticity;
- action conditioning;
- uncertainty;
- planner distribution shift.

---

## World-model interview framework

When asked “design a world model,” answer in this order:

1. observation/state definition;
2. latent representation;
3. action-conditioned dynamics;
4. uncertainty/stochasticity;
5. reward/value prediction;
6. training objective;
7. rollout/planning method;
8. evaluation over horizon;
9. failure under distribution shift.
<!-- DEEP_DIVE_END -->

<!-- SECOND_DEEP_DIVE_START -->
## World-model taxonomy

A useful taxonomy separates what the model predicts.

### Observation model


$$
p(o_{t+1}|o_{\le t},a_{\le t}).
$$


### Latent dynamics


$$
p(z_{t+1}|z_t,a_t).
$$


### Trajectory model


$$
p(o_{t+1:t+H}|o_{\le t},a_{t:t+H-1}).
$$


### Value/reward predictive model
Predicts task-relevant consequences.

Different job descriptions may call all of these “world models,” so clarify what state, action, horizon, and planning role are meant.

---

## Action conditioning is central

Without action:


$$
p(z_{t+1}|z_t)
$$


models passive dynamics.

For control we need interventions:


$$
p(z_{t+1}|z_t,a_t).
$$


Two futures can differ only because the agent chose different actions. If action is omitted, the model cannot support counterfactual planning properly.

---

## Counterfactual reasoning

World model allows queries:

```text
same current state
├→ action A → future A
├→ action B → future B
└→ action C → future C
```

Planning compares these counterfactual consequences before real execution.

This is a key distinction from ordinary sequence prediction.

---

## Latent overshooting / multi-step training

One-step objective:


$$
L_1
=
\|F(z_t,a_t)-z_{t+1}\|.
$$


Multi-step objective:


$$
L_H
=
\sum_{k=1}^H
w_k
\|\hat z_{t+k}-z_{t+k}\|.
$$


Multi-step losses expose the model to recursive errors during training.

Tradeoff:
long-horizon target becomes harder and may over-penalize inherently uncertain futures.

---

## Ensemble uncertainty

Train dynamics models:


$$
F_1,\ldots,F_M.
$$


For candidate next state predictions, disagreement:


$$
U(z,a)
=
\mathrm{Var}_m[F_m(z,a)].
$$


Planner can penalize uncertain trajectories:


$$
J_{\rm conservative}
=
J_{\rm reward}
-
\lambda U.
$$


Ensemble disagreement is an approximate epistemic uncertainty signal.

---

## Reconstruction may be unnecessary

If planner only needs:
- object position;
- velocity;
- reachable state;
- reward features,

reconstructing every pixel may waste capacity.

But purely abstract latent prediction creates evaluation difficulty because latent coordinates may not be directly interpretable.

This is why representation-learning objective is a core design choice, not an implementation detail.

---

## World model for video

Video frames:


$$
o_t\in\mathbb R^{H\times W\times C}.
$$


A video prediction model


$$
p(o_{t+1}|o_{\le t})
$$


models passive world evolution.

To become useful for an embodied agent, incorporate control:


$$
p(o_{t+1}|o_{\le t},a_t).
$$


Large-scale generative video models and action-conditioned world models overlap but are not identical categories.

---

## World model for scientific systems

The concept is broader than robotics/games.

Suppose physical system state $s_t$, control $a_t$, measurement $o_t$.

A learned surrogate can predict:


$$
s_{t+1}=F_\theta(s_t,a_t).
$$


Use cases:
- experiment design;
- treatment/control planning;
- simulation acceleration;
- digital twins.

Physics-based simulators and learned dynamics can be hybridized:


$$
F(s,a)
=
F_{\rm physics}(s,a)
+
\Delta_\theta(s,a).
$$


This connects world-model thinking to model mismatch correction.

---

## Mini-project: latent CartPole/Pendulum model

Collect trajectories:


$$
(s_t,a_t,r_t,s_{t+1}).
$$


Even if environment state is low dimensional, intentionally learn latent encoder/dynamics to practice architecture.

Experiments:
1. one-step dynamics;
2. multi-step rollout;
3. uncertainty ensemble;
4. MPC/CEM planning;
5. compare against model-free policy.

Plot:
- prediction error vs horizon;
- return vs planning horizon;
- failure trajectories.

This gives a complete world-model interview story.
<!-- SECOND_DEEP_DIVE_END -->

<!-- THIRD_DEEP_DIVE_START -->
## Model learning vs policy learning

Keep two optimizations separate.

World model:


$$
\theta^*
=
\arg\min_\theta
L_{\rm prediction}(\theta).
$$


Policy:


$$
\phi^*
=
\arg\max_\phi
\mathbb E[
\sum_t\gamma^tr_t
].
$$


A better predictive model does not automatically guarantee a better policy if prediction improvements occur on irrelevant state dimensions.

This motivates task-aware representations and planning metrics.

---

## What should a world model predict?

Possible targets:

### Full observation
High information, expensive, includes nuisance detail.

### Latent representation
Efficient, but representation may hide important factors.

### Reward/value only
Task-specific and efficient, poor general transfer.

### Multi-task semantic state
Object positions, geometry, dynamics, etc.

The target determines what “world understanding” means.

---

## Planning horizon

Longer horizon sees farther:


$$
H\uparrow
\Rightarrow
\text{potentially better long-term decisions}.
$$


But:
- model error compounds;
- optimization becomes harder;
- uncertainty grows.

Therefore there is often an optimal finite horizon.

MPC partly solves this by replanning after every real observation.

---

## Latent collapse in world models

If latent learning is unconstrained, representation may become constant or ignore dynamics.

Need objectives ensuring:
- observation information;
- predictive information;
- controllability/action relevance;
- non-collapse.

This connects world models to SSL collapse problems.

---

## Causal caution

Action-conditioned prediction is not automatically causal identification.

If behavior policy chooses actions non-randomly, training data may have confounding/state coverage issues.

The model learns transitions under observed data distribution. Counterfactual extrapolation to rare actions can be unreliable.

This is another reason planning outside training support is dangerous.

---

## Interview comparison: world model vs digital twin

A digital twin often implies a structured model of a specific real system, potentially combining mechanistic physics, sensors, calibration, and learned components.

A world model is a broader learned predictive representation/dynamics concept.

They overlap strongly when a learned model is used to simulate a physical system for decision-making.
<!-- THIRD_DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Evaluate both one-step prediction and long-horizon rollout quality.
- Planning can exploit model errors; uncertainty and out-of-distribution behavior matter.
- Distinguish representation learning, dynamics learning, reward/value prediction, and policy/planner.

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
