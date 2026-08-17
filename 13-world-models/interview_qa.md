# World Models — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. What exactly makes a model a world model?

A world model learns how the environment evolves, usually including how the agent's action changes the future. A minimal control-relevant form is:

```math
p_\theta(s_{t+1}|s_t,a_t).
```

For high-dimensional observations, the model often learns a latent state:

```math
z_t=E(o_t),
\qquad
p_\theta(z_{t+1}|z_t,a_t).
```

**Key idea.** A generic generator models plausible data; a world model models **consequences**. It supports questions such as: “from this same state, what happens if I take action A instead of action B?”

That action-conditioned counterfactual structure is what enables planning.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Why use a latent state instead of predicting pixels directly?

Raw pixels contain enormous amounts of detail that can be irrelevant to decisions: texture, lighting, background, or sensor noise. A latent state can compress observations into variables useful for predicting future dynamics, reward, and controllability.

**Tradeoff.** If the latent discards action-relevant information, planning fails. If it retains too much irrelevant detail, dynamics learning becomes harder.

The desired state is therefore not simply “low dimensional.” It should be **predictively sufficient** for the future quantities and decisions the agent cares about.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. Why is one-step prediction error a weak evaluation for a world model?

One-step evaluation always starts from the true state. Planning recursively feeds the model's own prediction back into future predictions:

```math
\hat z_{t+2}
=
F(\hat z_{t+1},a_{t+1}).
```

Small one-step errors can accumulate rapidly. A model can have excellent one-step MSE but unrealistic long open-loop rollouts.

A better evaluation plots error versus rollout horizon, tests reward/value prediction, and measures downstream planning performance.

**Key idea.** The model should be evaluated under the way it will actually be used—recursive imagination—not only under teacher-forced one-step prediction.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. How does Model Predictive Control use a world model?

Model Predictive Control (MPC) evaluates candidate future action sequences in the learned dynamics model, scores their predicted returns, executes only the first action of the best sequence, observes the real next state, and replans.

**Why execute only the first action?** A fresh real observation corrects accumulated model error.

The loop is:

```text
observe
→ imagine candidate futures
→ choose best sequence
→ execute first action
→ observe real outcome
→ replan
```

MPC therefore trades online computation for robustness to imperfect long-horizon predictions.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. Why can a planner exploit model errors?

A planner searches for actions with high **predicted** reward. If the learned model has an unrealistic region where reward is overestimated, optimization may deliberately drive the imagined trajectory there.

This is a distribution-shift problem: planning can evaluate states/actions poorly covered by training data.

Mitigation includes:
- model ensembles and disagreement;
- uncertainty penalties;
- conservative planning;
- shorter MPC horizons;
- restricting actions to data support;
- online data collection.

This failure mode is one of the main practical reasons world-model uncertainty matters.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. What is Dreamer-style imagination?

Dreamer learns a latent world model from real experience. It then rolls latent states forward inside the learned dynamics and trains actor/critic components using those **imagined** trajectories instead of requiring every policy update to come from new real interaction.

**Key idea.** Real environment interaction is expensive; model-generated latent trajectories are cheap.

The method separates:
1. representation/state inference;
2. latent dynamics;
3. reward/value prediction;
4. policy/value optimization.

Its usefulness depends on the imagined trajectories being sufficiently accurate in decision-relevant regions.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q7. JEPA-style prediction versus pixel prediction?

Pixel prediction asks the model to reproduce exact future observations. This can waste capacity on unpredictable low-level detail.

Joint-Embedding Predictive Architecture (JEPA)-style approaches predict a target representation:

```math
\hat z_{\mathrm{future}}
\approx
z_{\mathrm{future}}.
```

**Design idea.** Predict stable semantic or physical structure rather than every raw pixel.

**Tradeoff.** The target representation must preserve information needed for understanding/control. Abstract prediction is not automatically useful if important dynamics are hidden by the representation.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q8. Can a diffusion model be a world model?

Yes—if diffusion models action-conditioned future states or trajectories, for example:

```math
p(o_{t+1}|o_t,a_t).
```

Conditional diffusion is attractive when the future is multimodal and there are several plausible outcomes.

But a text-to-image diffusion model trained only on static images is not automatically a world model. The defining property is modeling environment dynamics and action consequences, not the denoising algorithm itself.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

