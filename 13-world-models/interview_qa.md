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

## Extended Interview Q&A

### E1. Why is one-step prediction error not enough?

**Answer:** Planning recursively feeds predictions back into the model. Small one-step errors can compound, so long-horizon rollout quality matters.

### E2. What makes a latent useful for planning?

**Answer:** It should retain state information needed to predict controllable future consequences while discarding irrelevant observation detail.

### E3. Model predictive control vs learned policy?

**Answer:** MPC solves an optimization over future action sequences online using the model; a learned policy directly maps state to action after training.

### E4. Why can uncertainty matter in world-model planning?

**Answer:** The planner may exploit model errors and choose actions in poorly modeled regions. Uncertainty can penalize or detect unreliable imagined trajectories.

### E5. Can diffusion be a world model?

**Answer:** Yes, if used to model action-conditioned future states/trajectories. But diffusion used only for static data generation is not automatically a world model.


## Whiteboard / drill questions

- Why can a planner exploit model error?
- How would you evaluate world-model error as a function of horizon?
- Why combine deterministic and stochastic latent state?
- Compare MPC with an actor learned from imagined trajectories.
- Under what condition does a diffusion model qualify as a world model?


<!-- ADVANCED_QA_START -->
## Advanced / system follow-ups

### A1. What is the difference between open-loop and closed-loop prediction?

**Answer:** Open-loop recursively feeds model predictions forward without real correction. Closed-loop/receding-horizon control repeatedly incorporates new real observations, limiting accumulated model drift.

### A2. Why can a high-fidelity pixel model still be a poor controller?

**Answer:** It may spend capacity on visually accurate but decision-irrelevant detail while failing to predict controllable variables or rewards precisely enough for planning.

### A3. How do ensembles help a world model?

**Answer:** Disagreement among independently trained dynamics models provides a rough epistemic uncertainty signal, useful for penalizing poorly supported imagined trajectories.

### A4. Why is action coverage important in the training dataset?

**Answer:** Counterfactual planning evaluates actions that may differ from historical behavior. If the data rarely contains those actions in similar states, transition predictions may be unreliable.

### A5. What does 'imagination' mean in Dreamer-like methods?

**Answer:** It means rolling forward trajectories using learned latent dynamics rather than interacting with the real environment. Actor/critic learning can then use these simulated latent trajectories.

<!-- ADVANCED_QA_END -->
