# Machine Learning Foundations

**Status:** Strong / Review

## Why this matters

These are the concepts interviewers use to test whether later deep-learning answers are built on solid statistical reasoning.

## Learning objectives

- Derive common losses from probabilistic assumptions.
- Explain generalization, regularization, bias–variance, calibration, and leakage.
- Compare SGD, Momentum, Adam, and AdamW from their update rules.

## Chapter map

- Supervised vs unsupervised vs self-supervised learning
- Regression vs classification; probabilistic interpretation of losses
- Train/validation/test split and data leakage
- Bias–variance tradeoff, underfitting, overfitting
- Regularization: L1, L2, weight decay, dropout, early stopping
- Metrics: precision, recall, F1, ROC-AUC, PR-AUC, calibration
- Optimization basics: SGD, momentum, Adam, AdamW
- Learning-rate schedules, gradient clipping, batch-size effects
- Class imbalance and sampling/weighting strategies
- Cross-validation and distribution shift


---

## Core concepts and theory

### 1. Empirical Risk Minimization

Given data $\{(x_i,y_i)\}_{i=1}^N$, supervised learning usually solves


$$
\hat{\theta}=\arg\min_\theta \frac1N\sum_{i=1}^N \ell(f_\theta(x_i),y_i)+\lambda\Omega(\theta).
$$


- $\ell$: data-fit loss.
- $\Omega$: regularizer / prior.
- $\lambda$: tradeoff between fitting and complexity.

The population objective is


$$
R(\theta)=\mathbb E_{(x,y)\sim p_{\rm data}}\left[\ell(f_\theta(x),y)\right],
$$


but we only observe the empirical approximation.

#### Interview connection
Generalization is the gap between empirical risk and population risk.

---

### 2. MSE from Maximum Likelihood

Assume regression noise is Gaussian:


$$
y=f_\theta(x)+\epsilon,\qquad \epsilon\sim\mathcal N(0,\sigma^2).
$$


Then


$$
p(y|x,\theta)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(y-f_\theta(x))^2}{2\sigma^2}\right].
$$


Negative log-likelihood:


$$
-\log p(y|x,\theta)
=
\frac{(y-f_\theta(x))^2}{2\sigma^2}+C.
$$


Therefore minimizing Gaussian NLL is equivalent to minimizing MSE.

#### What changes with MAE?
MAE corresponds to a Laplace observation model:


$$
p(y|x,\theta)\propto \exp\left(-\frac{|y-f_\theta(x)|}{b}\right).
$$


This makes MAE more robust to outliers.

---

### 3. Cross-Entropy from Maximum Likelihood

For multiclass classification,


$$
p_\theta(y=k|x)=\frac{e^{z_k}}{\sum_j e^{z_j}}.
$$


For one-hot target $y$,


$$
\mathcal L_{\rm CE}
=
-\sum_k y_k\log p_\theta(y=k|x).
$$


Since only the correct class has $y_k=1$,


$$
\mathcal L_{\rm CE}=-\log p_\theta(y_{\rm true}|x).
$$


So cross-entropy is simply categorical negative log-likelihood.

---

### 4. Bias–Variance Decomposition

For squared error,


$$
\mathbb E[(y-\hat f(x))^2]
=
\sigma^2
+
\left(\mathbb E[\hat f(x)]-f(x)\right)^2
+
\mathbb E\left[(\hat f(x)-\mathbb E[\hat f(x)])^2\right].
$$


Terms:
- irreducible noise;
- squared bias;
- variance.

High-capacity models generally reduce bias but can increase variance.

---

### 5. L2 Regularization as a Prior

Objective:


$$
\mathcal L(\theta)+\lambda\|\theta\|_2^2.
$$


A Gaussian prior


$$
p(\theta)\propto \exp\left(-\frac{\|\theta\|_2^2}{2\sigma_\theta^2}\right)
$$


makes MAP estimation equivalent to L2-regularized optimization.

Similarly, L1 regularization corresponds to a Laplace prior and encourages sparsity.

---

### 6. SGD, Momentum, Adam, AdamW

#### SGD


$$
\theta_{t+1}=\theta_t-\eta g_t.
$$


#### Momentum


$$
v_t=\beta v_{t-1}+g_t,\qquad
\theta_{t+1}=\theta_t-\eta v_t.
$$


Momentum smooths noisy gradients and accelerates progress along persistent directions.

#### Adam


$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$


$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$


Bias-correct:


$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
$$


Update:


$$
\theta_{t+1}
=
\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$


#### AdamW
AdamW decouples weight decay from the adaptive gradient:


$$
\theta_{t+1}
=
(1-\eta\lambda)\theta_t
-
\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$


This is not exactly the same as adding an L2 penalty inside Adam because Adam rescales gradients coordinate-wise.

---

### 7. Calibration

A model is calibrated if


$$
P(Y=\hat Y\mid \hat p=0.8)\approx0.8.
$$


Accuracy and calibration are different properties. A highly accurate model can still be overconfident.

Common metrics:
- NLL;
- Brier score;
- ECE;
- coverage of prediction intervals.

---

### 8. Data Leakage

Leakage occurs when information unavailable at deployment enters training or model selection.

Examples:
- normalizing using full-dataset statistics before splitting;
- subject-level medical scans distributed across train and test;
- selecting hyperparameters using the test set.

The safest principle: **split first, fit preprocessing only on training data.**

<!-- DEEP_DIVE_START -->
## Deep dive: how to reason about an ML problem

A strong interview answer usually starts **before** choosing a network. Ask:

1. What is the random variable we want to predict?
2. What information is available at training time and deployment time?
3. What is the loss really assuming about the data?
4. What is the unit of independence: image, patient, video, trajectory, user?
5. Which distribution shifts are plausible?
6. What metric reflects the real cost of error?

For example, suppose $y$ is a continuous target and you use MSE. You are implicitly treating deviations symmetrically and, under the maximum-likelihood interpretation, assuming Gaussian residuals with constant variance. If the residual distribution has heavy tails, MAE or a robust likelihood may be more appropriate.

### Proper data splitting

For independent samples, random splitting can be reasonable. For grouped data, split by group:


$$
\text{subject}_i \in \{\text{train},\text{val},\text{test}\},
$$


not individual scans from the same subject. For temporal forecasting, random splitting can leak future information; time-based splitting is safer.

A useful mental model is:


$$
\text{training pipeline}
=
\text{split}
\rightarrow
\text{fit preprocessing on train}
\rightarrow
\text{fit model}
\rightarrow
\text{select on val}
\rightarrow
\text{report once on test}.
$$


### Precision, recall, ROC and PR curves

For binary classification,


$$
\mathrm{Precision}=\frac{TP}{TP+FP},
\qquad
\mathrm{Recall}=\frac{TP}{TP+FN}.
$$


F1 is the harmonic mean:


$$
F1
=
2\frac{PR}{P+R}.
$$


ROC varies the decision threshold and plots TPR against FPR. PR curves are often more informative for highly imbalanced positive classes because they expose the precision cost of retrieving more positives.

### Calibration vs discrimination

Two models can rank examples equally well but assign very different confidence.

Suppose model A outputs


$$
(0.51,0.52,0.53)
$$


for three correctly ranked positives, while model B outputs


$$
(0.8,0.9,0.99).
$$


Ranking metrics may be similar, but calibration differs. This matters whenever confidence drives a downstream decision.

### Regularization is broader than adding a penalty

Regularization includes:
- explicit penalties: L1/L2;
- architectural priors: convolution, low rank, locality;
- data augmentation;
- dropout;
- early stopping;
- parameter freezing;
- limited model capacity;
- noise injection.

The general idea is not “make weights small.” It is **restrict the set of functions the optimizer can easily choose**.

### Batch size and learning rate

The mini-batch gradient


$$
g_B=\frac1B\sum_{i=1}^B\nabla_\theta \ell_i
$$


has lower variance as $B$ increases. Large batches can improve hardware efficiency but reduce gradient noise. This changes optimization dynamics, which is why batch size and learning rate are often tuned together.

### Worked example: choosing a loss

Suppose a regression model predicts a physical quantity with occasional severe outliers.

- MSE strongly emphasizes large residuals because error is squared.
- MAE grows linearly and is more robust.
- Huber combines quadratic behavior near zero with linear behavior in the tails.
- Heteroscedastic Gaussian NLL can model input-dependent uncertainty:


$$
\mathcal L
=
\frac{(y-\mu_\theta(x))^2}{2\sigma_\theta^2(x)}
+
\frac12\log \sigma_\theta^2(x).
$$


The second term prevents the model from trivially inflating uncertainty.

### Minimal implementation pattern

```python
model.train()
for x, y in train_loader:
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```

What interviewers may ask next:
- Why `zero_grad`?
- Why `set_to_none=True`?
- Where would mixed precision go?
- Where would gradient clipping go?
- Why not use the test set every epoch?
<!-- DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Always identify the probabilistic assumption behind a loss when possible.
- Split data before fitting preprocessing statistics.
- Report both discrimination/performance and calibration when probabilities matter.
- In interviews, connect optimizer choice to noise, conditioning, and weight decay rather than memorized defaults.

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
