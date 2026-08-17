# Machine Learning Foundations — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. What is the difference between supervised, unsupervised, and self-supervised learning?

**Short answer.** Supervised learning has externally provided targets, unsupervised learning models structure without those targets, and self-supervised learning creates the prediction target from the data itself.

**Key idea.** The distinction is where the learning signal comes from. In supervised classification, an image may come with a human label such as “cat.” In self-supervised learning, the same image can generate its own target—for example, reconstruct a masked patch or make two augmentations of the same image have similar representations. Self-supervision is especially important for foundation models because unlabeled raw data are much more abundant than carefully annotated data.

**Common misconception.** Self-supervised does not mean “there are no targets.” The targets are automatically constructed from the raw input.

**Examples.** Next-token prediction, masked language modeling, masked image modeling, and contrastive learning are all self-supervised objectives.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Why does mean-squared error correspond to a Gaussian noise model?

Assume the target is generated as `y = f_theta(x) + epsilon`, where the residual noise is Gaussian with variance `sigma^2`. The conditional likelihood is Gaussian, and the negative log-likelihood becomes a squared residual plus constants:

```math
-\log p(y|x,\theta)
=
\frac{(y-f_\theta(x))^2}{2\sigma^2}
+
C.
```

Therefore minimizing mean-squared error is equivalent to maximum-likelihood estimation under a Gaussian residual model.

**Design idea.** A loss encodes an assumption. If residuals have heavy tails, mean absolute error or Huber loss can be more robust. If uncertainty changes with the input, a heteroscedastic likelihood that predicts both mean and variance may be more appropriate.

**Interview follow-up.** Do not answer only “MSE penalizes large errors.” Explain the statistical model that makes MSE natural.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. Explain bias–variance tradeoff and how regularization changes it.

**Bias** is systematic error from an overly restricted model; **variance** is sensitivity to which training sample happened to be observed. Increasing capacity tends to reduce bias but can increase variance.

Regularization changes the functions the optimizer can easily choose. L2 penalties discourage large weights, dropout injects stochastic constraints, data augmentation enforces invariances, early stopping limits optimization, and parameter freezing/PEFT can constrain adaptation.

**Key idea.** Regularization is broader than adding a penalty term. It is any modeling or optimization choice that biases the learner toward solutions expected to generalize.

**Modern nuance.** Very large pretrained models complicate the classical picture: parameter count alone does not determine effective complexity because pretraining, data scale, optimizer behavior, and constrained fine-tuning strongly shape the solution.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. When should I care about ROC-AUC versus PR-AUC?

ROC-AUC summarizes the Receiver Operating Characteristic curve, which plots true-positive rate against false-positive rate across thresholds. PR-AUC summarizes the Precision–Recall curve.

When positives are rare, ROC-AUC can look optimistic because false-positive rate divides by a very large number of negatives. Precision asks a more operational question: among predicted positives, how many are actually positive?

**Example.** If disease prevalence is 1%, a classifier can produce many false alarms while still having a low false-positive rate. Precision will expose that problem.

**Rule of thumb.** For strongly imbalanced positive classes, PR-AUC is often more informative. But the real answer is always to choose metrics based on the cost of false positives and false negatives.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. What is calibration, and how is it different from accuracy?

A calibrated classifier's confidence should match empirical frequency. Among predictions made with confidence 0.8, roughly 80% should be correct.

Accuracy, ranking, and calibration are different properties. A model can rank examples well but be systematically overconfident. Conversely, a less accurate model can still assign well-calibrated probabilities.

Calibration matters when probability controls downstream action—risk scoring, triage, abstention, uncertainty thresholds, or decision support. Common metrics include Negative Log-Likelihood (NLL), Brier score, and Expected Calibration Error (ECE).

**Useful follow-up.** Temperature scaling can recalibrate logits without changing class ranking, which illustrates why discrimination and calibration are distinct.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. Adam versus AdamW: what is the real difference?

Adam rescales each gradient coordinate using moving estimates of first and second moments. If L2 regularization is added directly to the loss, that regularization gradient is also transformed by Adam's adaptive scaling.

AdamW applies weight decay directly to the parameters, decoupled from the adaptive gradient update:

```math
\theta_{t+1}
=
(1-\eta\lambda)\theta_t
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
```

**Key idea.** In adaptive optimizers, “L2 penalty” and “weight decay” are not exactly the same operation. AdamW makes the shrinkage explicit and is a common default in Transformer training.

**Follow-up.** Be ready to explain learning rate, beta1/beta2, epsilon, and why optimizer choice does not replace learning-rate tuning.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

