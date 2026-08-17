# Representation & Self-Supervised Learning — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. What is an autoencoder, and why is a bottleneck useful?

An autoencoder maps input `x` to latent representation `z` and then reconstructs the input.

```math
z=E_\phi(x),
\qquad
\hat x=D_\theta(z).
```

The bottleneck restricts information flow, encouraging the encoder to capture reusable structure instead of copying every input dimension.

**Key idea.** Reconstruction alone does not guarantee semantic features. A sufficiently powerful model may preserve nuisance detail or learn shortcuts. Bottleneck size, corruption, sparsity, data augmentation, or additional objectives determine what information becomes important.

**Common follow-up.** A deterministic autoencoder is not automatically a probabilistic generative model because it does not necessarily define a simple latent distribution from which realistic new samples can be drawn.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Derive and explain the VAE ELBO.

A Variational Autoencoder (VAE) defines a latent-variable model but the exact posterior over latent variables is usually intractable. Introduce an approximate posterior `q_phi(z|x)`.

The key identity is:

```math
\log p_\theta(x)
=
\mathcal L_{\mathrm{ELBO}}
+
D_{\mathrm{KL}}
\left(
q_\phi(z|x)
\|
p_\theta(z|x)

\right).
```

Because KL divergence is nonnegative, the Evidence Lower Bound (ELBO) satisfies:

```math
\mathcal L_{\mathrm{ELBO}}
=
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x|z)]
-
D_{\mathrm{KL}}
(q_\phi(z|x)\|p(z))
\le
\log p_\theta(x).
```

**Design idea.** The reconstruction term preserves input information; the KL term makes latent codes compatible with a simple prior so sampling/interpolation are meaningful.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. What is InfoNCE really doing?

InfoNCE can be interpreted as a classification problem: given an anchor, identify which candidate is the positive match.

```math
L_i
=
-\log
\frac{\exp(s_{i,+}/\tau)}
{\sum_j\exp(s_{i,j}/\tau)}.
```

**Key idea.** The important modeling decision is not the softmax itself; it is what you call a positive pair. Two augmented views declared positive tell the model which transformations should be invariant. Negative examples tell it which samples should remain distinguishable.

**Temperature.** Smaller temperature sharpens competition; too small can overemphasize hard negatives, while too large can make the objective weakly discriminative.

This same pattern appears in SimCLR, CLIP, and dense retrieval.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. What is representation collapse and how do SSL methods prevent it?

Representation collapse means many or all inputs map to nearly the same latent vector, producing a representation with little useful information.

If the objective only says “two views of the same input should match,” a constant representation can be a trivial solution.

Contrastive methods prevent this by requiring the positive to be distinguished from negatives. Non-contrastive methods use mechanisms such as stop-gradient, teacher–student asymmetry, predictor networks, centering, or explicit variance/covariance constraints.

**Design question to ask in any SSL paper:** what prevents the constant solution? This single question often reveals the purpose of several architectural tricks.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. Masked modeling versus contrastive learning: what different biases do they create?

Masked modeling asks a representation to predict hidden content from surrounding context. It therefore emphasizes information useful for reconstruction/prediction within an example.

Contrastive learning asks the representation to identify correspondence across views or samples. Its semantics are heavily determined by augmentation and negative sampling.

Neither is universally better. MAE-style reconstruction often works well for visual representation learning; CLIP-style contrastive alignment is ideal when a shared image–language semantic space is desired.

**Key principle.** The pretext objective defines which information the representation is encouraged to retain or ignore.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

