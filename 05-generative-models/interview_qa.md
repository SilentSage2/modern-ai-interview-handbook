# Generative Models — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. Compare autoregressive models, VAEs, GANs, flows, and diffusion models.

Compare them along likelihood, sampling speed, training stability, and architectural constraints.

- **Autoregressive:** exact factorized likelihood and stable maximum-likelihood training; sampling is sequential.
- **VAE:** variational likelihood objective and fast decoder sampling; approximate inference and likelihood choice can blur details.
- **GAN:** fast one-pass sampling and sharp outputs; adversarial optimization can be unstable and exact likelihood is unavailable.
- **Normalizing flow:** exact likelihood and invertibility; transformations must have tractable Jacobian determinants.
- **Diffusion:** stable denoising/score training and strong sample quality; iterative sampling is expensive.

The interview goal is not to pick a universal winner but to explain which compromise matches the application.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Why is the GAN discriminator related to a density ratio?

For a fixed generator, the discriminator maximizes:

```math
p_{\mathrm{data}}(x)\log D(x)
+
p_g(x)\log(1-D(x)).
```

Optimizing pointwise gives:

```math
D^*(x)
=
\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_g(x)}.
```

Therefore the optimal discriminator contains information about how real and generated densities differ.

**Key idea.** GANs replace explicit likelihood modeling with a learned discrimination problem. The generator improves by making generated samples harder to distinguish from real data.

**Failure mode.** The generator and discriminator form a game, so optimization can oscillate, become imbalanced, or collapse to limited modes.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. Why are latent-variable models useful?

A latent-variable model assumes observations can be explained by lower-dimensional hidden factors:

```math
z\sim p(z),
\qquad
x\sim p_\theta(x|z).
```

This can separate high-level factors from observation noise and provide a compact space for interpolation, control, or conditioning.

The hard part is inference: given observed `x`, what latent `z` explains it? VAEs learn an approximate posterior; flows obtain an exact inverse through invertibility; latent diffusion first learns an autoencoder and then models the latent distribution.

**Design lesson.** Latent modeling is as much about choosing useful hidden variables as about generating data.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. What does a normalizing flow gain and what constraint does it pay?

A normalizing flow uses an invertible mapping between a simple latent distribution and data. Change of variables provides exact likelihood:

```math
\log p_X(x)
=
\log p_Z(f^{-1}(x))
+
\log
\left|
\det
\frac{\partial f^{-1}(x)}{\partial x}

ight|.
```

**Gain:** exact density evaluation and exact latent inversion.

**Cost:** the architecture must remain invertible and its Jacobian determinant must be tractable, which restricts design freedom.

This is a clear example of mathematics shaping architecture: exact likelihood is obtained by paying with structural constraints.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. Likelihood versus visual sample quality: why are they not identical?

A model can achieve good likelihood by covering many modes and modeling low-level statistics while producing samples that humans judge mediocre. Conversely, a GAN can generate visually sharp examples while missing parts of the data distribution.

Likelihood rewards probability coverage; perceptual sample quality is a different property.

Therefore generative evaluation often needs multiple views:
- likelihood or reconstruction statistics;
- sample fidelity;
- diversity/coverage;
- downstream task utility;
- human or domain-specific evaluation.

A single scalar metric rarely captures the whole generative distribution.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

