# Diffusion & Score-Based Models — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. Explain the key idea of the diffusion forward process.

The forward process deliberately makes the data distribution simpler by gradually adding Gaussian noise. At time zero the sample is structured data; after many steps it approaches a simple Gaussian distribution.

A key analytic result is that any noisy time point can be sampled directly:

```math
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
```

**Why this design?** The corruption process is known exactly. Training pairs `(x_t, noise)` can be generated at arbitrary noise levels without simulating all previous steps. The model learns the local reverse direction for every noise scale.

The broader design philosophy is to replace one difficult global mapping with many easier denoising problems.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Why does predicting noise correspond to score estimation?

For Gaussian corruption, the conditional score of a noisy sample is proportional to the negative injected noise:

```math

\nabla_{x_t}\log q(x_t|x_0)
=
-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
```

Therefore a network trained to predict `epsilon` is, up to a known scale, learning a vector field related to the score of the noisy distribution.

**Key idea.** The score tells each noisy point which direction increases probability density. Reverse diffusion uses these local directions while moving from a simple noise distribution back toward the data distribution.

This connection is why denoising, score matching, and reverse-time SDE formulations are different views of the same underlying process.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. DDPM versus DDIM: what changes?

A Denoising Diffusion Probabilistic Model (DDPM) uses a stochastic reverse Markov process. DDIM constructs an alternative reverse process that can use the same trained denoiser but follow a deterministic path when stochasticity is set to zero.

**Key idea.** Model training and numerical sampling are partly separable. You can change the integration/sampling path without replacing the learned network.

**Tradeoff.** Fewer deterministic steps can accelerate sampling but may affect diversity or quality depending on the model and schedule.

**Interview trap.** DDIM is not simply “a different network architecture”; it is primarily a different sampling formulation.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. What is classifier-free guidance doing conceptually?

Classifier-Free Guidance (CFG) trains the model with and without a condition, then extrapolates from the unconditional prediction toward the conditional one:

```math
\epsilon_{\mathrm{guided}}
=
\epsilon_{\mathrm{uncond}}
+
w
(
\epsilon_{\mathrm{cond}}
-
\epsilon_{\mathrm{uncond}}
).
```

The difference term represents how the condition changes the denoising direction. Multiplying it by guidance scale `w` increases conditional fidelity.

**Tradeoff.** Large guidance can reduce diversity, exaggerate features, or introduce artifacts. Guidance strength is therefore a fidelity-versus-diversity control, not a free accuracy improvement.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. Why use latent diffusion?

Pixel-space diffusion is expensive because every denoising step processes a full-resolution image. Latent diffusion first compresses the image with an autoencoder, performs diffusion on a smaller latent grid, then decodes the result.

**Key idea.** Spend iterative generative compute on a compact semantic representation rather than every raw pixel.

**Tradeoff.** The autoencoder becomes a hard information bottleneck. Fine detail lost during encoding cannot be reconstructed by the diffusion process. Latent resolution therefore affects the best achievable output fidelity.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. What is a Diffusion Transformer (DiT), and what stays the same?

A Diffusion Transformer (DiT) replaces the U-Net-style denoising network with Transformer blocks over image or latent tokens.

What changes is the **architecture used to parameterize the denoiser**.

What stays the same is the **diffusion learning problem**: the network still receives a noisy state and time/conditioning information and predicts noise, velocity, x0, or another reverse-process quantity.

This distinction is important in interviews because “diffusion” describes the generative process/objective, while U-Net or Transformer describes the neural architecture implementing the denoising function.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

