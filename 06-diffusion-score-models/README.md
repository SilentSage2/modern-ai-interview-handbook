# Diffusion & Score-Based Models

**Status:** Strong

## Why this matters

Diffusion is already a strength, so this chapter is organized as a concise but rigorous interview reference that connects DDPM, score models, SDEs, latent diffusion, DiT, and newer flow-based views.

## Learning objectives

- Derive the closed-form forward process and noise-prediction objective.
- Connect epsilon prediction to score estimation.
- Explain DDIM, CFG, SDE/ODE views, latent diffusion, and DiT.

## Chapter map

- Forward noising and reverse denoising
- DDPM objective and epsilon/x0/v prediction
- Score matching
- DDIM and non-Markovian sampling
- SDE/ODE views of diffusion
- Conditional diffusion and classifier-free guidance
- Latent diffusion
- Diffusion Transformers (DiT)
- Posterior sampling / inverse problems
- Flow matching and rectified-flow connections


---

## Core concepts and theory

### 1. Forward Diffusion

DDPM defines

\[
q(x_t|x_{t-1})
=
\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t I).
\]

Let

\[
\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{s=1}^t\alpha_s.
\]

Then the closed form is

\[
q(x_t|x_0)
=
\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),
\]

so

\[
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\quad
\epsilon\sim\mathcal N(0,I).
\]

This lets us sample arbitrary noise levels without simulating all previous steps.

---

### 2. Reverse Process

Learn

\[
p_\theta(x_{t-1}|x_t)
=
\mathcal N(\mu_\theta(x_t,t),\Sigma_t).
\]

A common parameterization predicts the forward noise

\[
\epsilon_\theta(x_t,t).
\]

The simplified training objective is

\[
\mathcal L_{\rm simple}
=
\mathbb E_{x_0,t,\epsilon}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2
\right].
\]

---

### 3. Score Connection

For Gaussian corruption,

\[
\nabla_{x_t}\log q(x_t|x_0)
=
-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
\]

Therefore predicting \(\epsilon\) is equivalent, up to a scale factor, to estimating the score.

---

### 4. DDIM

DDIM constructs a non-Markovian reverse process sharing the same training objective as DDPM.

With stochasticity parameter \(\eta=0\), sampling becomes deterministic for fixed initial noise and can use fewer steps.

---

### 5. Classifier-Free Guidance

Train conditionally and with condition dropout.

At inference,

\[
\epsilon_{\rm guided}
=
\epsilon_{\rm uncond}
+
w(\epsilon_{\rm cond}-\epsilon_{\rm uncond}).
\]

Higher \(w\) strengthens conditional fidelity but can reduce diversity and cause artifacts.

---

### 6. SDE View

Forward SDE:

\[
dx=f(x,t)dt+g(t)dW_t.
\]

Reverse-time SDE:

\[
dx=
\left[f(x,t)-g(t)^2\nabla_x\log p_t(x)\right]dt
+
g(t)d\bar W_t.
\]

Once the score is known, reverse-time simulation generates samples.

---

### 7. Probability-Flow ODE

Associated deterministic ODE:

\[
dx=
\left[
f(x,t)-\frac12g(t)^2\nabla_x\log p_t(x)
\right]dt.
\]

It shares the same marginal distributions as the SDE.

---

### 8. Latent Diffusion

Instead of applying diffusion in pixel space:

\[
x\overset{E}{\rightarrow} z,
\]

run diffusion in compressed latent \(z\), then decode:

\[
z_0\overset{D}{\rightarrow}\hat x.
\]

This reduces spatial dimensionality and computational cost.

---

### 9. Diffusion Transformer

DiT replaces the U-Net denoiser with Transformer blocks over latent/image tokens.

The important distinction:
- diffusion defines the generative process/objective;
- Transformer defines the denoising network architecture.

<!-- DEEP_DIVE_START -->
## Deep dive: parameterizations and modern connections

### Predicting \(\epsilon\), \(x_0\), or \(v\)

From

\[
x_t
=
\alpha_t x_0+\sigma_t\epsilon,
\]

if the network predicts \(\epsilon\),

\[
\hat x_0
=
\frac{x_t-\sigma_t\hat\epsilon}{\alpha_t}.
\]

If it predicts \(x_0\), recover noise:

\[
\hat\epsilon
=
\frac{x_t-\alpha_t\hat x_0}{\sigma_t}.
\]

Different parameterizations emphasize different signal-to-noise regimes and numerical conditioning.

### Why denoising learns a distribution

At each noise level, the model learns how noisy samples should move toward regions of higher data probability. Across all \(t\), these local denoising directions define a path from a simple prior to the data distribution.

This is a useful intuition for score models:

\[
s_\theta(x_t,t)
\approx
\nabla_{x_t}\log p_t(x_t).
\]

### Guidance as score modification

Classifier-free guidance combines conditional and unconditional predictions:

\[
s_{\rm guided}
=
s_{\rm uncond}
+
w(s_{\rm cond}-s_{\rm uncond}).
\]

The difference term points toward features that are more compatible with the condition.

### Latent diffusion tradeoff

Compression reduces cost:

\[
H\times W\times C
\rightarrow
h\times w\times c,\qquad hw\ll HW.
\]

But the autoencoder becomes part of the generative model. Information lost by the encoder cannot be recovered by diffusion.

### Diffusion vs flow matching

A useful high-level distinction:

- diffusion/score training often learns denoising or score fields associated with noisy marginals;
- flow matching directly learns a time-dependent velocity field for transporting samples along a probability path.

Both can be understood as learning dynamics that transform an easy distribution into the data distribution.

### Worked implementation sketch

```python
t = torch.randint(0, T, (B,), device=x0.device)
eps = torch.randn_like(x0)

a = alpha_bar[t].sqrt().view(B, 1, 1, 1)
s = (1 - alpha_bar[t]).sqrt().view(B, 1, 1, 1)

xt = a * x0 + s * eps
eps_hat = model(xt, t, cond)
loss = (eps_hat - eps).square().mean()
```

Interview follow-up:
- Why can \(x_t\) be sampled directly?
- What information does time embedding provide?
- Why does sampling still require a reverse-time solver?
<!-- DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Keep forward process, reverse process, parameterization, and sampler conceptually separate.
- Know whether a statement refers to epsilon prediction, x0 prediction, v prediction, or score estimation.
- Distinguish diffusion as a generative process from U-Net/Transformer as denoiser architectures.

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
