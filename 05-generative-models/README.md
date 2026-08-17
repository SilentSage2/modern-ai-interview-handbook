# Generative Models

**Status:** Strong / Review

## Why this matters

Diffusion, world models, and multimodal generation are easier to understand when autoregressive, VAE, GAN, flow, and energy-based formulations are placed in one taxonomy.

## Learning objectives

- Compare likelihood-based and implicit generative models.
- Derive the GAN objective intuition and flow change-of-variables formula.
- Know the tradeoffs among major generative-model families.

## Chapter map

- Autoregressive factorization
- VAE: latent-variable modeling and ELBO
- GAN: generator/discriminator minimax game
- Normalizing flows and exact likelihood
- Energy-based modeling
- Likelihood vs sample-quality tradeoffs
- Conditional generation


---

## Core concepts and theory

### 1. Autoregressive Modeling

Factorize


```math
p(x_1,\ldots,x_T)
=
\prod_{t=1}^T p(x_t|x_{<t}).
```


Pros:
- exact likelihood;
- stable maximum-likelihood training.

Cons:
- sequential sampling.

---

### 2. VAE

See representation module for ELBO. VAEs trade exact likelihood for tractable variational optimization.

Typical issue: blurry samples when Gaussian reconstruction likelihood strongly penalizes pixelwise errors.

---

### 3. GAN

Generator $G(z)$, discriminator $D(x)$:


```math
\min_G\max_D
\mathbb E_{x\sim p_{\rm data}}[\log D(x)]
+
\mathbb E_{z\sim p(z)}[\log(1-D(G(z)))].
```


For fixed $G$, optimal discriminator is


```math
D^*(x)
=
\frac{p_{\rm data}(x)}
{p_{\rm data}(x)+p_g(x)}.
```


Plugging this back connects GAN training to Jensen–Shannon divergence.

Main challenge: adversarial optimization is a game, not ordinary minimization.

---

### 4. Normalizing Flow

Invertible transform


```math
x=f_\theta(z),\qquad z=f_\theta^{-1}(x).
```


Change of variables:


```math
\log p_X(x)
=
\log p_Z(z)
+
\log\left|
\det\frac{\partial f_\theta^{-1}(x)}{\partial x}
\right|.
```


Pros:
- exact likelihood;
- exact latent inference.

Constraint:
- architecture must make Jacobian determinant tractable.

---

### 5. Energy-Based Models

Define unnormalized density


```math
p_\theta(x)
=
\frac{e^{-E_\theta(x)}}{Z_\theta}.
```


Challenge: partition function


```math
Z_\theta=\int e^{-E_\theta(x)}dx
```


is generally intractable.

<!-- DEEP_DIVE_START -->
## Deep dive: a unified generative-model view

Every generative model tries to represent or sample from a target distribution


```math
p_{\rm data}(x),
```


but differs in how probability, latent variables, and sampling are handled.

| Family | Likelihood | Sampling | Core difficulty |
|---|---|---|---|
| Autoregressive | exact factorized | sequential | slow generation |
| VAE | variational lower bound | fast decoder | approximate inference / blurry likelihood |
| GAN | implicit | fast | unstable adversarial training |
| Flow | exact | fast/invertible | restrictive invertible architecture |
| Diffusion | variational/score view | iterative | many denoising steps |

### Autoregressive ordering

For images, one can choose an ordering:


```math
p(x)=\prod_i p(x_i|x_{<i}).
```


The factorization is mathematically valid for any ordering, but architecture and computational efficiency depend strongly on it.

### GAN discriminator derivation

For fixed generator, maximize pointwise:


```math
p_{\rm data}(x)\log D(x)+p_g(x)\log(1-D(x)).
```


Differentiate with respect to $D(x)$:


```math
\frac{p_{\rm data}}{D}
-
\frac{p_g}{1-D}=0.
```


Solve:


```math
D^*(x)
=
\frac{p_{\rm data}(x)}
{p_{\rm data}(x)+p_g(x)}.
```


Thus the optimal discriminator estimates a density-ratio-like quantity.

### Latent variables

Many generative models introduce


```math
z\sim p(z),\qquad x\sim p_\theta(x|z).
```


The modeling question becomes:

> What variations should be represented in $z$, and how can we infer $z$ from observed $x$?

That question reappears in VAEs, latent diffusion, and world models.
<!-- DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Compare methods along likelihood tractability, sampling speed, training stability, and sample quality.
- Distinguish the probabilistic model from the neural-network architecture used to parameterize it.
- Be ready to explain when latent-variable modeling is useful.

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
