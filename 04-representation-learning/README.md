# Representation & Self-Supervised Learning

**Status:** Strong + Fill Gaps

## Why this matters

Foundation models depend on learning transferable representations before downstream adaptation.

## Learning objectives

- Derive the VAE ELBO and reparameterization trick.
- Explain InfoNCE and contrastive learning.
- Understand representation collapse and self-supervised learning.

## Chapter map

- Autoencoders and bottleneck representations
- Variational autoencoders and ELBO
- Contrastive learning and InfoNCE
- Positive/negative pairs and representation collapse
- Masked modeling
- Metric learning and embedding geometry
- Self-supervised pretraining and transfer


---

## Core concepts and theory

### 1. Autoencoder


$$
z=E_\phi(x),\qquad \hat x=D_\theta(z).
$$


Train with


$$
\mathcal L=\|x-\hat x\|^2.
$$


A bottleneck encourages $z$ to retain useful structure, but reconstruction alone does not guarantee semantic representations.

---

### 2. VAE and ELBO

Latent-variable model:


$$
p_\theta(x,z)=p(z)p_\theta(x|z).
$$


We want


$$
\log p_\theta(x)
=
\log\int p_\theta(x,z)\,dz,
$$


which is often intractable.

Introduce approximate posterior $q_\phi(z|x)$:


$$
\log p_\theta(x)
=
\mathcal L_{\rm ELBO}
+
D_{\rm KL}(q_\phi(z|x)\|p_\theta(z|x)).
$$


Therefore


$$
\log p_\theta(x)\ge \mathcal L_{\rm ELBO},
$$


with


$$
\mathcal L_{\rm ELBO}
=
\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]
-
D_{\rm KL}(q_\phi(z|x)\|p(z)).
$$


Interpretation:
- reconstruction term preserves information;
- KL regularizes latent distribution toward prior.

---

### 3. Reparameterization Trick

For Gaussian encoder


$$
q_\phi(z|x)=\mathcal N(\mu_\phi(x),\sigma_\phi^2(x)),
$$


sample via


$$
\epsilon\sim\mathcal N(0,I),\qquad
z=\mu+\sigma\odot\epsilon.
$$


Randomness moves outside the parameterized path, enabling backpropagation.

---

### 4. Contrastive Learning / InfoNCE

For anchor $i$, positive $j$:


$$
\mathcal L_i
=
-\log
\frac{\exp(\mathrm{sim}(z_i,z_j)/\tau)}
{\sum_k \exp(\mathrm{sim}(z_i,z_k)/\tau)}.
$$


The loss:
- pulls positives together;
- pushes competing samples apart;
- temperature $\tau$ controls concentration.

---

### 5. Representation Collapse

A trivial solution $z_i=c$ for all samples destroys information.

Contrastive negatives prevent collapse directly. Other SSL methods use:
- stop-gradient;
- predictor asymmetry;
- variance/covariance regularization;
- teacher-student targets.

<!-- DEEP_DIVE_START -->
## Deep dive: what makes a representation transferable?

A representation $z=f(x)$ is useful when it preserves information relevant to many downstream tasks while discarding nuisance variation.

This creates tension:


$$
\text{invariance}
\quad \text{vs} \quad
\text{information preservation}.
$$


If augmentation says two views should map closely, the model is encouraged to become invariant to whatever differs between those views. Therefore augmentation design implicitly defines the desired invariances.

### InfoNCE as classification

For one anchor, choosing the positive among $B$ candidates can be viewed as a $B$-way classification problem.

Logit for candidate $j$:


$$
\ell_j=\frac{z_i^\top z_j}{\tau}.
$$


Cross entropy over these logits yields InfoNCE.

This makes contrastive learning easy to connect to ordinary supervised classification.

### Temperature

Small $\tau$ sharpens similarity differences:


$$
p_j
=
\frac{\exp(s_j/\tau)}
{\sum_k\exp(s_k/\tau)}.
$$


If $\tau$ is too small, optimization may become dominated by very hard negatives. If too large, similarities become insufficiently discriminative.

### VAE posterior collapse

ELBO:


$$
\mathbb E_q[\log p_\theta(x|z)]
-
D_{\rm KL}(q_\phi(z|x)\|p(z)).
$$


If the decoder is extremely powerful, it may model $x$ while ignoring $z$. Then


$$
q_\phi(z|x)\approx p(z)
$$


and the KL becomes near zero. This is posterior collapse.

Possible strategies:
- KL annealing;
- weaker decoder;
- free bits;
- better latent architecture.

### Self-supervised objectives as choices of prediction target

SSL families can be organized by what is predicted:

- contrastive: identify corresponding views;
- masked modeling: reconstruct missing content;
- teacher-student: match target representation;
- predictive representation: predict future/hidden latent features.

This taxonomy connects directly to MAE, DINO, JEPA, and world models.
<!-- DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- For contrastive learning, define positives, negatives, similarity function, and temperature.
- For VAE, distinguish approximate posterior, prior, likelihood, and generative sampling path.
- Always ask what representation properties the pretext objective encourages.

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
