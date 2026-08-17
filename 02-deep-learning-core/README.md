# Deep Learning Core

**Status:** Strong / Review

## Why this matters

This chapter is the common language behind CNNs, Transformers, diffusion models, and modern foundation models.

## Learning objectives

- Explain backpropagation as repeated chain rule.
- Reason about vanishing/exploding gradients and residual connections.
- Compare normalization, initialization, and mixed-precision choices.

## Chapter map

- MLP, activation functions, universal approximation intuition
- Backpropagation and computational graphs
- Vanishing/exploding gradients
- Residual connections and optimization depth
- BatchNorm vs LayerNorm vs GroupNorm
- Dropout and stochastic regularization
- Initialization: Xavier/Glorot, He/Kaiming
- Mixed precision and numerical stability


---

## Core concepts and theory

### 1. Backpropagation

For a composition


$$
y=f_L(f_{L-1}(\cdots f_1(x))),
$$


the chain rule gives


$$
\frac{\partial \mathcal L}{\partial h_l}
=
\frac{\partial \mathcal L}{\partial h_{l+1}}
\frac{\partial h_{l+1}}{\partial h_l}.
$$


Backpropagation is dynamic programming for repeatedly applying this chain rule while reusing intermediate derivatives.

---

### 2. Vanishing and Exploding Gradients

For a deep linearized network,


$$
\frac{\partial h_L}{\partial h_0}
=
\prod_{l=1}^L J_l.
$$


If typical singular values of $J_l$ are below 1, the product shrinks exponentially; above 1, it can explode.

Solutions:
- residual connections;
- normalization;
- careful initialization;
- gated architectures;
- gradient clipping.

---

### 3. Residual Connections

A residual block:


$$
y=x+F(x).
$$


Derivative:


$$
\frac{\partial y}{\partial x}
=
I+\frac{\partial F}{\partial x}.
$$


The identity path allows gradient flow even if the residual branch has poor conditioning. It also biases optimization toward learning corrections to an identity mapping.

---

### 4. Initialization

For layer


$$
y_j=\sum_{i=1}^{n}w_{ji}x_i,
$$


assuming independent zero-mean terms,


$$
\mathrm{Var}(y_j)\approx n\,\mathrm{Var}(w)\mathrm{Var}(x).
$$


To preserve variance, choose


$$
\mathrm{Var}(w)\propto \frac1n.
$$


Xavier/Glorot accounts for fan-in and fan-out. He/Kaiming adjusts for ReLU, which zeroes roughly half the activations.

---

### 5. BatchNorm vs LayerNorm

#### BatchNorm
For each feature/channel, normalize across the batch/spatial axes.

#### LayerNorm
For each sample/token, normalize across hidden features.

Transformers favor LayerNorm because:
- sequence lengths/batches vary;
- token-wise normalization does not depend on batch statistics;
- autoregressive inference may use batch size 1.

---

### 6. Mixed Precision

FP16/BF16 reduce memory and increase tensor-core throughput.

Main issue: limited numerical range/precision.

Typical safeguards:
- keep selected reductions in FP32;
- loss scaling for FP16;
- BF16 is often more robust because it has FP32-like exponent range.

<!-- DEEP_DIVE_START -->
## Deep dive: optimization through a deep network

Consider a layer


$$
h_{l+1}=\phi(W_lh_l+b_l).
$$


The Jacobian is


$$
J_l
=
\frac{\partial h_{l+1}}{\partial h_l}
=
D_{\phi'(u_l)}W_l.
$$


Across many layers,


$$
\frac{\partial h_L}{\partial h_0}
=
J_{L-1}J_{L-2}\cdots J_0.
$$


The product of many Jacobians explains why singular values matter. If the typical gain is $0.8$, after 50 layers it becomes roughly


$$
0.8^{50}\approx1.4\times10^{-5}.
$$


This is the basic vanishing-gradient intuition.

### Residual blocks as perturbations of identity

Residual block:


$$
h_{l+1}=h_l+F_l(h_l).
$$


If $F_l$ initially learns a small correction, then


$$
J_l
=
I+\frac{\partial F_l}{\partial h_l},
$$


so signal and gradients do not need to pass exclusively through a long product of arbitrary transforms.

This is why the statement “skip connections preserve detail” is incomplete. In ResNet, their major role is also **optimization and conditioning**.

### Activation functions

#### Sigmoid


$$
\sigma(x)=\frac1{1+e^{-x}}.
$$


Derivative:


$$
\sigma'(x)=\sigma(x)(1-\sigma(x)).
$$


It saturates for large $|x|$, creating small gradients.

#### ReLU


$$
\mathrm{ReLU}(x)=\max(0,x).
$$


Cheap and non-saturating for positive inputs, but units can die if they remain negative.

#### GELU
A smooth gating-like activation widely used in Transformers:


$$
\mathrm{GELU}(x)=x\Phi(x).
$$


Interpretation: instead of a hard threshold at zero, it smoothly weights the input by how likely it is to be positive under a standard Gaussian.

### Normalization: what axis?

A good way to remember normalization methods is to ask **which dimensions share statistics**.

- BatchNorm: statistics across batch/spatial examples for each channel.
- LayerNorm: statistics across hidden features within each token/sample.
- GroupNorm: groups of channels within each sample.

This axis-level reasoning is more reliable than memorizing names.

### Pre-norm Transformer pattern

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

The identity residual path stays clean. This generally improves optimization stability in very deep Transformer stacks.

### Gradient clipping

Global-norm clipping:


$$
g\leftarrow
g\cdot
\min\left(1,\frac{\tau}{\|g\|_2}\right).
$$


It does not solve bad optimization fundamentally, but prevents rare extreme updates from destabilizing training.

### Mixed precision mental model

You do not need every tensor in the same dtype.

Typical idea:
- weights/activations in BF16 or FP16;
- sensitive reductions/accumulation may use FP32;
- optimizer master states may remain FP32.

The objective is not “use half precision everywhere”; it is to reduce expensive memory/compute while preserving numerical stability.
<!-- DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Track tensor shapes explicitly during forward and backward reasoning.
- Residual paths, normalization, and initialization all help conditioning, but they solve different pieces of the optimization problem.
- Mixed precision is both a numerical and systems topic: know which quantities are safe in low precision.

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
