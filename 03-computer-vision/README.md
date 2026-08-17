# CNN & Computer Vision

**Status:** Strong / Review

## Why this matters

CNNs remain core interview material even for ViT/VLM roles, and many vision systems mix convolutional and Transformer components.

## Learning objectives

- Compute convolution output shapes and receptive fields.
- Explain CNN inductive bias and U-Net/ResNet design.
- Compare CNN and ViT modeling assumptions.

## Chapter map

- Convolution, receptive field, stride, padding, dilation
- CNN inductive biases: locality and translation equivariance
- ResNet and residual blocks
- U-Net and skip connections
- Classification, segmentation, detection, reconstruction
- Data augmentation and invariances
- CNN vs ViT tradeoffs


---

## Core concepts and theory

### 1. Convolution

For 2D input \(x\) and kernel \(w\),

\[
y[i,j]=\sum_{u,v} w[u,v]x[i+u,j+v].
\]

Weight sharing means the same detector is applied across the image.

Two core inductive biases:
- locality;
- translation equivariance.

---

### 2. Output Size

For one dimension,

\[
L_{\rm out}
=
\left\lfloor
\frac{L_{\rm in}+2P-D(K-1)-1}{S}+1
\right\rfloor.
\]

Where:
- \(K\): kernel size;
- \(P\): padding;
- \(S\): stride;
- \(D\): dilation.

---

### 3. Receptive Field

Stacking local convolutions increases effective receptive field.

For stride 1 and kernel size \(K\), \(L\) layers approximately produce

\[
R=1+L(K-1).
\]

Strides and dilation grow it faster.

---

### 4. U-Net Skip Connections

Encoder features contain spatial detail at multiple resolutions. Decoder upsampling reconstructs semantic outputs but may lose precise localization.

U-Net concatenates encoder and decoder features:

\[
h^{\rm dec}_l
=
F([h^{\rm dec}_{l+1},h^{\rm enc}_l]).
\]

This combines:
- high-level context;
- high-resolution localization.

---

### 5. CNN vs ViT

CNN:
- stronger locality/equivariance priors;
- efficient on moderate data;
- hierarchical feature maps naturally emerge.

ViT:
- weaker spatial prior;
- global interaction from self-attention;
- scales strongly with data/model size;
- unifies vision with token-based architectures.

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Be able to compute output sizes by hand.
- For U-Net, state clearly what information skip connections preserve.
- When comparing CNN and ViT, discuss inductive bias, data scale, global interaction, and compute—not only architecture names.

---

## Hands-on / practice

## Level 1 — Reproduce
Implement or run a canonical example that demonstrates the central idea.

## Level 2 — Compare
Create at least one controlled comparison (baseline vs method, accuracy vs compute, or full vs efficient version).

## Level 3 — Explain
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
