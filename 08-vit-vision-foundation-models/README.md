# ViT & Vision Foundation Models

**Status:** Priority

## Why this matters

ViT is the main bridge from classical computer vision to multimodal foundation models.

## Learning objectives

- Convert images into patch tokens and derive token/attention scaling.
- Explain CLS token, positional embeddings, MAE, DINO, CLIP, and SAM.
- Compare ViT and CNN inductive biases and fine-tuning behavior.

## Chapter map

- Patchification and patch embeddings
- CLS token and positional embeddings
- ViT encoder architecture
- ViT vs CNN inductive bias
- Patch size vs compute/resolution
- MAE-style masked image modeling
- DINO/self-distillation style representation learning
- CLIP-style vision-language alignment
- SAM-style promptable vision foundation models
- Fine-tuning and linear probing of pretrained vision encoders


---

## Core concepts and theory

### 1. Image to Tokens

Input image:

\[
x\in\mathbb R^{H\times W\times C}.
\]

Patch size \(P\times P\).

Number of patches:

\[
N=\frac{HW}{P^2}.
\]

Each patch has

\[
P^2C
\]

values.

Flatten patch \(x_p^{(i)}\) and project:

\[
z_i=x_p^{(i)}E,
\qquad
E\in\mathbb R^{(P^2C)\times d}.
\]

Now the image becomes a token sequence:

\[
Z\in\mathbb R^{N\times d}.
\]

---

### 2. Patch Embedding as Convolution

Patchification + linear projection can be implemented with Conv2D:

- kernel size \(P\);
- stride \(P\);
- output channels \(d\).

Why?

Each convolutional kernel sees exactly one non-overlapping patch and produces its embedding.

This is a useful implementation equivalence.

---

### 3. CLS Token

Add learnable

\[
z_{\rm cls}\in\mathbb R^d
\]

to the token sequence.

After all Transformer blocks, use its final representation:

\[
h_{\rm cls}
\]

for classification.

Alternative modern designs may use global average pooling instead.

---

### 4. Positional Embeddings

Patch tokens need location information.

Input:

\[
Z_0=
[z_{\rm cls};z_1;\ldots;z_N]
+
E_{\rm pos}.
\]

If fine-tuning at a different resolution, positional embeddings may need interpolation.

---

### 5. Complexity and Patch Size

Number of tokens:

\[
N=\frac{HW}{P^2}.
\]

Attention cost:

\[
O(N^2d).
\]

Therefore halving patch size approximately quadruples token count and can increase attention cost by about \(16\times\).

This is why patch size has a large compute-resolution tradeoff.

---

### 6. CNN vs ViT Inductive Bias

CNN assumes:
- local interactions;
- translation-equivariant filters;
- hierarchical composition.

ViT assumes much less:
- patch tokenization;
- positional information;
- global learned interaction.

Consequence:
- CNNs often learn well with less data;
- ViTs can benefit more strongly from large-scale pretraining.

---

### 7. MAE

Masked Autoencoder pipeline:

1. mask a large fraction of image patches;
2. encode only visible patches;
3. lightweight decoder reconstructs masked patches.

Objective:

\[
\mathcal L
=
\frac1{|\mathcal M|}
\sum_{i\in\mathcal M}
\|x_i-\hat x_i\|^2.
\]

Why high masking ratio can work:
natural images are highly redundant, so reconstruction forces learning broader structure rather than copying local texture.

---

### 8. DINO-Style Self-Distillation

Student and teacher receive different image views.

Teacher parameters are often an EMA of student parameters:

\[
\theta_{\rm teacher}
\leftarrow
m\theta_{\rm teacher}
+
(1-m)\theta_{\rm student}.
\]

Train student output distribution to match teacher targets.

Main challenge: avoid representation collapse.

---

### 9. CLIP as a Vision Foundation Model Bridge

Image encoder:

\[
z_i=f_I(x_i).
\]

Text encoder:

\[
t_i=f_T(c_i).
\]

Normalize embeddings and compute similarity:

\[
s_{ij}
=
\frac{z_i^\top t_j}{\tau}.
\]

For image-to-text classification over a batch:

\[
\mathcal L_{I\to T}
=
-\frac1B
\sum_i
\log
\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}.
\]

Symmetric text-to-image loss is added.

CLIP connects ViT-style encoders to multimodal foundation modeling.

---

### 10. SAM-Style Promptable Vision

A promptable segmentation model can be decomposed into:
- image encoder;
- prompt encoder;
- mask decoder.

The key foundation-model idea is not merely segmentation accuracy. It is training one broad model to respond to diverse prompts and transfer across downstream settings.

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Patch embedding can be implemented as a strided convolution.
- When changing input resolution, pay attention to positional embeddings and token count.
- Distinguish supervised ViT training from self-supervised pretraining such as MAE/DINO.

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
