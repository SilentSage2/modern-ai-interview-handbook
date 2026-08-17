# VLM & Multimodal Foundation Models

**Status:** Priority + Hands-on

## Why this matters

VLMs connect visual representation learning with language reasoning and are increasingly common in multimodal AI roles.

## Learning objectives

- Derive CLIP-style contrastive alignment.
- Understand projector and cross-attention interfaces between vision encoders and LLMs.
- Explain multimodal instruction tuning, freezing strategies, and hallucination.

## Chapter map

- Dual encoders and contrastive image-text alignment
- CLIP loss and retrieval
- Vision encoder + projector + LLM architecture
- Cross-attention and multimodal fusion
- Visual tokens and language tokens
- Freezing vs tuning vision encoder/projector/LLM
- Multimodal instruction tuning
- Image-text grounding and hallucination
- VLM evaluation: VQA, retrieval, grounding, robustness
- Medical/scientific VLM adaptation


---

## Core concepts and theory

### 1. Why Multimodal Alignment?

Vision encoder and language model initially live in different representation spaces.

We need a mechanism that maps visual information into a representation compatible with language reasoning/generation.

---

### 2. CLIP Dual Encoder

Image embedding:

\[
z_i=
\frac{f_I(x_i)}{\|f_I(x_i)\|}.
\]

Text embedding:

\[
t_i=
\frac{f_T(c_i)}{\|f_T(c_i)\|}.
\]

Similarity:

\[
s_{ij}
=
\frac{z_i^\top t_j}{\tau}.
\]

Image-to-text loss:

\[
\mathcal L_{I\to T}
=
-\frac1B\sum_i
\log
\frac{e^{s_{ii}}}
{\sum_j e^{s_{ij}}}.
\]

Text-to-image:

\[
\mathcal L_{T\to I}
=
-\frac1B\sum_i
\log
\frac{e^{s_{ii}}}
{\sum_j e^{s_{ji}}}.
\]

Total:

\[
\mathcal L
=
\frac12(
\mathcal L_{I\to T}
+
\mathcal L_{T\to I}
).
\]

---

### 3. Why Cosine Similarity?

L2 normalization removes embedding magnitude:

\[
z^\top t=\cos\theta.
\]

The model is trained primarily on angular alignment rather than arbitrary vector norms.

---

### 4. Vision Encoder + Projector + LLM

Visual encoder:

\[
H_v=f_v(I)
\in\mathbb R^{N_v\times d_v}.
\]

Project:

\[
\tilde H_v=P(H_v)
\in\mathbb R^{N_v\times d_{\rm LLM}}.
\]

Then combine with text token embeddings \(H_t\):

\[
[\tilde H_v;H_t].
\]

The LLM can now attend over visual and textual tokens.

---

### 5. Why Use a Projector?

The vision representation and LLM hidden space differ in:
- dimension;
- statistical scale;
- semantic organization.

A projector learns the interface.

Simple projector:

\[
P(h)=Wh+b.
\]

More expressive versions:
- MLP;
- resampler;
- cross-attention module;
- query transformer.

---

### 6. Cross-Attention Fusion

Let text hidden states be queries:

\[
Q=H_tW_Q.
\]

Visual states provide keys and values:

\[
K=H_vW_K,\qquad
V=H_vW_V.
\]

Then

\[
H_{\rm fused}
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)V.
\]

This lets language tokens selectively retrieve visual information.

---

### 7. Freeze or Fine-Tune?

Possible stages:

#### Stage 1
Freeze vision encoder and LLM, train projector.

Purpose:
align modalities cheaply.

#### Stage 2
Unfreeze selected modules or apply LoRA.

Purpose:
learn deeper multimodal reasoning/instruction behavior.

Tradeoff:
more adaptation vs more compute and forgetting risk.

---

### 8. Multimodal Instruction Tuning

Training example:

\[
(I,x)\to y.
\]

Optimize autoregressive response likelihood:

\[
\mathcal L
=
-\sum_{t\in y}
\log
p_\theta(y_t|I,x,y_{<t}).
\]

This teaches the model to condition language generation on images.

---

### 9. Hallucination

A VLM hallucinates when generated text is not grounded in the visual input.

Potential causes:
- language prior dominates weak visual signal;
- projector discards detail;
- insufficient grounding supervision;
- ambiguous image;
- decoding bias.

Evaluation should include grounding-sensitive tests, not only fluency.

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Track where visual information enters the language model.
- Distinguish dual-encoder alignment (CLIP-style) from generative VLM fusion.
- For fine-tuning, state exactly which components are frozen, partially tuned, or adapted with LoRA.

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
