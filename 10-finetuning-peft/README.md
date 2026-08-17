# Fine-Tuning, LoRA & PEFT

**Status:** Priority + Hands-on

## Why this matters

Many job descriptions explicitly ask for experience adapting pretrained models; this chapter turns that phrase into concrete methods and tradeoffs.

## Learning objectives

- Compare linear probing, partial/full fine-tuning, and PEFT.
- Derive LoRA and understand rank, alpha, target modules, and merging.
- Explain QLoRA, catastrophic forgetting, and fine-tuning vs RAG.

## Chapter map

- Linear probing vs partial vs full fine-tuning
- Catastrophic forgetting and domain adaptation
- Adapters
- LoRA: low-rank updates ΔW = BA
- QLoRA and low-bit base models
- PEFT tradeoffs: memory, throughput, storage, quality
- Target-module selection
- Rank, alpha and dropout choices
- Instruction tuning datasets and response-only loss masking
- Evaluation of adaptation vs base model


---

## Core concepts and theory

### 1. Transfer Learning

Pretrain:

\[
\theta_0
=
\arg\min_\theta
\mathcal L_{\rm pretrain}.
\]

Adapt:

\[
\theta^*
=
\arg\min_\theta
\mathcal L_{\rm downstream}
\quad
\text{initialized at }\theta_0.
\]

The key assumption is that pretraining learned reusable representations/behaviors.

---

### 2. Full Fine-Tuning

Update all parameters:

\[
\theta=\theta_0+\Delta\theta.
\]

Pros:
- maximum adaptation capacity.

Cons:
- gradient + optimizer memory for all parameters;
- one full model copy per task;
- can overfit small datasets;
- can cause catastrophic forgetting.

---

### 3. Linear Probe

Freeze backbone \(f_{\theta_0}\):

\[
z=f_{\theta_0}(x),
\]

train head

\[
\hat y=Wz+b.
\]

This evaluates representation quality with minimal adaptation.

---

### 4. LoRA

For pretrained linear layer

\[
y=Wx,
\qquad
W\in\mathbb R^{d_{\rm out}\times d_{\rm in}},
\]

freeze \(W\).

Parameterize update:

\[
\Delta W
=
\frac{\alpha}{r}BA,
\]

where

\[
A\in\mathbb R^{r\times d_{\rm in}},
\qquad
B\in\mathbb R^{d_{\rm out}\times r},
\qquad
r\ll \min(d_{\rm in},d_{\rm out}).
\]

Then

\[
y
=
Wx
+
\frac{\alpha}{r}BAx.
\]

Trainable parameter count:

\[
r(d_{\rm in}+d_{\rm out})
\]

instead of

\[
d_{\rm in}d_{\rm out}.
\]

---

### 5. Why Low Rank?

Suppose downstream adaptation only needs movement in a low-dimensional subspace of the full parameter space.

Then a full matrix \(\Delta W\) is unnecessarily expressive.

Low rank imposes

\[
\mathrm{rank}(\Delta W)\le r.
\]

This is a structural prior on the update.

---

### 6. LoRA Initialization

A common design:
- initialize \(A\) randomly;
- initialize \(B=0\).

Then initially:

\[
\Delta W=0,
\]

so the model starts exactly from the pretrained function.

---

### 7. Merging LoRA

At inference, one may form

\[
W_{\rm merged}
=
W+\frac{\alpha}{r}BA.
\]

Then no separate adapter branch is needed.

This can reduce inference overhead when the adapter is fixed.

---

### 8. Choosing Target Modules

Common LLM targets:
- \(W_Q\);
- \(W_K\);
- \(W_V\);
- \(W_O\);
- MLP projection matrices.

Tradeoff:
- more target modules → more adaptation capacity and parameters;
- fewer modules → cheaper and often sufficient.

---

### 9. Rank \(r\)

Higher \(r\):
- more trainable capacity;
- more memory/compute.

Lower \(r\):
- stronger regularization;
- cheaper;
- may underfit difficult domain shifts.

Treat rank as a model-capacity hyperparameter.

---

### 10. QLoRA

QLoRA combines:
- low-bit quantized frozen base weights;
- higher-precision LoRA adapter training.

Conceptually:

\[
W_q\approx W
\]

is stored in low precision, while trainable \(A,B\) remain in a training-friendly precision.

Main benefit:
dramatically lower memory while preserving useful adaptation quality.

---

### 11. Catastrophic Forgetting

Full fine-tuning on narrow data can move parameters away from broad pretrained behavior.

Mitigation:
- small learning rates;
- PEFT;
- data mixing/replay;
- regularization to reference model;
- freeze lower layers;
- evaluate both target and general capabilities.

---

### 12. Fine-Tuning vs RAG

Fine-tuning changes parameters:

\[
\theta\to\theta'.
\]

RAG changes context:

\[
x\to[x;d_1;\ldots;d_k].
\]

A useful interview rule:
- use fine-tuning to change behavior/style/task competence;
- use RAG to inject mutable/external knowledge.

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Always report trainable parameter count and GPU memory, not only downstream accuracy.
- For LoRA experiments, record target modules, rank, alpha, dropout, learning rate, and whether adapters are merged.
- Compare against a frozen-backbone baseline and, where feasible, full fine-tuning.

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
