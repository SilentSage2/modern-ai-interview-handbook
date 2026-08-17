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


```math
\theta_0
=
\arg\min_\theta
\mathcal L_{\rm pretrain}.
```


Adapt:


```math
\theta^*
=
\arg\min_\theta
\mathcal L_{\rm downstream}
\quad
\text{initialized at }\theta_0.
```


The key assumption is that pretraining learned reusable representations/behaviors.

---

### 2. Full Fine-Tuning

Update all parameters:


```math
\theta=\theta_0+\Delta\theta.
```


Pros:
- maximum adaptation capacity.

Cons:
- gradient + optimizer memory for all parameters;
- one full model copy per task;
- can overfit small datasets;
- can cause catastrophic forgetting.

---

### 3. Linear Probe

Freeze backbone $f_{\theta_0}$:


```math
z=f_{\theta_0}(x),
```


train head


```math
\hat y=Wz+b.
```


This evaluates representation quality with minimal adaptation.

---

### 4. LoRA

For pretrained linear layer


```math
y=Wx,
\qquad
W\in\mathbb R^{d_{\rm out}\times d_{\rm in}},
```


freeze $W$.

Parameterize update:


```math
\Delta W
=
\frac{\alpha}{r}BA,
```


where


```math
A\in\mathbb R^{r\times d_{\rm in}},
\qquad
B\in\mathbb R^{d_{\rm out}\times r},
\qquad
r\ll \min(d_{\rm in},d_{\rm out}).
```


Then


```math
y
=
Wx
+
\frac{\alpha}{r}BAx.
```


Trainable parameter count:


```math
r(d_{\rm in}+d_{\rm out})
```


instead of


```math
d_{\rm in}d_{\rm out}.
```


---

### 5. Why Low Rank?

Suppose downstream adaptation only needs movement in a low-dimensional subspace of the full parameter space.

Then a full matrix $\Delta W$ is unnecessarily expressive.

Low rank imposes


```math
\mathrm{rank}(\Delta W)\le r.
```


This is a structural prior on the update.

---

### 6. LoRA Initialization

A common design:
- initialize $A$ randomly;
- initialize $B=0$.

Then initially:


```math
\Delta W=0,
```


so the model starts exactly from the pretrained function.

---

### 7. Merging LoRA

At inference, one may form


```math
W_{\rm merged}
=
W+\frac{\alpha}{r}BA.
```


Then no separate adapter branch is needed.

This can reduce inference overhead when the adapter is fixed.

---

### 8. Choosing Target Modules

Common LLM targets:
- $W_Q$;
- $W_K$;
- $W_V$;
- $W_O$;
- MLP projection matrices.

Tradeoff:
- more target modules → more adaptation capacity and parameters;
- fewer modules → cheaper and often sufficient.

---

### 9. Rank $r$

Higher $r$:
- more trainable capacity;
- more memory/compute.

Lower $r$:
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


```math
W_q\approx W
```


is stored in low precision, while trainable $A,B$ remain in a training-friendly precision.

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


```math
\theta\to\theta'.
```


RAG changes context:


```math
x\to[x;d_1;\ldots;d_k].
```


A useful interview rule:
- use fine-tuning to change behavior/style/task competence;
- use RAG to inject mutable/external knowledge.

<!-- DEEP_DIVE_START -->
## Deep dive I: what exactly changes during fine-tuning?

Let pretrained parameters be $\theta_0$.

Full fine-tuning solves:


```math
\theta^*
=
\theta_0+\Delta\theta.
```


PEFT constrains the allowed update:


```math
\Delta\theta\in\mathcal S,
```


where $\mathcal S$ is a much smaller structured parameter space.

LoRA chooses a low-rank structure for selected matrices.

This is a useful unifying view:

> PEFT is not “less training”; it is training under a constrained adaptation parameterization.

---

## Deep dive II: LoRA parameter savings worked example

Suppose attention projection:


```math
W\in\mathbb R^{4096\times4096}.
```


Full matrix parameters:


```math
4096^2
=
16{,}777{,}216.
```


LoRA with $r=16$:


```math
A:16\times4096,
\qquad
B:4096\times16.
```


Trainable parameters:


```math
16(4096+4096)
=
131{,}072.
```


Ratio:


```math
\frac{131072}{16777216}
\approx0.78\%.
```


For one projection, LoRA trains under 1% as many matrix parameters.

---

## Deep dive III: why LoRA saves more than checkpoint size

During full Adam training, trainable parameters may need:
- gradient;
- first moment;
- second moment;
- sometimes master precision copy.

Freezing the base removes much of this training-state cost.

Therefore trainable-parameter reduction affects:
- gradient memory;
- optimizer memory;
- communication;
- checkpoint size.

Activation memory, however, does **not** disappear entirely, because activations are still needed to compute gradients through adapter paths.

---

## Deep dive IV: where to place LoRA

For Transformer attention:


```math
Q=XW_Q,\quad
K=XW_K,\quad
V=XW_V,\quad
O=HW_O.
```


Possible targets:
- $W_Q$;
- $W_K$;
- $W_V$;
- $W_O$;
- MLP projections.

If task mostly needs routing changes, attention projections may be sufficient. Broader domain shifts may benefit from MLP adaptation too.

There is no universal best target list; treat it as a capacity/design choice.

---

## Deep dive V: rank and alpha

LoRA update:


```math
\Delta W=\frac{\alpha}{r}BA.
```


Rank $r$ controls maximum rank of update.

$\alpha/r$ controls update scale.

If $r$ increases without scaling adjustment, update magnitude can change. This is why rank and scaling are coupled hyperparameters rather than independent cosmetic settings.

---

## Deep dive VI: QLoRA mental model

Base model:


```math
W\rightarrow Q(W)
```


stored at low bit width and frozen.

Forward path approximately uses dequantized compute representation:


```math
xQ(W)
```


plus high-precision LoRA update:


```math
x\Delta W.
```


Only adapter parameters receive optimizer states.

QLoRA therefore attacks a different memory source than LoRA alone:

- LoRA reduces **trainable state**;
- quantization reduces **frozen base-weight storage**.

---

## Deep dive VII: full FT vs LoRA experiment design

A convincing experiment should compare:

| Setting | Trainable params | Peak memory | Train time | Accuracy |
|---|---:|---:|---:|---:|
| Linear probe | low | low | low | baseline |
| LoRA | low-medium | medium | medium | ? |
| Full FT | high | high | high | ? |

Control:
- same dataset split;
- comparable augmentation;
- same evaluation metric;
- tuned learning rates per regime.

Do **not** compare one carefully tuned LoRA run against an untuned full-FT baseline and conclude LoRA is universally better.

---

## Deep dive VIII: SFT data formatting

Common instruction template:

```text
System: You are ...
User: ...
Assistant: ...
```

Important engineering choices:
- where BOS/EOS tokens appear;
- whether multiple turns are packed;
- which roles contribute to loss;
- maximum length;
- truncation side;
- masking;
- special token consistency.

Bad formatting can cause more damage than small optimizer changes.

---

## Minimal PEFT-style pseudocode

```python
base = AutoModelForCausalLM.from_pretrained(model_id)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(base, config)
model.print_trainable_parameters()
```

During an interview, do not stop at API syntax. Explain what layer is being replaced conceptually:


```math
Wx
\rightarrow
Wx+\frac{\alpha}{r}BAx.
```


---

## Fine-tuning failure modes

### Overfitting
Small task data causes memorization.

### Catastrophic forgetting
General behaviors degrade.

### Format overfitting
Model learns response style but not task competence.

### Distribution mismatch
Training examples are too clean/narrow compared with deployment.

### Evaluation leakage
Benchmark examples enter fine-tuning data.

### Adapter interference
Multiple adapters or merged modifications may conflict.

---

## Practical decision tree

Use **prompting** when the base model already knows the task.

Use **RAG** when knowledge is external/fresh.

Use **LoRA/PEFT** when behavior or domain mapping needs adaptation under limited compute.

Use **full fine-tuning** when strong adaptation is required and data/compute justify it.

These methods are complementary rather than mutually exclusive.
<!-- DEEP_DIVE_END -->

<!-- SECOND_DEEP_DIVE_START -->
## Fine-tuning as optimization around a pretrained solution

Instead of optimizing from random initialization, fine-tuning starts near a useful solution $\theta_0$.

Locally approximate downstream loss:


```math
L(\theta_0+\Delta\theta)
\approx
L(\theta_0)
+
g^\top\Delta\theta
+
\frac12\Delta\theta^\top H\Delta\theta.
```


If useful updates mostly lie in a low-dimensional subspace of parameter space, constraining $\Delta\theta$ can preserve much of full fine-tuning capacity.

This provides intuition for why low-rank/adapter methods can work even though the base network has billions of parameters.

---

## Adapter layers vs LoRA

### Adapter
Insert bottleneck network:


```math
h'
=
h+W_{\rm up}\sigma(W_{\rm down}h).
```


This creates an explicit extra computational branch.

### LoRA
Modify existing linear transform:


```math
Wh
\rightarrow
( W+BA )h.
```


Because $BA$ can be merged into $W$, LoRA can avoid persistent extra branch cost for a fixed merged adapter.

---

## Prompt tuning

Instead of changing internal matrices, learn virtual prompt embeddings:


```math
P\in\mathbb R^{m\times d}.
```


Input becomes:


```math
[P;x_1,\ldots,x_T].
```


Only $P$ is trainable.

Parameter count can be tiny, but adaptation capacity is also more constrained.

This illustrates that PEFT is a family of **where and how adaptation parameters are injected**.

---

## Fine-tuning learning rates

Pretrained weights contain useful information. Fine-tuning generally uses smaller learning rates than training from scratch.

Why?
A large update can destroy pretrained structure.

Some approaches use discriminative learning rates:
- lower LR in early/pretrained layers;
- higher LR in new head/adapters.

---

## Layer freezing strategy

Suppose a vision model has blocks $1,\ldots,L$.

Options:

```text
freeze all → linear probe
train last k blocks → partial FT
train all → full FT
```

A practical experiment can gradually increase $k$. This maps the performance–compute curve and reveals how much representation adaptation is needed.

---

## Why LoRA can be applied to vision and diffusion

LoRA only requires linear-like weight matrices.

Transformer attention in:
- LLM;
- ViT;
- diffusion Transformer;
- VLM

all contains projections


```math
W_Q,W_K,W_V,W_O.
```


Therefore the same low-rank adaptation principle is modality-agnostic.

---

## LoRA gradient flow

For:


```math
y=Wx+BAx,
```


$W$ frozen.

Gradient for $B$:


```math
\frac{\partial L}{\partial B}
=
\frac{\partial L}{\partial y}
(Ax)^\top.
```


Gradient for $A$:


```math
\frac{\partial L}{\partial A}
=
B^\top
\frac{\partial L}{\partial y}
x^\top.
```


If $B=0$ at initialization, initial gradient into $A$ is zero while $B$ begins learning; after $B$ moves, $A$ receives gradients. This is consistent with no-op initialization.

---

## Memory experiment design

Measure with:

```python
torch.cuda.reset_peak_memory_stats()
# training step(s)
peak = torch.cuda.max_memory_allocated()
```

Report:
- model dtype;
- optimizer;
- gradient checkpointing;
- batch/sequence size;
- rank;
- target modules.

Otherwise memory comparisons are not reproducible.

---

## Adapter checkpoint contents

A LoRA checkpoint should not need all frozen base weights.

Conceptually store:
- adapter configuration;
- $A,B$ weights;
- metadata indicating base model.

This makes per-task storage much smaller than copying the full model.

---

## Multi-task adaptation

One base model can support:

```text
base
├→ adapter medical
├→ adapter code
├→ adapter finance
└→ adapter customer support
```

At inference, select/load corresponding adapter.

This is operationally attractive but raises:
- adapter routing;
- hot-swapping;
- cache compatibility;
- version management.

---

## Fine-tuning interview project story

A strong story has exact choices:

> I compared a frozen linear probe, full fine-tuning, and LoRA on the same downstream dataset. For LoRA I targeted Q/V projections at rank 16, measured trainable parameters and peak GPU memory, and evaluated both target performance and out-of-domain regression. LoRA recovered most/full downstream performance with much lower trainable-state memory.

That is much stronger than:

> I know LoRA.
<!-- SECOND_DEEP_DIVE_END -->

<!-- THIRD_DEEP_DIVE_START -->
## Fine-tuning data quantity vs capacity

Adaptation method should match available data.

Very small dataset:
- full fine-tuning may overfit;
- linear probe or LoRA can act as stronger regularization.

Large domain dataset:
- full fine-tuning may exploit more capacity.

Therefore PEFT is not only a memory trick. It can also change the statistical bias of adaptation.

---

## Continual adaptation

Suppose a model is adapted sequentially:


```math
\theta_0
\rightarrow
\theta_A
\rightarrow
\theta_{A+B}.
```


Training on B may degrade A.

Adapters offer another strategy:


```math
\theta_0 + \Delta_A,
\qquad
\theta_0 + \Delta_B.
```


Tasks remain separated at parameter level.

This can simplify continual/multi-domain deployment.

---

## Adapter composition

If multiple learned updates exist:


```math
W'
=
W+\Delta W_A+\Delta W_B,
```


composition may or may not behave well because adaptations were trained independently.

Weighted merging:


```math
W'
=
W+\lambda_A\Delta W_A+\lambda_B\Delta W_B
```


is possible conceptually, but performance must be evaluated rather than assumed.

---

## What “fine-tuning experience” should mean on a resume

A credible project should let you answer:

- Which pretrained checkpoint?
- What dataset size?
- What data format?
- Which parameters were trainable?
- Which optimizer/LR?
- What precision?
- What GPU memory?
- How did you evaluate?
- What baseline?
- What failed?
- How did you choose LoRA rank/targets?

If you cannot answer these, “experience in fine-tuning” may sound superficial.

---

## Debugging a fine-tuning run

### Loss does not decrease
Check:
- labels shifted/masked correctly?
- adapters actually trainable?
- optimizer sees adapter parameters?
- learning rate too small?
- quantized model prepared correctly?

### Training loss decreases, validation worsens
Likely overfitting or domain mismatch.

### Output format broken
Check chat template and special tokens.

### GPU memory unexpectedly high
Check:
- full parameters accidentally trainable;
- optimizer states;
- sequence length;
- activation checkpointing;
- dtype;
- batch size.

This debugging knowledge is often more valuable than memorizing PEFT method names.
<!-- THIRD_DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Always report trainable parameter count and GPU memory, not only downstream accuracy.
- For LoRA experiments, record target modules, rank, alpha, dropout, learning rate, and whether adapters are merged.
- Compare against a frozen-backbone baseline and, where feasible, full fine-tuning.

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
