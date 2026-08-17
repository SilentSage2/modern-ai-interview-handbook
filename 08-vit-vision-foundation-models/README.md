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


```math
x\in\mathbb R^{H\times W\times C}.
```


Patch size $P\times P$.

Number of patches:


```math
N=\frac{HW}{P^2}.
```


Each patch has


```math
P^2C
```


values.

Flatten patch $x_p^{(i)}$ and project:


```math
z_i=x_p^{(i)}E,
\qquad
E\in\mathbb R^{(P^2C)\times d}.
```


Now the image becomes a token sequence:


```math
Z\in\mathbb R^{N\times d}.
```


---

### 2. Patch Embedding as Convolution

Patchification + linear projection can be implemented with Conv2D:

- kernel size $P$;
- stride $P$;
- output channels $d$.

Why?

Each convolutional kernel sees exactly one non-overlapping patch and produces its embedding.

This is a useful implementation equivalence.

---

### 3. CLS Token

Add learnable


```math
z_{\rm cls}\in\mathbb R^d
```


to the token sequence.

After all Transformer blocks, use its final representation:


```math
h_{\rm cls}
```


for classification.

Alternative modern designs may use global average pooling instead.

---

### 4. Positional Embeddings

Patch tokens need location information.

Input:


```math
Z_0=
[z_{\rm cls};z_1;\ldots;z_N]
+
E_{\rm pos}.
```


If fine-tuning at a different resolution, positional embeddings may need interpolation.

---

### 5. Complexity and Patch Size

Number of tokens:


```math
N=\frac{HW}{P^2}.
```


Attention cost:


```math
O(N^2d).
```


Therefore halving patch size approximately quadruples token count and can increase attention cost by about $16\times$.

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


```math
\mathcal L
=
\frac1{|\mathcal M|}
\sum_{i\in\mathcal M}
\|x_i-\hat x_i\|^2.
```


Why high masking ratio can work:
natural images are highly redundant, so reconstruction forces learning broader structure rather than copying local texture.

---

### 8. DINO-Style Self-Distillation

Student and teacher receive different image views.

Teacher parameters are often an EMA of student parameters:


```math
\theta_{\rm teacher}
\leftarrow
m\theta_{\rm teacher}
+
(1-m)\theta_{\rm student}.
```


Train student output distribution to match teacher targets.

Main challenge: avoid representation collapse.

---

### 9. CLIP as a Vision Foundation Model Bridge

Image encoder:


```math
z_i=f_I(x_i).
```


Text encoder:


```math
t_i=f_T(c_i).
```


Normalize embeddings and compute similarity:


```math
s_{ij}
=
\frac{z_i^\top t_j}{\tau}.
```


For image-to-text classification over a batch:


```math
\mathcal L_{I\to T}
=
-\frac1B
\sum_i
\log
\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}.
```


Symmetric text-to-image loss is added.

CLIP connects ViT-style encoders to multimodal foundation modeling.

---

### 10. SAM-Style Promptable Vision

A promptable segmentation model can be decomposed into:
- image encoder;
- prompt encoder;
- mask decoder.

The key foundation-model idea is not merely segmentation accuracy. It is training one broad model to respond to diverse prompts and transfer across downstream settings.

<!-- DEEP_DIVE_START -->
## Deep dive I: ViT as a tokenization choice

A standard ViT does not treat an image as intrinsically 2D after patch embedding. It first converts the image to a token sequence:


```math
I
\rightarrow
(x_1,\ldots,x_N).
```


The spatial prior is therefore mostly supplied by:
- how patches are formed;
- positional encoding;
- pretraining data/objective.

This is the conceptual reason ViT connects naturally to LLM/VLM architectures.

### Tensor example

For


```math
I\in\mathbb R^{B\times3\times224\times224},
```


patch size $P=16$:


```math
N=(224/16)^2=196.
```


If embedding dimension $d=768$:


```math
[B,3,224,224]
\rightarrow
[B,196,768].
```


With CLS token:


```math
[B,197,768].
```


---

## Deep dive II: patch embedding implementation

```python
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, d_model=768, patch=16):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, d_model,
            kernel_size=patch,
            stride=patch
        )

    def forward(self, x):
        x = self.proj(x)          # [B,D,H/P,W/P]
        x = x.flatten(2)          # [B,D,N]
        x = x.transpose(1, 2)     # [B,N,D]
        return x
```

Why is this equivalent to patch flattening + linear layer?
Each kernel covers exactly one $P\times P\times C$ patch and outputs a $d$-dimensional vector.

---

## Deep dive III: patch-size tradeoff numerically

224×224 image:

### $P=16$


```math
N=196.
```


### $P=8$


```math
N=784.
```


Token count increases $4\times$. Dense attention interactions increase:


```math
196^2\rightarrow784^2,
```


which is $16\times$ more score pairs.

Smaller patches preserve finer detail but rapidly increase compute.

---

## Deep dive IV: why ViT benefits from large-scale pretraining

CNNs encode strong assumptions:
- neighboring pixels matter;
- the same local detector should work anywhere.

ViT has weaker built-in locality. With limited data this can hurt sample efficiency. With large-scale pretraining, weaker priors can become an advantage because the model can learn broader interaction patterns from data.

A strong interview answer should avoid saying “ViT is better because attention is global.” It should discuss:
- inductive bias;
- data scale;
- pretraining;
- resolution;
- compute.

---

## Deep dive V: MAE architecture

Let patch set be


```math
\{x_1,\ldots,x_N\}.
```


Randomly choose visible subset $V$ and masked subset $M$.

Encoder sees only:


```math
\{x_i:i\in V\}.
```


This matters computationally. If $75\%$ are masked, encoder processes only $25\%$ of tokens, greatly reducing expensive self-attention.

Decoder receives encoded visible tokens plus mask tokens and reconstructs masked patches.

The asymmetry


```math
\text{heavy encoder} + \text{light decoder}
```


is deliberate: representation quality is assigned to the encoder.

---

## Deep dive VI: DINO intuition

DINO-family learning can be understood as consistency across transformed views without explicit class labels.

Teacher output:


```math
p_t(y|x_{\rm global}),
```


student output:


```math
p_s(y|x_{\rm local}).
```


Train student to match teacher.

The model must map local and global views of the same image into consistent semantic structure.

A subtle point: collapse prevention is essential. If both output the same constant distribution for all images, matching loss alone would be trivial.

---

## Deep dive VII: linear probe vs fine-tuning

Suppose a pretrained encoder $f_{\theta_0}$.

### Linear probe


```math
z=f_{\theta_0}(x),\qquad y=Wz.
```


Tests whether representation already makes task linearly accessible.

### Fine-tuning


```math
\theta_0\rightarrow\theta^*.
```


Allows representation itself to move.

If linear probe is strong but full fine-tuning only slightly improves, pretraining already matches the task well.

---

## Deep dive VIII: vision foundation model perspective

A model becomes foundation-like when one pretrained representation supports many downstream behaviors.

Examples of transfer mechanisms:
- classifier head;
- detector/segmenter decoder;
- prompts;
- adapters/LoRA;
- image-text alignment.

The key shift is:


```math
\text{train one model for one task}
\rightarrow
\text{pretrain broad representation and adapt}.
```


---

## Worked implementation exercise: tiny ViT

Architecture:

```text
image
→ patch embed
→ add position
→ TransformerEncoder × L
→ CLS / mean pooling
→ classifier
```

Minimal forward:

```python
x = patch_embed(img)                     # [B,N,D]
cls = cls_token.expand(x.size(0), -1, -1)
x = torch.cat([cls, x], dim=1)           # [B,N+1,D]
x = x + pos_embed[:, :x.size(1)]
x = blocks(x)
logits = head(norm(x[:, 0]))
```

Questions to answer while implementing:
- Why is CLS learnable?
- Why does position shape depend on resolution?
- What happens if the number of patches changes?
- Where would LoRA be inserted?
<!-- DEEP_DIVE_END -->

<!-- SECOND_DEEP_DIVE_START -->
## Architecture case study: ViT-B/16 style reasoning

The notation “B/16” commonly communicates:
- base-size Transformer;
- $16\times16$ image patches.

You should not memorize one exact configuration, but know how to derive compute from:
- image size;
- patch size;
- hidden dimension;
- number of layers;
- attention heads.

For 224×224 images and 16×16 patches:


```math
N=14^2=196.
```


With CLS token:


```math
T=197.
```


If image resolution rises to 448×448:


```math
N=28^2=784.
```


Dense score pairs scale:


```math
197^2\approx3.9\times10^4
```


to


```math
785^2\approx6.2\times10^5,
```


roughly $16\times$.

This is why high-resolution vision requires careful token/attention design.

---

## Why hierarchical vision models exist

Dense prediction tasks need features at multiple scales.

CNN naturally creates:

```text
high resolution / low semantic depth
→ medium resolution
→ low resolution / high semantic depth
```

Vanilla ViT keeps one token scale.

Hierarchical Transformers introduce:
- patch merging;
- window attention;
- pyramidal feature maps.

This makes integration with detection/segmentation heads easier.

---

## Swin-style window reasoning

If global attention cost is


```math
O(N^2),
```


partition tokens into windows containing $M$ tokens.

If there are $N/M$ windows, cost becomes approximately:


```math
\frac{N}{M}M^2
=
NM.
```


For fixed window size, complexity becomes linear in total token count.

Shifted windows allow information to cross previous window boundaries across layers.

---

## Positional embedding interpolation

Pretrained position table for 14×14 patches:


```math
E_{\rm pos}\in\mathbb R^{196\times d}.
```


Fine-tuning at 28×28 patches requires 784 positions.

One common strategy:
1. reshape patch position vectors to 2D grid;
2. interpolate grid spatially;
3. flatten back.

This works because spatial positions have a grid structure even though Transformer consumes a sequence.

---

## Vision self-supervision comparison

### MAE
Predict missing pixels/patches.

Pressure:
learn enough image structure to reconstruct hidden content.

### DINO-like
Match semantic representations across views.

Pressure:
learn view-invariant representation.

### CLIP
Align images with natural language.

Pressure:
learn concepts that correspond to textual semantics.

They can produce different downstream strengths because **pretraining target defines what information representation emphasizes**.

---

## Foundation-model adaptation for vision

A pretrained ViT can be adapted in multiple ways.

### Classification
Replace head.

### Segmentation
Use intermediate patch tokens / decoder.

### Retrieval
Project pooled representation to embedding.

### VLM
Pass patch features through projector/resampler into LLM.

### Domain-specific imaging
- linear probe;
- full FT;
- LoRA;
- self-supervised domain adaptation.

The backbone can therefore be the same while task interface changes.

---

## Worked problem: 3D ViT

Volume:


```math
X\in\mathbb R^{B\times C\times H\times W\times D}.
```


Use patch:


```math
P_H\times P_W\times P_D.
```


Token count:


```math
N
=
\frac{H}{P_H}
\frac{W}{P_W}
\frac{D}{P_D}.
```


3D token count can explode quickly.

For 192×192×128 volume with 16×16×16 patch:


```math
N=12\times12\times8=1152.
```


Dense attention already uses:


```math
1152^2\approx1.33\text{ million}
```


token pairs per head/layer/sample.

This shows why 3D medical ViTs often need:
- larger patches;
- windows;
- axial/factorized attention;
- hierarchical designs;
- patch/slice sampling.

---

## Implementation exercise: fine-tune pretrained ViT

Experiment:

### A. Linear probe
Freeze encoder.

### B. Last-block fine-tuning
Unfreeze final Transformer block + head.

### C. Full fine-tuning
Update all layers.

### D. LoRA
Adapt selected attention projections.

Record:
- trainable parameters;
- peak GPU memory;
- epoch time;
- validation score;
- robustness under domain shift.

The result should teach *when* representation adaptation matters rather than just which method wins.

---

## Common vision-foundation-model pitfalls

### Resolution mismatch
Pretraining resolution differs from downstream resolution.

### Domain shift
Natural-image features may not optimally represent medical/scientific texture.

### Shortcut learning
Model learns acquisition/site artifacts instead of semantic target.

### Token cost
High-resolution or 3D inputs make attention expensive.

### False “foundation model” claim
A model trained for one narrow task is not automatically a foundation model because it uses a Transformer.
<!-- SECOND_DEEP_DIVE_END -->

<!-- THIRD_DEEP_DIVE_START -->
## Design comparison: ViT, CNN, and hybrid models

A useful way to compare them is through the **interaction operator**.

### CNN


```math
y_i
=
\sum_{\Delta\in\mathcal N}
W_\Delta x_{i+\Delta}.
```


The neighborhood $\mathcal N$ is fixed and local.

### ViT


```math
y_i
=
\sum_j
A_{ij}(X)Vx_j.
```


The interaction neighborhood can be global and the weights $A_{ij}$ are input dependent.

### Hybrid
Convolution may first create local features/tokens, then attention models long-range interaction.

Hybrid design is attractive when:
- local spatial prior is known to be useful;
- data are limited;
- global correlation still matters.

The interview takeaway is not “one is best.” It is that architecture encodes assumptions about which interactions should be easy to learn.

---

## Why intermediate ViT tokens matter for dense prediction

Classification only needs one global vector. Segmentation/detection need spatial outputs.

Patch-token grid can be reshaped:


```math
[B,N,d]
\rightarrow
[B,H/P,W/P,d].
```


Intermediate blocks represent multiple semantic depths.

A dense decoder can combine features from several Transformer depths, analogous to feature pyramids in CNNs.

Thus ViT is not restricted to using only CLS token.

---

## Fine-tuning with domain shift

Consider natural-image pretraining and MRI downstream data.

The pretrained model may still transfer because early/mid-level representations contain generic structure, but:
- input channel count differs;
- intensity statistics differ;
- textures/semantics differ;
- 2D vs 3D structure differs.

Possible adaptation levels:

1. normalize/re-map input only;
2. train new patch embedding;
3. linear probe;
4. fine-tune upper layers;
5. LoRA all attention blocks;
6. domain-specific self-supervised pretraining.

This progression is a useful experimental plan.

---

## Vision evaluation beyond top-1 accuracy

For representation/foundation models, useful evaluation includes:
- linear probe;
- few-shot classification;
- full fine-tuning;
- retrieval;
- segmentation transfer;
- robustness to corruptions/domain shift;
- calibration.

A “good representation” is one that transfers across tasks, not only one that wins one supervised benchmark.
<!-- THIRD_DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Patch embedding can be implemented as a strided convolution.
- When changing input resolution, pay attention to positional embeddings and token count.
- Distinguish supervised ViT training from self-supervised pretraining such as MAE/DINO.

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
