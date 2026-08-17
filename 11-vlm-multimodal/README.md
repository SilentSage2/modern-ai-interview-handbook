# VLM & Multimodal Foundation Models

**Status:** Priority + Hands-on

## Why this matters

VLMs connect visual representation learning with language reasoning and are increasingly common in multimodal AI roles.

## Learning objectives

- Derive CLIP-style contrastive alignment.
- Understand projector and cross-attention interfaces between vision encoders and LLMs.
- Explain multimodal instruction tuning, freezing strategies, and hallucination.

## Terminology and abbreviations

Do not memorize an abbreviation before you understand what object it refers to.

| Term | Full name | Role in this chapter |
|---|---|---|
| **VLM** | Vision–Language Model | Joint visual/language model. |
| **CLIP** | Contrastive Language–Image Pretraining | Dual-encoder image-text alignment. |
| **VQA** | Visual Question Answering | Language answers grounded in images. |
| **OCR** | Optical Character Recognition | Reading text from images. |
| **LoRA** | Low-Rank Adaptation | Efficient Transformer adaptation. |

For the repository-wide list, see [`../GLOSSARY.md`](../GLOSSARY.md).


## Big picture and design philosophy

### A VLM needs an interface between visual and language representations

A projector, resampler, or cross-attention module maps visual evidence into a form the LLM can use. This interface is often the central architectural choice.

### Alignment and generation are different problems

CLIP learns whether image/text match; a generative VLM must also condition autoregressive language generation on visual evidence.

### Grounding is the main reliability challenge

A strong language prior can dominate weak visual evidence, creating fluent but unsupported details. Evaluation must test whether claims are actually grounded in the image.

> **How to read the equations below:** first identify the problem, what each variable represents, why this formulation was chosen, and what tradeoff it introduces. The equation is the precise implementation of the idea—not the idea itself.


## Chapter map

- Dual encoders and contrastive image-text alignment
- Contrastive Language–Image Pretraining (CLIP) loss and retrieval
- Vision encoder + projector + LLM architecture
- Cross-attention and multimodal fusion
- Visual tokens and language tokens
- Freezing vs tuning vision encoder/projector/LLM
- Multimodal instruction tuning
- Image-text grounding and hallucination
- Vision–Language Model (VLM) evaluation: Visual Question Answering (VQA), retrieval, grounding, and robustness
- Medical/scientific VLM adaptation


---

## Core concepts and theory

### 1. Why Multimodal Alignment?

Vision encoder and language model initially live in different representation spaces.

We need a mechanism that maps visual information into a representation compatible with language reasoning/generation.

---

### 2. CLIP Dual Encoder

Image embedding:


```math
z_i=
\frac{f_I(x_i)}{\|f_I(x_i)\|}.
```


Text embedding:


```math
t_i=
\frac{f_T(c_i)}{\|f_T(c_i)\|}.
```


Similarity:


```math
s_{ij}
=
\frac{z_i^\top t_j}{\tau}.
```


Image-to-text loss:


```math
\mathcal L_{I\to T}
=
-\frac1B\sum_i
\log
\frac{e^{s_{ii}}}
{\sum_j e^{s_{ij}}}.
```


Text-to-image:


```math
\mathcal L_{T\to I}
=
-\frac1B\sum_i
\log
\frac{e^{s_{ii}}}
{\sum_j e^{s_{ji}}}.
```


Total:


```math
\mathcal L
=
\frac12(
\mathcal L_{I\to T}
+
\mathcal L_{T\to I}
).
```


---

### 3. Why Cosine Similarity?

L2 normalization removes embedding magnitude:


```math
z^\top t=\cos\theta.
```


The model is trained primarily on angular alignment rather than arbitrary vector norms.

---

### 4. Vision Encoder + Projector + LLM

Visual encoder:


```math
H_v=f_v(I)
\in\mathbb R^{N_v\times d_v}.
```


Project:


```math
\tilde H_v=P(H_v)
\in\mathbb R^{N_v\times d_{\rm LLM}}.
```


Then combine with text token embeddings $H_t$:


```math
[\tilde H_v;H_t].
```


The LLM can now attend over visual and textual tokens.

---

### 5. Why Use a Projector?

The vision representation and LLM hidden space differ in:
- dimension;
- statistical scale;
- semantic organization.

A projector learns the interface.

Simple projector:


```math
P(h)=Wh+b.
```


More expressive versions:
- MLP;
- resampler;
- cross-attention module;
- query transformer.

---

### 6. Cross-Attention Fusion

Let text hidden states be queries:


```math
Q=H_tW_Q.
```


Visual states provide keys and values:


```math
K=H_vW_K,\qquad
V=H_vW_V.
```


Then


```math
H_{\rm fused}
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)V.
```


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


```math
(I,x)\to y.
```


Optimize autoregressive response likelihood:


```math
\mathcal L
=
-\sum_{t\in y}
\log
p_\theta(y_t|I,x,y_{1:t-1}).
```


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

<!-- DEEP_DIVE_START -->
## Deep dive I: two major VLM families

### Dual encoder

```text
image → vision encoder → z_img
text  → text encoder   → z_txt
```

Train in shared embedding space.

Excellent for:
- retrieval;
- zero-shot classification;
- similarity.

But it does not directly generate long free-form answers.

### Generative VLM

```text
image → vision encoder → visual tokens
                         ↓
                     projector
                         ↓
text tokens ─────────→ LLM → response
```

Designed for:
- VQA;
- dialogue;
- captioning;
- multimodal reasoning.

This distinction should be immediate in an interview.

---

## Deep dive II: CLIP batch as a similarity matrix

Batch size $B$.

Image embeddings:


```math
Z_I\in\mathbb R^{B\times d}.
```


Text embeddings:


```math
Z_T\in\mathbb R^{B\times d}.
```


Similarity:


```math
S=Z_IZ_T^\top
\in\mathbb R^{B\times B}.
```


Diagonal entries are positive pairs. Off-diagonal entries act as negatives.

Training performs:
- image → correct text classification;
- text → correct image classification.

This makes CLIP extremely simple conceptually.

---

## Deep dive III: why a projector can work

A strong pretrained ViT already encodes objects, textures, spatial patterns, and semantics.

A strong LLM already encodes language/reasoning structure.

Instead of retraining both from scratch, learn an interface:


```math
P:\mathbb R^{d_v}\to\mathbb R^{d_{\rm LLM}}.
```


This is analogous to learning a coordinate transformation between representation systems.

If projector-only training works, it suggests much of the needed semantic information already exists in both pretrained spaces.

---

## Deep dive IV: token compression

Vision encoders may produce hundreds/thousands of tokens. Sending all into an LLM is expensive.

A resampler can compress:


```math
N_v\text{ visual tokens}
\rightarrow
M\text{ latent tokens},
\qquad M\ll N_v.
```


Cross-attention from learned queries $Q_l$ to image features:


```math
H=
\mathrm{softmax}
\left(
\frac{Q_lK_v^\top}{\sqrt d}
\right)V_v.
```


This creates a fixed-size visual summary.

Tradeoff:
- fewer tokens → cheaper LLM context;
- excessive compression → lost detail/grounding.

---

## Deep dive V: multimodal fine-tuning stages

A practical staged strategy:

### Stage A — alignment
Freeze LLM + vision encoder; train projector.

### Stage B — instruction tuning
Train projector + LoRA on LLM.

### Stage C — deeper adaptation
Possibly unfreeze selected vision layers or add vision-side LoRA.

This progression increases capacity gradually and helps diagnose where adaptation is needed.

---

## Deep dive VI: spatial grounding challenge

Pure global image-text matching does not guarantee localization.

For grounding, a model may need:
- region features;
- coordinates;
- detection supervision;
- segmentation masks;
- high-resolution visual tokens.

A model can answer “there is a dog” correctly while not knowing where the dog is.

So VLM evaluation should distinguish:
- semantic recognition;
- grounding;
- counting;
- OCR;
- spatial relations;
- fine-grained attributes.

---

## Deep dive VII: hallucination as modality imbalance

Suppose language model prior says:


```math
p(\text{common object}|\text{text context})
```


is high, but visual evidence is weak.

The combined model may choose a linguistically plausible answer not supported by the image.

You can conceptualize this as competition between:
- language prior;
- visual likelihood/evidence.

Mitigation:
- better visual features;
- stronger grounding data;
- higher-resolution input;
- refusal/uncertainty behavior;
- contrastive or region-level losses;
- retrieval/tool assistance.

---

## Minimal projector example

```python
vision = vision_encoder(image)      # [B,Nv,Dv]
visual_tokens = projector(vision)   # [B,Nv,Dllm]

text_tokens = token_embed(input_ids)  # [B,Nt,Dllm]

x = torch.cat([visual_tokens, text_tokens], dim=1)
logits = llm(inputs_embeds=x)
```

This minimal design hides many production details but makes the core interface explicit.

---

## Worked design question

**Build a medical VLM for image + clinical text. What would you freeze?**

Reasonable first experiment:
1. pretrained vision encoder;
2. pretrained language model;
3. train small projector on paired image/report data;
4. perform instruction tuning with LoRA;
5. evaluate both answer quality and visual grounding;
6. compare with text-only baseline to prove images are actually used.

The text-only baseline is important. Otherwise a model may answer from dataset priors without looking at the image.
<!-- DEEP_DIVE_END -->

<!-- SECOND_DEEP_DIVE_START -->
## Architecture case study: three ways to connect image and language

### 1. Shared embedding
CLIP style.


```math
I\rightarrow z_I,
\quad
T\rightarrow z_T.
```


Best for matching/retrieval.

### 2. Visual prefix
Project image tokens to LLM dimension and prepend:


```math
[\text{visual tokens};\text{text tokens}].
```


LLM self-attention handles both.

### 3. Cross-attention
Keep visual memory separate and let text query it.

Each design creates different compute and modularity tradeoffs.

---

## Visual token budget

Suppose ViT gives 576 tokens and LLM prompt has 1024 text tokens.

Concatenated context:


```math
T=1600.
```


Dense attention cost depends on


```math
T^2=2.56\times10^6.
```


Compress image to 64 tokens:


```math
T=1088,
\qquad
T^2\approx1.18\times10^6.
```


Visual compression can nearly halve total attention pair count in this example.

This is why VLM architecture cares deeply about visual token count.

---

## Multi-image and video inputs

For $F$ frames, naive token count:


```math
N_{\rm total}=F N_{\rm frame}.
```


Video quickly becomes expensive.

Common ideas:
- temporal pooling;
- frame sampling;
- spatial token compression;
- temporal attention;
- hierarchical video encoders.

This creates a bridge from VLMs to video/world models.

---

## Contrastive vs generative multimodal objectives

Contrastive:


```math
\text{align}(I,T).
```


Generative:


```math
p(T|I).
```


Contrastive objectives learn global compatibility. Generative objectives train detailed conditional language generation.

Combining both can encourage:
- good representation alignment;
- fluent grounded generation.

---

## Negative pairs in CLIP

Batch negatives assume other image/text pairs are unrelated.

But false negatives can occur: two different captions/images may describe the same concept.

This can limit contrastive learning.

Large diverse batches improve negative coverage but also increase compute and can introduce semantically similar negatives.

---

## OCR and high-resolution failure

A VLM can recognize global scene semantics but fail on small text.

Why?
Patch/downsampling may destroy characters before LLM ever sees them.

Possible solutions:
- higher image resolution;
- specialized OCR tool;
- multi-scale features;
- crop/zoom agent;
- dedicated text detector.

This is a good example of why “bigger LLM” cannot recover information discarded upstream.

---

## VLM ablation design

If you propose a new multimodal conditioning method, ablations should isolate:

1. vision encoder choice;
2. projector/resampler;
3. visual token count;
4. frozen vs tuned vision backbone;
5. LLM LoRA vs frozen;
6. image resolution;
7. multimodal training data.

Ablations answer **which component caused improvement**.

---

## Minimal contrastive loss code

```python
img = F.normalize(image_encoder(images), dim=-1)  # [B,D]
txt = F.normalize(text_encoder(tokens), dim=-1)   # [B,D]

logits = img @ txt.T / temperature                # [B,B]
label = torch.arange(img.size(0), device=img.device)

loss_i = F.cross_entropy(logits, label)
loss_t = F.cross_entropy(logits.T, label)
loss = 0.5 * (loss_i + loss_t)
```

Be able to explain why transpose gives text-to-image direction.

---

## Mini-project: small VLM adaptation

One practical project:

```text
pretrained ViT/CLIP vision encoder
+ small language model
+ projector
+ LoRA
```

Task:
scientific-image description or VQA.

Baselines:
- text only;
- image embedding + linear classifier;
- frozen VLM;
- LoRA VLM.

Metrics:
- exact task metric;
- grounding;
- hallucination/failure examples;
- trainable params;
- memory.

This is enough to credibly discuss multimodal fine-tuning without training a giant VLM from scratch.
<!-- SECOND_DEEP_DIVE_END -->

<!-- THIRD_DEEP_DIVE_START -->
## Cross-modal attention shape walkthrough

Text:


```math
H_t\in\mathbb R^{B\times T\times d}.
```


Vision:


```math
H_v\in\mathbb R^{B\times N_v\times d}.
```


Text queries vision:


```math
Q_t\in\mathbb R^{B\times h\times T\times d_h},
```


```math
K_v,V_v
\in
\mathbb R^{B\times h\times N_v\times d_h}.
```


Score:


```math
Q_tK_v^\top
\in
\mathbb R^{B\times h\times T\times N_v}.
```


Notice complexity is


```math
O(TN_vd),
```


not $O((T+N_v)^2d)$ for this isolated cross-attention operation.

This is one reason architectures may keep modalities separate rather than concatenate every token into full self-attention.

---

## Multimodal positional information

Visual tokens have 2D/3D spatial position; language tokens have sequence position.

When combining them, the model needs to preserve modality-specific structure.

Options:
- visual encoder handles image position before projection;
- modality/type embeddings;
- separate positional schemes;
- cross-attention rather than one shared position index.

This is an architectural design problem, especially for multiple images or video.

---

## Modality dropout

During training, sometimes remove or corrupt one modality.

Why?
To prevent brittle dependence and measure whether each modality contributes.

But excessive modality dropout can teach model to rely on language priors instead of visual evidence.

Ablation:
- image present;
- image masked;
- text-only;
- shuffled image.

If prediction barely changes when image is shuffled, the model may not be visually grounded.

---

## VLM data problem

Paired image-text data vary in quality.

Captions may be:
- incomplete;
- noisy;
- generated;
- biased toward obvious objects;
- not grounded to every detail.

Instruction data may teach fluent answering without improving perception.

Therefore multimodal capability depends on:
- visual pretraining;
- pairing quality;
- grounding supervision;
- task mixture.

This explains why VLM training is a data problem as much as an architecture problem.

---

## VLM interview design template

When asked to design a VLM:

1. define modalities and task;
2. choose pretrained encoders;
3. decide fusion interface;
4. decide token budget;
5. choose frozen vs adapted components;
6. define pretraining/alignment loss;
7. define instruction tuning;
8. evaluate modality grounding;
9. test hallucination and robustness;
10. profile compute/memory.
<!-- THIRD_DEEP_DIVE_END -->

---

## Practical intuition and implementation notes

Use this section while turning theory into code or system design.

- Track where visual information enters the language model.
- Distinguish dual-encoder alignment (CLIP-style) from generative VLM fusion.
- For fine-tuning, state exactly which components are frozen, partially tuned, or adapted with LoRA.

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
