# ViT & Vision Foundation Models — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. How does ViT convert an image into a Transformer sequence?

A Vision Transformer (ViT) divides an image into non-overlapping patches, flattens each patch, and linearly projects it to a token embedding.

For image height `H`, width `W`, and patch size `P`:

```math
N
=
\frac{HW}{P^2}.
```

A learnable CLS token may be prepended for global classification, and positional information is added before Transformer blocks.

**Key idea.** ViT turns a 2D image into the same abstract object a language Transformer expects: a sequence of vectors. This architectural unification is why Transformer technology transfers naturally across modalities.

**Implementation note.** Patch flattening plus linear projection can be implemented as Conv2D with kernel size and stride equal to `P`.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Why does patch size matter so much?

Smaller patches preserve finer spatial detail but create more tokens. Halving patch size in both image dimensions multiplies token count by four.

Because global self-attention scales quadratically with token count, the number of pairwise attention scores can grow roughly sixteen-fold.

**Tradeoff.**
- Large patches: cheaper but coarser.
- Small patches: detailed but expensive.
- High-resolution or 3D data: token count can become the dominant constraint.

This is why hierarchical, windowed, axial, or factorized attention designs are common in high-resolution vision.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. Why can ViTs require more data than CNNs?

CNNs encode strong image priors: locality, weight sharing, and translation equivariance. These assumptions reduce what must be learned from data.

ViTs impose weaker spatial priors. Their flexibility can be advantageous with large-scale pretraining, but with small datasets they may be less sample efficient.

**Best interview framing.** This is a bias–data tradeoff, not simply “attention is global.” Large pretrained ViTs transfer extremely well because pretraining supplies broad visual structure that a small supervised dataset could not teach from scratch.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. MAE versus DINO: what is the conceptual difference?

Masked Autoencoder (MAE) hides many image patches and trains the model to reconstruct the missing content. This encourages representations that preserve contextual information useful for predicting what is absent.

DINO-style Self-Distillation with No Labels trains a student to match a teacher's representation across different image views. This emphasizes view-invariant semantic structure rather than raw reconstruction.

**Design lesson.** The self-supervised target determines what the representation values:
- MAE: predictive/reconstructive information;
- DINO: invariant semantic consistency;
- CLIP: language-aligned concepts.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. Why is CLIP important for vision foundation models?

Contrastive Language–Image Pretraining (CLIP) learns image and text encoders whose normalized embeddings are close for matching pairs and far for mismatched pairs.

This creates:
1. a semantic visual representation tied to natural language;
2. zero-shot classification by comparing an image embedding to text prompt embeddings;
3. a reusable retrieval/alignment space for multimodal systems.

**Bigger idea.** Language becomes an open-ended label space. Instead of a classifier with fixed class indices, the model can compare images to arbitrary textual concepts.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. What makes a model a vision foundation model rather than simply a ViT?

The architecture alone does not define a foundation model. A vision foundation model should be pretrained broadly enough that its representation can be adapted across multiple downstream tasks, domains, or prompting interfaces.

Evidence might include transfer to classification, segmentation, retrieval, detection, promptable segmentation, or multimodal alignment.

A narrow ViT trained only for one task is still task-specific. Foundation-model claims should be justified by breadth of pretraining and demonstrated transfer, not by size or the word Transformer.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q7. How would you adapt a 2D ViT to 3D medical images?

Several approaches are reasonable:
- process slices independently and aggregate across depth;
- use 3D patch embedding and 3D attention;
- use windowed/axial/factorized attention;
- reuse a 2D pretrained encoder and add a depth aggregation module;
- use LoRA or domain-specific self-supervised adaptation.

**First calculation:** token budget. A 192×192×128 volume with 16×16×16 patches has 1,152 tokens, so full global attention already creates over a million token pairs per head/layer.

The architecture should therefore be chosen after considering memory and whether through-plane interactions truly need global attention.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

