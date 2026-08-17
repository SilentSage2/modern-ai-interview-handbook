# CNN & Computer Vision — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. What is the key inductive bias of convolution?

Convolution assumes useful interactions are local and that the same local feature detector can be reused across spatial positions. Weight sharing makes parameter count independent of image size for a fixed kernel and creates translation equivariance.

**Why this helps.** Images contain strong local structure: edges and textures have similar meaning in many locations. By encoding this assumption in the architecture, CNNs do not need to learn locality from scratch.

**Tradeoff.** The same local prior can be restrictive when long-range interactions matter. Deep stacks, dilation, larger kernels, feature pyramids, or attention expand the context.

**Interview wording.** Say “locality + weight sharing + translation equivariance,” not simply “CNN is good for images.”

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. How do you calculate receptive field and why does it matter?

The receptive field of a feature is the region of the original input that can influence it. Stacking local convolutions expands that region; stride and dilation expand it faster.

For stride-one layers with kernel size `K`, a simple approximation after `L` layers is:

```math
R
=
1 + L(K-1).
```

**Design meaning.** A model classifying global structure needs enough receptive field to combine distant evidence. A segmentation model needs global context while retaining localization, motivating encoder–decoder architectures and skip connections.

**Nuance.** The theoretical receptive field can be much larger than the effective receptive field, because influence is often concentrated near the center.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. ResNet skip connection versus U-Net skip connection: what is the difference?

A ResNet residual connection adds a block input to its transformed output: `x + F(x)`. Its primary role is optimization and iterative refinement.

A U-Net long skip sends high-resolution encoder features to a decoder at a matching spatial scale. Its primary role is restoring fine localization/detail lost during downsampling.

They are both called skip connections, but they solve different problems. A strong interview answer explicitly separates:
- **residual skip:** gradient flow / optimization;
- **U-Net skip:** spatial detail / multi-scale fusion.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. Dice loss versus cross-entropy for segmentation?

Cross-entropy treats every pixel or voxel as a classification example. When background dominates, the aggregate loss can be driven by many easy background pixels.

Dice overlap measures set agreement:

```math
\operatorname{Dice}
=
\frac{2|P\cap G|}{|P|+|G|}.
```

A differentiable soft-Dice loss focuses directly on overlap and is often useful for small foreground structures.

**Tradeoff.** Dice couples predictions through a global denominator and can behave awkwardly when a class is absent. Many practical systems combine cross-entropy and Dice rather than choosing only one.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. CNN versus ViT: what should a strong answer include?

Do not answer only “ViT has global attention.”

CNNs impose locality, translation equivariance, and hierarchical feature extraction. These priors make them data-efficient and computationally efficient for dense spatial tasks.

Vision Transformers (ViTs) tokenize images and learn input-dependent interactions with weaker built-in spatial priors. They benefit strongly from large-scale pretraining and connect naturally to multimodal/LLM-style architectures, but global attention becomes expensive at high resolution.

**Best conclusion.** The choice depends on data scale, input resolution, required global context, transfer learning, and compute. Neither architecture is universally superior.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

