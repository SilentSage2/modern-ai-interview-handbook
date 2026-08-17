# VLM & Multimodal Foundation Models — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. CLIP versus a generative VLM: what is the difference?

CLIP is a dual-encoder model. Image and text are encoded independently into a shared embedding space and trained so matching pairs have high similarity. This makes CLIP excellent for retrieval and zero-shot classification.

A generative Vision–Language Model (VLM) must additionally condition an autoregressive language model on visual features to produce free-form text.

**Key distinction.**
- CLIP asks: “Do these image and text representations match?”
- Generative VLM asks: “Given this image and prompt, what text should I produce?”

The two systems may share a pretrained vision backbone but differ in fusion architecture and training objective.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Why does a VLM need a projector or resampler?

A vision encoder outputs features in its own hidden dimension and representation space, while an LLM expects token-like states in the language model's hidden space.

A projector learns a mapping:

```math
P:
\mathbb R^{d_v}
\rightarrow
\mathbb R^{d_{\mathrm{LLM}}}.
```

A resampler can also compress hundreds of visual tokens into a smaller learned set.

**Design idea.** Instead of retraining two giant pretrained models from scratch, learn a compact interface between them. If projector-only training works, much of the required semantics already exists in the pretrained vision and language models.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. Concatenating visual tokens versus cross-attention: what is the tradeoff?

With concatenation, projected visual tokens are added to the same sequence as text tokens and processed by self-attention. This is simple and allows direct interaction, but increases total context length and dense attention cost.

With cross-attention, text hidden states query a separate visual memory. Its isolated score tensor scales with text length times visual-token count rather than the square of their concatenated total.

**Tradeoff.**
- Concatenation: simple, unified sequence, potentially expensive.
- Cross-attention: modular and can control visual access, but adds extra architectural modules.

The best choice depends on token budget, model reuse, and how tightly modalities must interact.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. How can you prove a VLM is actually using the image?

Use controlled ablations:
- correct image;
- shuffled image from another example;
- blank/masked image;
- text-only input.

If output accuracy barely changes, the model may rely on language priors or dataset shortcuts.

For stronger grounding evaluation, ask questions that cannot be answered from text priors, evaluate spatial localization, count objects, or perturb the visual evidence.

**Key idea.** High answer accuracy is not automatically evidence of visual grounding, especially in biased datasets where the question text predicts the answer distribution.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. Why do VLMs hallucinate visual details?

A language model has a strong prior over plausible sentences. If the vision encoder/projector provides weak, ambiguous, or low-resolution evidence, the language prior can dominate and generate a likely but unsupported object or attribute.

Other causes include:
- aggressive visual token compression;
- insufficient grounding supervision;
- low input resolution;
- instruction data rewarding fluency more than evidence.

Mitigation includes better visual features, higher resolution, region-level supervision, retrieval/tools, explicit grounding objectives, and refusal/uncertainty training.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. What does visual token budget mean and why does it matter?

Every visual token consumes context and attention compute. If a vision encoder outputs 576 tokens and the prompt uses 1,024 text tokens, concatenation creates 1,600 total tokens.

Compressing the image to 64 visual tokens can substantially reduce attention and KV-cache cost.

**Tradeoff.** Aggressive compression can remove small objects, text, or spatial relationships. The visual token budget is therefore a quality-versus-compute design parameter, not just a serving optimization.

This becomes even more important for multiple images or video, where token count can scale with frame count.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q7. How would you build a medical or scientific VLM efficiently?

A reasonable staged strategy is:
1. start from a pretrained vision encoder and LLM;
2. train a projector on paired image–report or image–caption data;
3. instruction-tune with LoRA;
4. optionally adapt selected vision layers;
5. compare with text-only and image-shuffled baselines;
6. evaluate both task accuracy and visual grounding;
7. report memory, trainable parameters, and failure modes.

This creates credible multimodal fine-tuning experience without pretraining a huge VLM from scratch.

A strong project also evaluates domain shift and whether visual evidence is actually used rather than memorized dataset correlations.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

