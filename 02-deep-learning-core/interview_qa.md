# Deep Learning Core — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. Explain backpropagation without saying only 'chain rule'.

Backpropagation is an efficient algorithm for computing gradients through a computation graph. The chain rule gives the mathematics, but the algorithmic insight is to reuse intermediate derivatives rather than recompute every path separately.

During the forward pass, the framework stores information needed by local operations. During backward, each node receives an upstream gradient, multiplies it by the local derivative or Jacobian-vector product, accumulates contributions, and passes the result toward earlier parameters.

**Why reverse mode?** Neural-network training has many parameters but usually one scalar loss. Reverse-mode automatic differentiation computes all parameter gradients with a cost on the same order as a small multiple of the forward computation.

**Systems connection.** Storing forward activations consumes memory. Gradient checkpointing deliberately discards some activations and recomputes them during backward, trading extra FLOPs for lower memory.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Why do gradients vanish or explode, and why do residual connections help?

Across many layers, the gradient contains a product of layer Jacobians. If the typical singular values are below one, the product shrinks exponentially; if above one, it can grow explosively.

A residual block uses `y = x + F(x)`, so the Jacobian becomes `I + dF/dx`. The identity term creates a direct route for information and gradients, and the layer can learn a correction instead of a complete replacement transformation.

**Design idea.** Residual connections are primarily an optimization/conditioning mechanism in ResNet-style networks. This is different from U-Net long skips, whose main purpose is to transfer high-resolution spatial information around an encoder bottleneck.

**Other stabilizers.** Initialization, normalization, gated activations, learning-rate control, and gradient clipping also help.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. BatchNorm versus LayerNorm: how do you decide?

The main difference is which axes provide normalization statistics.

**Batch Normalization (BatchNorm)** computes per-channel statistics using the batch and often spatial positions. It works well in many CNNs but depends on batch statistics and requires running estimates for inference.

**Layer Normalization (LayerNorm)** computes statistics across hidden features within each sample or token. It does not depend on other batch examples, which makes it natural for variable-length sequences and autoregressive inference.

**Design rationale.** Transformers may decode with batch size one and need stable behavior independent of neighboring examples, so LayerNorm or RMSNorm is much more convenient than BatchNorm.

**Follow-up.** GroupNorm is a common alternative for CNNs when batch sizes are too small for stable BatchNorm statistics.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. Why does initialization matter?

A deep network repeatedly transforms activations. If weight variance is too large, activation and gradient variance can grow with depth; if too small, signals collapse.

Variance-preserving initialization chooses weight scale based on fan-in/fan-out. Xavier/Glorot is suited to roughly symmetric activations, while He/Kaiming accounts for ReLU-like nonlinearities.

**Key idea.** Initialization places the network in a numerically trainable regime before learning begins. Residual connections and normalization reduce sensitivity but do not make initialization irrelevant.

**Fine-tuning connection.** Even when the backbone is pretrained, newly added heads, projectors, adapters, or LoRA factors still require sensible initialization.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. What is mixed-precision training and why is BF16 often preferred for large models?

Mixed precision stores and computes many tensors in 16-bit formats while keeping numerically sensitive quantities or optimizer state at higher precision.

FP16 has relatively high mantissa precision but a narrow exponent range. BF16 (Brain Floating Point 16) has the same exponent width as FP32 and therefore tolerates much wider value ranges, reducing overflow/underflow problems in large models.

**Systems benefit.** Lower precision reduces memory traffic and often unlocks fast tensor-core instructions, improving both memory footprint and throughput.

**Caveat.** Sensitive reductions, softmax, normalization, or optimizer accumulators may still need higher precision. Mixed precision is not “convert everything to 16 bit.”

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

