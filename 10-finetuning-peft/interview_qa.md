# Fine-Tuning, LoRA & PEFT — Interview Q&A

These are **full interview answers**, not answer outlines. Practice each at three levels:

- **30 seconds:** definition + key idea.
- **2 minutes:** mechanism/equation + design rationale.
- **5 minutes:** add tradeoffs, failure modes, implementation, and an example.

Do not memorize the wording; learn the reasoning structure.

## Q1. Full fine-tuning versus PEFT: what changes mathematically and operationally?

Full fine-tuning allows every pretrained parameter to move. Parameter-Efficient Fine-Tuning (PEFT) restricts the update to a much smaller structured parameter set.

Conceptually:

```math
\theta^*
=
\theta_0+\Delta\theta.
```

Full fine-tuning gives `Delta theta` freedom across the full model; PEFT constrains it.

**Operational consequence.** Frozen base weights do not need parameter gradients or optimizer moments, reducing training-state memory and task-specific checkpoint size.

**Statistical consequence.** The restriction can act as regularization. On small datasets, LoRA can sometimes match or outperform full fine-tuning because the latter has enough freedom to overfit.

The correct comparison therefore includes both systems cost and generalization.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q2. Derive LoRA and explain why low rank is plausible.

For a pretrained linear layer `W`, Low-Rank Adaptation (LoRA) freezes `W` and parameterizes its update as:

```math
\Delta W
=
\frac{\alpha}{r}BA,
```

where rank `r` is much smaller than the matrix dimensions.

The layer becomes:

```math
y
=
Wx
+
\frac{\alpha}{r}BAx.
```

Parameter count changes from `d_out × d_in` to `r(d_in + d_out)`.

**Key hypothesis.** Although the full network lives in a huge parameter space, the downstream task-specific update may have much lower intrinsic dimension.

**Initialization idea.** If one LoRA factor starts at zero, the initial update is zero, so the adapted model initially reproduces the pretrained function exactly.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q3. What does LoRA rank control, and how would you choose it?

Rank `r` bounds the maximum rank and therefore the capacity of the update.

- Small `r`: fewer trainable parameters, lower memory, stronger regularization.
- Large `r`: more adaptation capacity, more memory/compute.

There is no universal correct rank. Treat it as a capacity hyperparameter and validate it under the actual domain shift.

A convincing experiment sweeps a small set of ranks while keeping data split and evaluation fixed, then reports:
- target metric;
- trainable parameters;
- peak GPU memory;
- training time;
- retention/general-capability metric.

This demonstrates why a chosen rank is justified rather than copied from a tutorial.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q4. What is QLoRA and why does it save more memory than LoRA alone?

LoRA freezes the base model but normally still stores those frozen weights at a relatively high precision. Quantized LoRA (QLoRA) additionally stores the frozen base in a low-bit representation while training LoRA adapters in a training-friendly precision.

The techniques attack different memory terms:
- **LoRA:** reduces trainable gradients and optimizer state.
- **Quantization:** reduces frozen base-weight storage.

This makes it possible to adapt a much larger model on the same GPU memory.

**Important nuance.** Quantization introduces approximation and kernel/data-type constraints. QLoRA is not simply “LoRA with a smaller rank.”

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q5. How do you choose LoRA target modules?

Transformer candidates include query/key/value/output attention projections and MLP projections.

The design question is: **where does the downstream task need adaptation capacity?**

- Attention-only adapters can change routing and relational interactions cheaply.
- Adding MLP projections changes feature transformations more broadly.
- Very narrow targeting may underfit a large domain shift.

A strong project should report the exact target modules, rank, alpha, dropout, learning rate, sequence length, and trainable parameter count.

There is no universally optimal module list; it should be treated as an architecture/hyperparameter choice.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q6. How do you detect catastrophic forgetting?

Evaluate both the target task and representative original/general capabilities before and after adaptation.

If target performance improves but unrelated/general tasks regress strongly, the model has forgotten broad behavior.

Mitigation includes:
- smaller learning rate;
- PEFT instead of full fine-tuning;
- mixing general/pretraining-like data;
- regularization to a reference;
- freezing lower layers;
- early stopping.

**Key point.** Target validation alone cannot reveal forgetting. A retention evaluation suite is necessary if preserving broad model capability matters.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q7. Fine-tuning versus RAG: when should you use which?

Retrieval-Augmented Generation (RAG) changes the **context**; fine-tuning changes the **parameters**.

Use RAG when the primary issue is missing, proprietary, fresh, or frequently changing knowledge. Use fine-tuning when the model needs a new behavior, style, output format, domain mapping, or task competence.

They are complementary. A domain assistant can use LoRA to learn how to follow a workflow while using RAG to retrieve current documentation.

A useful interview shorthand is: **fine-tuning teaches how to behave; RAG supplies what to know right now**, while acknowledging that real systems can have overlap.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

## Q8. What would a convincing fine-tuning project report?

A strong report includes:
- pretrained checkpoint;
- dataset size and split;
- chat/prompt format;
- trainable modules;
- optimizer and learning rate;
- precision;
- batch and sequence length;
- peak GPU memory;
- training time;
- task metric;
- retention/generalization metric;
- failure examples.

At minimum compare a frozen baseline and LoRA; include full fine-tuning if compute permits.

The goal is to show that “fine-tuning experience” means understanding data formatting, optimization, parameter efficiency, systems cost, evaluation, and failure modes—not only calling a PEFT library function.

**Likely follow-up:** connect the concept to an implementation, compare with the nearest alternative, and identify one failure mode.

---

