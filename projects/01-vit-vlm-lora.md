# Project 1 — ViT/VLM + LoRA Fine-Tuning

## Goal
Demonstrate real foundation-model adaptation experience.

## Minimum experiment
1. Select a pretrained ViT or small VLM.
2. Build a small downstream dataset/task.
3. Compare:
   - frozen backbone + linear head;
   - full fine-tuning;
   - LoRA/PEFT.
4. Report trainable parameters, GPU memory, training time, validation metric.
5. Add qualitative failure examples.

## Interview story
Why adapt rather than train from scratch? Why LoRA? Which modules were targeted? What rank? What changed in memory and quality?
