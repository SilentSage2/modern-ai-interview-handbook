# AI Systems Cheat Sheet

## Training memory
Parameters + gradients + optimizer states + activations.

## Parallelism
- Data parallel: replicate model, split batches.
- Tensor parallel: split tensor operations/weights across devices.
- Pipeline parallel: split layers/stages.
- FSDP/ZeRO: shard model states.

## Inference
- Prefill: process prompt, build KV cache.
- Decode: autoregressive token generation.
- Latency: time per request.
- Throughput: work per unit time.
- Quantization: lower precision to reduce memory/bandwidth and often accelerate execution.
