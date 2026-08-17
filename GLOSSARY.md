# Glossary — Abbreviations and Core Terms

Use this whenever a job description, paper, or chapter introduces an unfamiliar abbreviation. Learn the **full name, purpose, mechanism, and tradeoff** before memorizing the acronym.

| Term | Full name | What it means / why it matters |
|---|---|---|
| **AI** | Artificial Intelligence | Broad field of building systems that perform tasks associated with intelligent behavior. |
| **ML** | Machine Learning | Learning predictive or decision rules from data. |
| **DL** | Deep Learning | Machine learning based on multi-layer neural networks. |
| **SGD** | Stochastic Gradient Descent | Mini-batch gradient optimizer. |
| **Adam** | Adaptive Moment Estimation | Adaptive optimizer using first and second gradient moments. |
| **AdamW** | Adam with decoupled Weight Decay | Adam variant that applies weight decay separately from adaptive gradient scaling. |
| **ROC-AUC** | Area Under the Receiver Operating Characteristic Curve | Threshold-independent ranking metric for binary classification. |
| **PR-AUC** | Area Under the Precision–Recall Curve | Ranking metric often more informative for imbalanced positive classes. |
| **ECE** | Expected Calibration Error | Approximate confidence-calibration metric. |
| **NLL** | Negative Log-Likelihood | Minus the log probability assigned to observed data. |
| **MLP** | Multi-Layer Perceptron | Feed-forward neural network. |
| **CNN** | Convolutional Neural Network | Neural network using local convolution and weight sharing. |
| **ResNet** | Residual Network | Deep network using residual connections such as x + F(x). |
| **U-Net** | U-shaped encoder–decoder network | Architecture with long encoder-to-decoder skip connections. |
| **VAE** | Variational Autoencoder | Latent-variable generative model trained with variational inference. |
| **ELBO** | Evidence Lower Bound | Trainable lower bound on log-likelihood used in VAEs. |
| **KL** | Kullback–Leibler divergence | Measure of discrepancy between probability distributions. |
| **GAN** | Generative Adversarial Network | Generator and discriminator trained adversarially. |
| **SSL** | Self-Supervised Learning | Learns from targets automatically constructed from raw data. |
| **InfoNCE** | Information Noise-Contrastive Estimation objective | Contrastive objective that separates a positive pair from negative candidates. |
| **DDPM** | Denoising Diffusion Probabilistic Model | Learns to reverse a gradual Gaussian noising process. |
| **DDIM** | Denoising Diffusion Implicit Model | Alternative diffusion sampling formulation enabling deterministic/fewer-step generation. |
| **SDE** | Stochastic Differential Equation | Continuous-time stochastic dynamics used in score-based diffusion. |
| **ODE** | Ordinary Differential Equation | Deterministic continuous-time dynamics; probability-flow ODE is associated with score models. |
| **CFG** | Classifier-Free Guidance | Combines conditional and unconditional diffusion predictions to strengthen conditioning. |
| **DiT** | Diffusion Transformer | Transformer denoiser used inside a diffusion model. |
| **MHA** | Multi-Head Attention | Attention with multiple query/key/value heads. |
| **MQA** | Multi-Query Attention | Many query heads share one key/value head. |
| **GQA** | Grouped-Query Attention | Groups of query heads share key/value heads. |
| **FFN** | Feed-Forward Network | Per-token nonlinear MLP inside a Transformer block. |
| **RoPE** | Rotary Position Embedding | Encodes position through rotations of query/key features. |
| **RMSNorm** | Root Mean Square Normalization | Normalization based on root-mean-square magnitude. |
| **KV cache** | Key–Value cache | Stores past attention keys/values during autoregressive decoding. |
| **ViT** | Vision Transformer | Transformer applied to image patch tokens. |
| **MAE** | Masked Autoencoder | Self-supervised vision model reconstructing masked patches. |
| **DINO** | Self-Distillation with No Labels | Teacher–student self-supervised vision representation family. |
| **CLIP** | Contrastive Language–Image Pretraining | Learns aligned image and text embeddings. |
| **SAM** | Segment Anything Model | Promptable segmentation foundation model. |
| **LLM** | Large Language Model | Large pretrained token-sequence model, usually Transformer-based. |
| **SFT** | Supervised Fine-Tuning | Post-training on supervised instruction/response examples. |
| **RLHF** | Reinforcement Learning from Human Feedback | Preference-based reward and policy optimization. |
| **DPO** | Direct Preference Optimization | Directly trains preferred versus rejected responses relative to a reference model. |
| **MoE** | Mixture of Experts | Conditional computation using a router and specialized experts. |
| **PEFT** | Parameter-Efficient Fine-Tuning | Adapts a pretrained model while training a small parameter subset/addition. |
| **LoRA** | Low-Rank Adaptation | PEFT method parameterizing weight updates with low-rank factors. |
| **QLoRA** | Quantized LoRA | LoRA training over a quantized frozen base model. |
| **VLM** | Vision–Language Model | Model jointly processing visual and language information. |
| **RL** | Reinforcement Learning | Learning sequential actions/policies from reward. |
| **MDP** | Markov Decision Process | State/action/transition/reward/discount formalism for sequential decisions. |
| **TD** | Temporal-Difference learning | Bootstrapped value learning using reward plus next-state value. |
| **DQN** | Deep Q-Network | Neural Q-learning method using replay and target networks. |
| **GAE** | Generalized Advantage Estimation | Bias–variance controlled advantage estimator. |
| **PPO** | Proximal Policy Optimization | Policy-gradient method using clipped probability ratios. |
| **MPC** | Model Predictive Control | Plan with a model, execute the first action, observe, and replan. |
| **CEM** | Cross-Entropy Method | Sampling-based optimizer often used for action-sequence search. |
| **JEPA** | Joint-Embedding Predictive Architecture | Predicts representations rather than necessarily raw observations. |
| **RAG** | Retrieval-Augmented Generation | Retrieves external evidence into context before generation. |
| **ANN search** | Approximate Nearest Neighbor search | Fast approximate vector search for large embedding collections. |
| **API** | Application Programming Interface | Programmatic interface for calling software/services. |
| **SDK** | Software Development Kit | Libraries/tools for building against a platform. |
| **GPU** | Graphics Processing Unit | Highly parallel accelerator. |
| **CUDA** | Compute Unified Device Architecture | NVIDIA programming platform/runtime for GPU computing. |
| **HBM** | High-Bandwidth Memory | Fast accelerator-attached memory. |
| **DDP** | Distributed Data Parallel | Replicates a model and synchronizes gradients. |
| **FSDP** | Fully Sharded Data Parallel | Shards model parameters/gradients/optimizer states. |
| **ZeRO** | Zero Redundancy Optimizer | Family of sharding strategies for model/optimizer state. |
| **NCCL** | NVIDIA Collective Communications Library | GPU collective-communication library. |
| **FP32** | 32-bit floating point | Single-precision floating-point format. |
| **FP16** | 16-bit floating point | Half precision with narrow exponent range. |
| **BF16** | Brain Floating Point 16 | 16-bit format with FP32-like exponent range. |
| **FP8** | 8-bit floating point | Very low-precision floating-point formats on newer accelerators. |
| **ONNX** | Open Neural Network Exchange | Portable neural-network computation graph format. |
| **TensorRT** | NVIDIA TensorRT | Inference compiler/runtime for NVIDIA GPUs. |
| **PTQ** | Post-Training Quantization | Quantizes a trained model using calibration rather than full retraining. |
| **QAT** | Quantization-Aware Training | Simulates quantization during training. |
| **TTFT** | Time To First Token | Latency from request arrival to the first generated token. |
| **p50 / p95 / p99** | Latency percentiles | Median and tail-latency statistics. |
