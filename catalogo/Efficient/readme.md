## Efficient



### GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection

**2024-06-02**

https://arxiv.org/pdf/2403.03507v2

The paper introduces GaLore, a novel training strategy for Large Language Models (LLMs) that enhances memory efficiency by utilizing Gradient Low-Rank Projection. Unlike traditional low-rank adaptation methods like LoRA, which limit parameter search and require full-rank warm starts, GaLore allows full-parameter learning while significantly reducing memory usage—up to 65.5% for optimizer states. The authors demonstrate that GaLore maintains performance comparable to full-rank training during both pre-training and fine-tuning stages, enabling the training of 7B parameter models on consumer GPUs without advanced memory management techniques. The method is compatible with various optimizers and introduces minimal additional hyperparameters, making it a practical solution for efficient LLM training.

---

### Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients

**2024-11-07**

https://arxiv.org/pdf/2407.08296

This paper introduces Q-GaLore, a novel method for training large language models (LLMs) that significantly reduces memory usage by combining quantization and low-rank projection techniques. Q-GaLore addresses the limitations of the existing GaLore method, which relies on computationally expensive Singular Value Decomposition (SVD) operations and offers minimal improvements in efficiency. By adaptively updating the gradient subspace based on convergence statistics and utilizing low-precision formats (INT4 for projection matrices and INT8 for weights), Q-GaLore achieves competitive performance while reducing memory consumption by up to 50% compared to LoRA and GaLore. The method allows for the pre-training of a LLaMA-7B model on a single NVIDIA RTX 4060 Ti with only 16 GB of memory, showcasing its practicality and efficiency in resource-constrained environments.

---