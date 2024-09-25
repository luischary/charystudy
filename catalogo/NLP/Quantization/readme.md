## NLP -> Quantization



### The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

**2024-02-27**

https://arxiv.org/pdf/2402.17764

This paper introduces BitNet b1.58, a 1-bit Large Language Model (LLM) variant where each parameter is ternary (-1, 0, 1). BitNet b1.58 matches the performance of full-precision LLMs (FP16/BF16) in perplexity and end-task performance while significantly reducing latency, memory usage, throughput, and energy consumption. The authors propose a new scaling law for LLMs, emphasizing the cost-effectiveness and efficiency of 1-bit LLMs, which can lead to the development of specialized hardware. The study demonstrates that BitNet b1.58 outperforms traditional models in various metrics, including zero-shot accuracy and energy efficiency, paving the way for future advancements in LLM architectures.

---

### GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection

**2024-06-03**

https://arxiv.org/pdf/2403.03507

This paper presents GaLore, a novel training strategy for large language models (LLMs) that enhances memory efficiency by utilizing gradient low-rank projection. Unlike traditional low-rank adaptation methods like LoRA, which limit the parameter search space, GaLore allows for full-parameter learning while significantly reducing memory usage—up to 65.5% in optimizer states. The authors demonstrate that GaLore maintains performance comparable to full-rank training during both pre-training and fine-tuning phases, enabling the training of a 7B model on consumer GPUs with limited memory. The method is compatible with various optimizers and introduces minimal additional hyperparameters, making it a practical solution for memory-efficient LLM training.

---