## NLP -> LongContext -> Positional encoding



### Effective Long-Context Scaling of Foundation Models

This paper presents a series of long-context large language models (LLMs) capable of processing up to 32,768 tokens, built through continual pretraining from LLAMA 2 with longer training sequences. The authors demonstrate that their models achieve significant improvements over LLAMA 2 on both long-context and standard short-context tasks, outperforming existing open-source models and even surpassing gpt-3.5-turbo-16k on long-context benchmarks. Key contributions include a refined positional encoding method to enhance long-context modeling, a cost-effective instruction tuning procedure without human-annotated data, and extensive evaluations across various tasks. The findings suggest that context length is a crucial factor in scaling LLMs, and the paper provides insights into the design choices that influence model performance.

---

### A Length-Extrapolatable Transformer

**2022-12-20**

https://arxiv.org/pdf/2212.10554

This paper addresses the challenge of length extrapolation in Transformers, which typically struggle with input lengths beyond their training range. The authors introduce the Length-Extrapolatable (LEX) Transformer, which employs a novel Extrapolatable Position Embedding (XPOS) to enhance attention resolution and enable effective modeling of longer sequences. Key contributions include defining attention resolution as a metric for extrapolation, proposing a relative position encoding method, and implementing blockwise causal attention during inference. Experimental results demonstrate that the LEX Transformer outperforms existing models in both interpolation and extrapolation tasks, achieving lower perplexity across varying input lengths.

---

### The Impact of Positional Encoding on Length Generalization in Transformers

**2023-05-31**

https://arxiv.org/pdf/2305.19466

This paper investigates the role of positional encoding (PE) in length generalization for decoder-only Transformers. The authors compare five PE methods: Absolute Position Embedding (APE), T5's Relative PE, ALiBi, Rotary, and a model without positional encoding (NoPE). Their systematic empirical study reveals that NoPE outperforms all explicit PEs in length generalization tasks, achieving this without additional computational overhead. The findings suggest that explicit PEs may not be essential for effective generalization to longer sequences. Additionally, the study highlights that the effectiveness of scratchpad techniques varies by task and does not universally enhance generalization. The authors provide theoretical insights into how NoPE can represent both absolute and relative positional information.

---

### EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION

**2023-06-28**

https://arxiv.org/pdf/arXiv:2306.15595v2

This paper introduces Position Interpolation (PI), a method to extend the context window sizes of RoPE-based pretrained large language models (LLMs) like LLaMA from 2048 to 32768 tokens with minimal fine-tuning. The authors demonstrate that PI effectively preserves model quality on tasks within the original context window while significantly improving performance on tasks requiring longer contexts, such as language modeling and document summarization. The method down-scales input position indices to match the original context window, avoiding the pitfalls of extrapolation that can lead to high attention scores and instability. Empirical results show that models extended via PI require only 1000 fine-tuning steps to adapt, achieving notable perplexity gains and maintaining competitive performance on standard benchmarks.

---

### EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION

**2023-06-28**

https://arxiv.org/pdf/arXiv:2306.15595v2

This paper introduces Position Interpolation (PI), a method to extend the context window sizes of RoPE-based pretrained large language models (LLMs) like LLaMA from 2048 to 32768 tokens with minimal fine-tuning. The authors demonstrate that PI effectively preserves model quality on tasks within the original context window while significantly improving performance on tasks requiring longer contexts, such as language modeling and document summarization. The method down-scales input position indices to match the original context window, avoiding the pitfalls of extrapolation that can lead to high attention scores and instability. Empirical results show that models extended via PI require only 1000 fine-tuning steps to adapt, achieving notable perplexity gains and maintaining competitive performance on standard benchmarks.

---

### YaRN: Efficient Context Window Extension of Large Language Models

**2023-08-31**

https://arxiv.org/pdf/submit/5089635

The paper presents YaRN (Yet another RoPE extensioN method), a novel approach to extend the context window of transformer-based language models using Rotary Position Embeddings (RoPE). YaRN is shown to be compute-efficient, requiring 10x fewer tokens and 2.5x fewer training steps compared to previous methods. The authors demonstrate that LLaMA models can effectively utilize longer context lengths beyond their original training limits and achieve state-of-the-art performance in context window extension. The method also allows for extrapolation beyond the fine-tuning dataset's context length, making it suitable for various NLP tasks. The authors provide checkpoints for LLaMA 2 models fine-tuned with YaRN, showcasing its effectiveness in improving perplexity and maintaining performance across standardized benchmarks.

---

### Functional Interpolation for Relative Positions Improves Long Context Transformers

**2023-10-06**

https://arxiv.org/pdf/2310.04418

This paper introduces FIRE (Functional Interpolation for Relative Positional Encoding), a novel method to enhance the length generalization of Transformer models. The authors argue that traditional position encodings limit performance on longer inputs, despite Transformers' theoretical capacity to handle them. FIRE employs a learnable function and progressive interpolation to create a bounded input for position encoding, allowing better adaptation to varying input lengths. The authors prove that FIRE can represent existing popular relative position encodings and demonstrate its superior performance in zero-shot language modeling and long text benchmarks, significantly outperforming other methods like T5's RPE, Alibi, and Kerple. Additionally, FIRE maintains competitive performance on shorter sequences, indicating its versatility and effectiveness in diverse natural language processing tasks.

---

### Effective Long-Context Scaling of Foundation Models

**2023-10-17**

https://arxiv.org/pdf/2309.16039v2

This paper presents a series of long-context large language models (LLMs) that effectively handle context windows of up to 32,768 tokens. The models are built through continual pretraining from LLAMA 2 with longer training sequences and a dataset emphasizing long texts. The authors demonstrate significant improvements in both long-context and standard short-context tasks compared to LLAMA 2, with the 70B variant outperforming gpt-3.5-turbo-16k on various benchmarks. Key contributions include a novel positional encoding modification to enhance long-context modeling, an efficient continual pretraining approach, and a cost-effective instruction tuning method that does not rely on human-annotated long instruction data. The paper also provides extensive evaluations and analyses of the models' performance across multiple tasks, highlighting the importance of context length in scaling LLMs.

---

### Transformers Can Achieve Length Generalization But Not Robustly

**2024-02-14**

https://arxiv.org/pdf/2402.09371v1

This paper investigates the length generalization capabilities of Transformers, specifically in the context of adding two integers. The authors demonstrate that Transformers can extrapolate to sequences 2.5 times longer than those seen during training, achieving over 98% accuracy with the right combination of position encoding and data format. Key contributions include identifying the critical role of position encoding and data formatting in length generalization, revealing the fragility of this generalization influenced by factors like weight initialization and training data order, and providing a systematic evaluation of various position encodings and data formats. Despite achieving significant results, the authors note that robust length generalization remains a challenge.

---

### POSE: EFFICIENT CONTEXT WINDOW EXTENSION OF LLMS VIA POSITIONAL SKIP-WISE TRAINING

**2024-02-21**

https://arxiv.org/pdf/2309.10400

This paper introduces Positional Skip-wisE (PoSE) training, a novel method for efficiently extending the context window of Large Language Models (LLMs) without the need for full-length fine-tuning. PoSE simulates long inputs by manipulating position indices within a fixed context window, significantly reducing memory and time overhead while maintaining performance. The authors demonstrate that PoSE can extend the LLaMA model's context to 128k tokens using only a 2k training context window, and it is compatible with various RoPE-based LLMs and position interpolation strategies. Experimental results show that PoSE achieves comparable performance to full-length fine-tuning with minimal degradation, making it a promising approach for handling long input sequences in various applications.

---