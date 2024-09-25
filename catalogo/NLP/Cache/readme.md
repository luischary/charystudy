## NLP -> Cache



### Layer-Condensed KV Cache for Efficient Inference of Large Language Models

**2024-05-17**

https://arxiv.org/pdf/arXiv:2405.10637

This paper presents a novel method to reduce memory consumption and improve throughput in large language models (LLMs) by condensing the key-value (KV) cache. The authors propose a transformer decoder variant that computes and caches KVs from only the top layer, significantly decreasing memory usage while maintaining competitive performance in language modeling and downstream tasks. Their method achieves up to 26× higher throughput compared to standard transformers and allows for integration with existing memory-saving techniques. The paper also addresses challenges in training due to sequential dependencies and introduces an approximate training method to enable parallelization. Extensive experiments demonstrate the effectiveness and efficiency of the proposed approach.

---

### LazyLLM: DYNAMIC TOKEN PRUNING FOR EFFICIENT LONG CONTEXT LLM INFERENCE

**2024-07-19**

https://arxiv.org/pdf/2407.14057

The paper introduces LazyLLM, a novel method for improving the inference efficiency of transformer-based large language models (LLMs) during the prefilling stage. LazyLLM selectively computes the key-value (KV) cache for only the tokens essential for predicting the next token, allowing for dynamic token pruning across different generation steps. This approach contrasts with static pruning methods by enabling the revival of previously pruned tokens, thus maintaining accuracy. Experimental results demonstrate that LazyLLM significantly accelerates the time-to-first-token (TTFT) and overall generation speed without requiring fine-tuning, achieving up to 2.34× speedup in multi-document question-answering tasks while preserving performance. The method can be seamlessly integrated into existing LLMs, making it a practical solution for enhancing inference efficiency.

---