## NLP -> State spaces



### Repeat After Me: Transformers are Better than State Space Models at Copying

**2024-02-01**

https://arxiv.org/pdf/2402.01032v1

This paper investigates the performance of transformers compared to generalized state space models (GSSMs) in tasks requiring copying from input context. The authors demonstrate that while GSSMs offer inference-time efficiency with fixed-size latent states, they are fundamentally limited in their ability to copy long sequences. Theoretical analyses show that transformers can copy strings of exponential length relative to their size, whereas GSSMs cannot accurately copy strings longer than their latent state size. Empirical results reveal that transformers outperform GSSMs in training efficiency, generalization to longer inputs, and memory-intensive tasks, even when GSSMs achieve lower perplexity in language modeling. The findings suggest a significant architectural advantage of transformers in tasks requiring context retrieval and copying.

---

### Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks

**2024-02-06**

https://arxiv.org/pdf/2402.04248v1

This study evaluates the in-context learning (ICL) capabilities of state-space models (SSMs), particularly Mamba, in comparison to Transformer models. The authors find that while SSMs perform comparably to Transformers in standard regression tasks, they excel in specific tasks like sparse parity learning but struggle with non-standard retrieval tasks. To overcome these limitations, the authors propose a hybrid model, MambaFormer, which integrates Mamba with attention blocks, achieving superior performance across various ICL tasks. The findings suggest that hybrid architectures can enhance ICL capabilities in language models, highlighting the potential of attention-free models in this domain.

---

### DenseMamba: State Space Models with Dense Hidden Connection for Efficient Large Language Models

**2024-03-05**

https://arxiv.org/pdf/2403.00818v2

This paper introduces DenseSSM, a novel framework aimed at enhancing the flow of hidden information in state space models (SSMs) for large language models (LLMs). The authors address the computational and memory challenges posed by traditional Transformer architectures by proposing a dense connection mechanism that integrates shallow-layer hidden states into deeper layers, thereby preserving fine-grained information crucial for model performance. DenseSSM maintains the training parallelizability and inference efficiency of SSMs while achieving significant accuracy improvements, exemplified by the DenseRetNet model, which outperforms the original RetNet by up to 5% on public benchmarks. The framework is applicable to various SSM types, including RetNet and Mamba, demonstrating its versatility and effectiveness in enhancing LLM capabilities.

---

### Jamba: A Hybrid Transformer-Mamba Language Model

**2024-03-28**

https://arxiv.org/pdf/2403.19887

Jamba is a novel large language model that integrates a hybrid architecture combining Transformer and Mamba layers with a mixture-of-experts (MoE) component. This architecture enhances performance and throughput while maintaining a manageable memory footprint, allowing for context lengths of up to 256K tokens. Jamba achieves state-of-the-art results on standard language model benchmarks and excels in long-context evaluations, outperforming similar models like Mixtral and Llama-2. The authors explore various architectural configurations and release their implementation under a permissive license to encourage further research and experimentation.

---

### 202405 Mamba

**2024-05-31**

https://arxiv.org/pdf/2312.00752

---

### 202407 Mamba 2

**2024-05-31**

https://arxiv.org/pdf/2405.2106

---

### SAMBA: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling

**2024-06-11**

https://arxiv.org/pdf/2406.07522v1

The paper introduces SAMBA, a hybrid architecture that combines Mamba, a selective State Space Model (SSM), with Sliding Window Attention (SWA) to efficiently model sequences with unlimited context length. SAMBA achieves linear computation complexity, allowing it to extrapolate from 4K to 256K context lengths while maintaining high memory recall and improved token predictions. The authors demonstrate that SAMBA outperforms state-of-the-art models in various benchmarks, achieving significant speedups in processing and decoding throughput. The architecture's design harmonizes the strengths of SSMs and attention mechanisms, leading to superior performance in commonsense reasoning, language understanding, and coding tasks.

---