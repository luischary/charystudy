## NLP -> Attention



### Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention

**2020-08-31**

https://arxiv.org/pdf/2006.16236v3

This paper introduces a linear transformer model that reduces the computational and memory complexity of traditional transformers from O(N²) to O(N), where N is the sequence length. By employing a kernel-based formulation of self-attention and utilizing the associative property of matrix products, the authors demonstrate that their linear transformers achieve comparable performance to standard transformers while being significantly faster—up to 4000 times faster during autoregressive inference. The paper also establishes a connection between transformers and recurrent neural networks (RNNs), allowing for efficient autoregressive modeling. Experimental results on tasks such as image generation and automatic speech recognition validate the effectiveness of the proposed model.

---

### Flash Attention

**2022-06-23**

https://arxiv.org/pdf/2205.14135

---

### SELF-ATTENTION DOES NOT NEED O(n2) MEMORY

**2022-10-10**

https://arxiv.org/pdf/2112.05682v3

This paper presents a novel algorithm for attention that significantly reduces memory requirements from the commonly assumed O(n²) to O(1) for single-query attention and O(log n) for self-attention. The authors propose a practical implementation that achieves O(√n) memory usage while maintaining numerical stability and performance close to standard attention methods. The work highlights the potential for processing longer sequences in neural architectures, particularly in memory-constrained environments like modern accelerators. Empirical results demonstrate substantial memory savings during inference and differentiation, with minimal impact on compute speed and accuracy in training tasks.

---

### GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**2023-05-22**

https://arxiv.org/pdf/arXiv:2305.13245v1

This paper presents two key contributions to enhance inference speed in Transformer models while maintaining quality. First, it introduces a method to uptrain existing multi-head language model checkpoints into multi-query attention (MQA) models using only 5% of the original pre-training compute, addressing the quality degradation associated with MQA. Second, it proposes grouped-query attention (GQA), which serves as an interpolation between multi-head and multi-query attention, utilizing an intermediate number of key-value heads. The results show that uptrained GQA achieves performance close to multi-head attention with inference speeds comparable to MQA, thus providing a favorable trade-off for large language models.

---

### Scaling TransNormer to 175 Billion Parameters

**2023-07-27**

https://arxiv.org/pdf/2307.14995

The paper presents TransNormerLLM, a linear attention-based Large Language Model (LLM) that surpasses traditional softmax attention models in both accuracy and efficiency. Key advancements include positional embedding, linear attention acceleration, a gating mechanism, and tensor normalization, leading to significant improvements in training and inference speed. The model scales up to 175 billion parameters and is validated on a self-collected corpus of over 6TB and 2 trillion tokens. The authors also introduce Lightning Attention, which enhances runtime performance and memory efficiency. They plan to open-source their pre-trained models to encourage further research in efficient transformer structures.

---

### Blockwise Parallel Transformer for Large Context Models

**2023-08-28**

https://arxiv.org/pdf/2305.19370v3

The paper presents the Blockwise Parallel Transformer (BPT), a novel approach to address the memory limitations of traditional Transformers when processing long sequences. BPT utilizes blockwise computation for self-attention and fuses it with feedforward network calculations, enabling the handling of input sequences up to 32 times longer than standard Transformers and 4 times longer than existing memory-efficient methods. The authors demonstrate BPT's effectiveness through extensive experiments in language modeling and reinforcement learning, showing significant reductions in memory usage and improved performance. The work contributes to the scalability of Transformer models for complex AI tasks requiring long context lengths.

---

### HyperAttention: Long-context Attention in Near-Linear Time

**2023-10-11**

https://arxiv.org/pdf/arXiv:2310.05869v2

The paper introduces HyperAttention, an approximate attention mechanism designed to improve the efficiency of long-context processing in Large Language Models (LLMs). It addresses the computational challenges of quadratic time complexity in traditional attention mechanisms by utilizing two parameters that capture the hardness of the problem. HyperAttention achieves a linear time sampling algorithm even with unbounded entries or large stable rank in the attention matrix, and it integrates well with existing implementations like FlashAttention. Empirical results demonstrate significant speed improvements, including a 50% reduction in inference time for ChatGLM2 with a context length of 32k, while maintaining competitive performance on various long-context datasets. The authors also explore the application of HyperAttention in causal masking scenarios, showcasing its versatility and efficiency in handling long sequences.

---

### Ring Attention with Blockwise Transformers for Near-Infinite Context

**2023-11-27**

https://arxiv.org/pdf/arXiv:2310.01889v4

This paper presents Ring Attention with Blockwise Transformers, a novel approach to enhance the scalability of Transformers for long sequences by distributing computations across multiple devices. The method leverages blockwise computation of self-attention and feedforward networks, allowing for training and inference of sequences significantly longer than previous memory-efficient Transformers, achieving context lengths exceeding 100 million tokens without approximations or additional overheads. The authors demonstrate the effectiveness of their approach through extensive experiments on language modeling and reinforcement learning tasks, highlighting its potential for applications requiring large context sizes.

---

### Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models

**2024-01-09**

https://arxiv.org/pdf/2401.04658

This paper introduces Lightning Attention-2, an innovative linear attention mechanism designed to efficiently handle unlimited sequence lengths in large language models (LLMs). The authors address the limitations of existing linear attention algorithms, particularly in causal settings, by employing a 'divide and conquer' strategy that separates intra-block and inter-block computations. This approach allows for consistent training and inference speeds regardless of sequence length while maintaining a fixed memory footprint. The implementation leverages GPU hardware effectively through tiling techniques. Experimental results demonstrate that Lightning Attention-2 significantly outperforms previous attention mechanisms, such as FlashAttention-2 and Lightning Attention-1, in terms of speed and memory efficiency, making it a promising solution for training LLMs with long sequences.

---

### BurstAttention: An Efficient Distributed Attention Framework for Extremely Long Sequences

**2024-03-14**

https://arxiv.org/pdf/2403.09347

The paper introduces BurstAttention, a distributed attention framework designed to efficiently process extremely long sequences in Transformer-based large language models (LLMs). The authors address the challenges of quadratic time and memory complexities in traditional attention modules by proposing a two-step partitioning strategy that divides sequences across multiple devices and further splits subsequences within each device. BurstAttention optimizes memory access and communication through global attention optimization (GAO) and local attention optimization (LAO), significantly reducing communication overheads by 40% and achieving a 2× speedup during training on long sequences. Experimental results demonstrate that BurstAttention outperforms existing solutions, such as RingAttention and tensor parallelism, in terms of memory efficiency and scalability, making it a promising approach for handling long-sequence attention tasks.

---