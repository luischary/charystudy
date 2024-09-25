## NLP -> LongContext -> Memory



### MEMORIZING TRANSFORMERS

**2022-03-16**

https://arxiv.org/pdf/2203.08913

This paper introduces a novel approach to enhance language models by enabling them to memorize and retrieve information at inference time using a k-nearest neighbor (kNN) lookup mechanism. The authors demonstrate that integrating a non-differentiable memory of (key, value) pairs significantly improves language modeling performance across various benchmarks, including web text, mathematics, and code. The study shows that increasing the memory size leads to better model perplexity and that models can generalize to larger memory sizes than they were trained on. The proposed method allows for efficient long-range attention without backpropagating gradients into the external memory, making it scalable for large datasets. The results indicate that the Memorizing Transformer can achieve performance comparable to larger models with significantly fewer parameters, highlighting its potential for future applications in natural language processing.

---

### Recurrent Memory Transformer

**2022-12-08**

https://arxiv.org/pdf/2207.06881v2

The paper introduces the Recurrent Memory Transformer (RMT), a memory-augmented segment-level recurrent Transformer model designed to address the limitations of traditional Transformers in handling long sequences. RMT incorporates memory tokens to store and process both local and global information, enabling effective information transfer between segments. The authors demonstrate that RMT outperforms Transformer-XL in tasks requiring long-term dependencies, such as copy, reverse, and associative retrieval tasks, while also achieving comparable performance in language modeling with significantly smaller memory sizes. The study highlights the potential of RMT for applications in algorithmic tasks and reasoning, and it shows that RMT can be combined with existing Transformer architectures to enhance performance.

---

### Extended Mind Transformers

**2024-04-06**

https://arxiv.org/pdf/2406.02332v1

This paper introduces Extended Mind Transformers, a novel decoder-only transformer architecture that enhances the retrieval and utilization of external memories during text generation without the need for fine-tuning. The authors address limitations of previous methods, particularly those related to positional encodings and the integration of retrieval mechanisms. They demonstrate that their approach outperforms state-of-the-art models by 6% on average in a new counterfactual long-range retrieval benchmark. Key contributions include the introduction of a kNN-based retrieval mechanism, the importance of using retrieved information across multiple decoder layers, and the efficiency of their method in terms of inference time. The paper also discusses new techniques for causal citations and active learning generation, highlighting the potential for improved reasoning and reduced hallucinations in language models.

---

### Associative Recurrent Memory Transformer

**2024-07-05**

https://arxiv.org/pdf/2407.04841v1

This paper presents the Associative Recurrent Memory Transformer (ARMT), a neural architecture designed for processing very long sequences with constant time complexity. ARMT combines transformer self-attention for local context with segment-level recurrence for long-term memory, outperforming existing models in associative retrieval tasks and achieving a new accuracy record of 79.9% on the BABILong multi-task benchmark with inputs of up to 50 million tokens. Key contributions include the introduction of a novel architecture that enhances memory capacity and generalization, as well as an original evaluation method for memory in associative retrieval tasks. The authors demonstrate that ARMT significantly surpasses previous models like RMT and Mamba in both memory efficiency and long-context processing capabilities.

---

### HUMAN-LIKE EPISODIC MEMORY FOR INFINITE CONTEXT LLMS

**2024-07-12**

https://arxiv.org/pdf/2407.09450

This paper introduces EM-LLM, a novel architecture that integrates human-like episodic memory and event cognition into large language models (LLMs), enabling them to process virtually infinite context lengths efficiently. EM-LLM organizes token sequences into coherent episodic events using surprise-based segmentation and graph-theoretic boundary refinement. The model demonstrates superior performance on long-context tasks, outperforming the state-of-the-art InfLLM by 4.3% overall, with a notable 33% improvement on the PassageRetrieval task. The findings suggest strong correlations between EM-LLM's event segmentation and human-perceived events, bridging AI and cognitive science, and providing a framework for exploring human memory mechanisms.

---