## State space



### 2023 Mamba

---

### 2022 S4

**2022-05-08**

https://arxiv.org/pdf/2111.00396

---

### Efficient Long Sequence Modeling via State Space Augmented Transformer

**2022-12-15**

https://arxiv.org/pdf/2212.08136

The paper introduces SPADE (State space Augmented Transformer), a novel model designed to efficiently handle long sequences in natural language processing. SPADE combines a state space model (SSM) in its bottom layer to capture global dependencies with local attention mechanisms in subsequent layers to refine local information. This hybrid approach addresses the limitations of traditional Transformers, which struggle with long sequences due to quadratic computational costs. Experimental results demonstrate SPADE's superior performance on the Long Range Arena benchmark and various language modeling tasks, showcasing its scalability and efficiency compared to existing models.

---

### Hungry Hungry Hippos: Towards Language Modeling with State Space Models

**2022-12-28**

https://arxiv.org/pdf/arXiv:2212.14052v3

This paper explores the performance of state space models (SSMs) in language modeling, addressing their limitations compared to Transformers. The authors introduce a new SSM layer, H3, designed to enhance the models' capabilities in recalling and comparing tokens, which are critical for language tasks. H3 demonstrates competitive performance, matching attention on synthetic tasks and nearly closing the perplexity gap with Transformers on the OpenWebText dataset. Additionally, the paper presents FlashConv, an efficient training algorithm for SSMs that significantly improves hardware utilization and speeds up inference. The results indicate that hybrid models combining H3 and attention outperform Transformers in various benchmarks, suggesting a promising direction for future research in language modeling.

---

### Hyena Hierarchy: Towards Larger Convolutional Language Models

**2023-04-21**

https://arxiv.org/pdf/2302.10866

This paper introduces Hyena, a subquadratic alternative to the attention mechanism in Transformers, addressing the computational limitations associated with long input sequences. Hyena combines long convolutions and data-controlled gating to achieve significant improvements in accuracy and efficiency on language modeling tasks, outperforming existing subquadratic methods. The authors demonstrate that Hyena operators can match the performance of attention-based models while requiring 20% less training compute and achieving up to 100× speedup at longer sequence lengths. The work sets a new state-of-the-art for dense-attention-free architectures, showing Hyena's potential for scaling in both language and vision tasks.

---

### Retentive Network: A Successor to Transformer for Large Language Models

**2023-07-19**

https://arxiv.org/pdf/arXiv:2307.08621v2

This paper introduces the Retentive Network (RETNET), a new architecture for large language models that combines training parallelism, low-cost inference, and competitive performance. The authors derive a connection between recurrence and attention, proposing a retention mechanism that supports parallel, recurrent, and chunkwise recurrent computation paradigms. RETNET achieves O(1) inference complexity, significantly reducing memory consumption and latency compared to Transformers, while maintaining high throughput. Experimental results demonstrate that RETNET outperforms Transformers in language modeling tasks, particularly for larger model sizes, and offers substantial improvements in training efficiency and inference costs.

---

### MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts

**2024-01-08**

https://arxiv.org/pdf/2401.04081v1

This paper introduces MoE-Mamba, a novel model that integrates Mixture of Experts (MoE) with the Mamba State Space Model (SSM) to enhance efficiency in sequential modeling. MoE-Mamba demonstrates superior performance by achieving the same results as the vanilla Mamba model in 2.2 times fewer training steps while maintaining inference performance. The authors argue that combining SSMs with MoE can unlock the potential for scaling up to larger models, with preliminary results indicating promising directions for future research. The paper also discusses the architecture, training setup, and various experiments comparing different model configurations.

---

### BlackMamba: Mixture of Experts for State-Space Models

**2024-02-01**

https://arxiv.org/pdf/2402.01771v1

This paper introduces BlackMamba, a novel architecture that integrates the Mamba state-space model (SSM) with mixture-of-experts (MoE) to enhance performance in language modeling and long sequence processing. BlackMamba achieves linear computational complexity, allowing for efficient autoregressive generation and reduced training FLOPs compared to dense transformer models. The authors present two model variants, 340M/1.5B and 630M/2.8B parameters, trained on 300 billion tokens from a custom dataset. The results demonstrate that BlackMamba outperforms both Mamba and transformer baselines in terms of inference speed and training efficiency, while also providing open-source access to model weights and inference code.

---

### SAMBA: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling

**2024-06-11**

https://arxiv.org/pdf/2406.07522v1

The paper introduces SAMBA, a hybrid architecture that combines Mamba, a selective State Space Model (SSM), with Sliding Window Attention (SWA) to efficiently model sequences with unlimited context length. SAMBA achieves linear computation complexity, allowing it to extrapolate from 4K to 256K context lengths while maintaining high memory recall and improved token predictions. The authors demonstrate that SAMBA outperforms state-of-the-art models in various benchmarks, achieving significant speedups in processing and decoding throughput. The architecture's design harmonizes the strengths of SSMs and attention mechanisms, leading to superior performance in commonsense reasoning, language understanding, and coding tasks.

---

### Jamba: A Hybrid Transformer-Mamba Language Model

**2024-07-03**

https://arxiv.org/pdf/2403.19887v2

Jamba is a novel large language model that integrates a hybrid architecture combining Transformer and Mamba layers, enhanced with a mixture-of-experts (MoE) module. This design allows for improved performance, high throughput, and a reduced memory footprint, particularly for long-context processing, supporting up to 256K tokens. The model achieves state-of-the-art results on various language benchmarks while fitting within a single 80GB GPU. The authors explore architectural configurations, revealing critical insights into the balance between memory usage, training efficiency, and model capabilities. Jamba's implementation is publicly available to encourage further research and optimization.

---