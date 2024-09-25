## NLP -> MoE



### Yuan 2.0-M32: Mixture of Experts with Attention Router

Yuan 2.0-M32 is a bilingual mixture-of-experts (MoE) language model that utilizes an Attention Router to improve expert selection efficiency and accuracy. The model features 32 experts with only 2 active at a time, achieving competitive performance in coding, math, and various domains while using only 3.7 billion active parameters out of 40 billion total. It was trained on 2000 billion tokens and demonstrates significant computational efficiency, consuming only 9.25% of the resources required by a dense model of similar scale. Yuan 2.0-M32 outperforms the Llama3-70B model on MATH and ARC-Challenge benchmarks, achieving accuracy scores of 55.89 and 95.8, respectively. The model and its source code are publicly available on GitHub.

---

### 2023 Mixture-of-Experts Meets Instruction Tuning

**2023-05-07**

https://arxiv.org/pdf/2305.14705

---

### Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

**2023-06-18**

https://arxiv.org/pdf/2303.06182v2

This paper addresses the challenges of deploying Mixture-of-Experts (MoE) models for inference, which, despite their efficiency during training, exhibit significant latency and memory inefficiencies during deployment. The authors characterize two key MoE workloads—Language Modeling (LM) and Machine Translation (MT)—and identify sources of inefficiencies related to expert activation patterns and static gating policies. They propose three optimization techniques: Dynamic Gating, which enhances throughput and reduces memory usage; Expert Buffering, a caching mechanism that offloads inactive experts to CPU memory; and Expert Load Balancing, which redistributes workloads based on historical activation data. The proposed methods collectively improve inference efficiency, allowing for larger batch sizes and reduced resource consumption while maintaining model quality.

---

### Parameter-Efficient Sparsity Crafting from Dense to Mixture-of-Experts for Instruction Tuning on General Tasks

**2024-01-08**

https://arxiv.org/pdf/2401.02731v2

This paper introduces Parameter-Efficient Sparsity Crafting (PESC), a novel approach that transforms dense language models into sparse models using a Mixture-of-Experts (MoE) architecture. PESC integrates adapters into MoE layers, allowing for expert differentiation without modifying individual weights, which reduces computational costs and memory requirements. The authors demonstrate that their sparse models, named Camelidae, achieve state-of-the-art performance across various benchmarks, outperforming existing open-source sparse models and showing superior capabilities compared to GPT-3.5. The study emphasizes the effectiveness of PESC in enhancing model capacity during instruction tuning for general tasks.

---

### Mixtral of Experts

**2024-01-08**

https://arxiv.org/pdf/2401.04088v1

The paper introduces Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model that outperforms or matches Llama 2 70B and GPT-3.5 across various benchmarks, particularly in mathematics, code generation, and multilingual tasks. Mixtral utilizes a unique architecture where each layer consists of 8 experts, allowing for efficient parameter usage—only 13B active parameters are utilized per token despite having access to 47B parameters. The model is pretrained with a context size of 32k tokens and is available under the Apache 2.0 license. Additionally, a fine-tuned version, Mixtral 8x7B – Instruct, demonstrates superior performance in instruction-following tasks compared to other leading models. The authors also provide insights into the model's architecture, performance metrics, and its reduced biases in various benchmarks.

---

### DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

**2024-01-11**

https://arxiv.org/pdf/2401.06066

The paper introduces DeepSeekMoE, a novel Mixture-of-Experts (MoE) architecture aimed at enhancing expert specialization in language models. The authors propose two key strategies: fine-grained expert segmentation, which allows for a more flexible combination of activated experts, and shared expert isolation, which captures common knowledge and reduces redundancy. Empirical results demonstrate that DeepSeekMoE achieves competitive performance with existing models like GShard while using fewer computational resources. The model scales effectively, with a 16B parameter version showing comparable results to LLaMA2 7B, and a preliminary 145B version exhibiting substantial advantages over GShard. The authors also release the model checkpoint for public use, emphasizing its efficiency and adaptability.

---

### Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM

**2024-03-13**

https://arxiv.org/pdf/2403.07816

This paper introduces Branch-Train-MiX (BTX), a method for training Large Language Models (LLMs) that enhances their capabilities across multiple specialized domains, such as coding and math reasoning. BTX begins with a seed model, branching it to create multiple expert models trained in parallel on distinct datasets. These experts are then integrated into a unified Mixture-of-Experts (MoE) model, which is fine-tuned to optimize token-level routing. The authors demonstrate that BTX achieves superior accuracy-efficiency compared to existing methods, such as Branch-Train-Merge and sparse upcycling, while maintaining high training throughput and resilience to hardware failures. Experimental results show that BTX significantly improves performance across various tasks without suffering from catastrophic forgetting, making it a more compute-efficient approach for continued pretraining of LLMs.

---

### Mixture-of-Depths: Dynamically allocating compute in transformer-based language models

**2024-04-02**

https://arxiv.org/pdf/2404.02258v1

This paper introduces the Mixture-of-Depths (MoD) approach, which allows transformer-based language models to dynamically allocate compute resources across input sequences, optimizing the use of FLOPs (floating point operations) at different layers. By implementing a top-k routing mechanism, the model can selectively process a subset of tokens in each layer, resulting in significant compute savings without sacrificing performance. The authors demonstrate that MoD transformers can achieve comparable or improved performance relative to traditional transformers while requiring fewer FLOPs per forward pass and being faster during post-training sampling. The study highlights the importance of learned routing decisions and presents a novel integration of MoD with Mixture-of-Experts (MoE) models, enhancing efficiency and performance in language modeling tasks.

---

### Mixture of A Million Experts

**2024-07-04**

https://arxiv.org/pdf/2407.04153v1

This paper introduces the Parameter Efficient Expert Retrieval (PEER) architecture, a novel mixture-of-experts (MoE) design that enables the use of over a million tiny experts while maintaining computational efficiency. PEER utilizes product key retrieval for efficient routing, allowing for a fine-grained approach to expert selection that enhances performance without increasing computational costs. The authors demonstrate that PEER outperforms dense feedforward layers and coarse-grained MoEs in language modeling tasks, achieving better performance-compute trade-offs. Key contributions include exploring extreme MoE settings, implementing a learned index structure for routing, and conducting comprehensive ablation studies to analyze the impact of various design choices.

---

### 202409 GRIN_ GRadient-INformed MoE

**2024-09-18**

https://arxiv.org/pdf/2409.12136

---