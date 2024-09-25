## NLP -> Finetune



### LIMA: Less Is More for Alignment

**2023-05-18**

https://arxiv.org/pdf/2305.11206v1

The paper presents LIMA, a 65B parameter LLaMa language model fine-tuned on only 1,000 curated prompts and responses, demonstrating that significant performance can be achieved with minimal instruction tuning. The authors argue that most knowledge in large language models is acquired during pretraining, and that alignment can be effectively achieved with a small, high-quality dataset. LIMA's performance is compared to state-of-the-art models, showing competitive results, particularly in human preference studies. The findings suggest that data quality and diversity are more critical than sheer quantity in training effective language models.

---

### Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes

**2023-07-05**

https://arxiv.org/pdf/2305.02301v2

This paper introduces 'Distilling step-by-step', a novel method for training smaller task-specific models that outperform large language models (LLMs) while requiring significantly less training data. The authors demonstrate that their approach extracts rationales from LLMs to provide additional supervision, enabling smaller models to achieve better performance with over 50% fewer training examples compared to traditional finetuning and distillation methods. The results show that a 770M T5 model can outperform a 540B PaLM model using only 80% of the available data, highlighting the efficiency of the proposed method. The study also emphasizes the potential for smaller models to be deployed more affordably and effectively in real-world applications.

---

### Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models

**2024-01-02**

https://arxiv.org/pdf/2401.01335

This paper introduces a novel fine-tuning method called Self-Play fIne-tuNing (SPIN) that enhances the performance of weak Large Language Models (LLMs) without requiring additional human-annotated data. SPIN utilizes a self-play mechanism where the LLM generates its own training data from previous iterations and refines its capabilities by distinguishing its responses from human-generated responses. The authors demonstrate that SPIN significantly improves model performance across various benchmarks, achieving results comparable to models trained with additional preference data. Theoretical proofs confirm that the method converges when the LLM's distribution aligns with the target data distribution, and empirical evaluations show consistent performance gains through iterative training.

---

### Instruction-tuned Language Models are Better Knowledge Learners

**2024-02-20**

https://arxiv.org/pdf/2402.12847

This paper presents a novel approach called pre-instruction-tuning (PIT) to enhance the ability of large language models (LLMs) to absorb knowledge from new documents. The authors argue that traditional methods, which involve continued pre-training followed by instruction-tuning, are limited due to the complexity of documents compared to straightforward question-answer (QA) pairs. PIT involves exposing LLMs to QA pairs before training on documents, allowing models to better understand how to access knowledge. Experimental results demonstrate that PIT significantly improves QA accuracy, outperforming standard instruction-tuning by 17.8% on Llama-2 7B and 16.3% on Llama-2 70B. The study also introduces the Wiki2023 dataset for evaluating continual knowledge acquisition and shows that PIT enhances cross-domain generalization.

---

### BLADE: Enhancing Black-box Large Language Models with Small Domain-Specific Models

**2024-03-27**

https://arxiv.org/pdf/2403.18365

This paper introduces BLADE, a framework designed to enhance general large language models (LLMs) by integrating small domain-specific models. The authors identify that general LLMs often lack the necessary domain-specific knowledge for specialized tasks in fields like law and medicine. BLADE addresses this by employing a three-step process: Domain-specific Pre-training (DP) to infuse domain knowledge into the small model, Knowledge Instruction Tuning (KIT) to improve the small model's ability to generate relevant knowledge, and Bayesian Prompted Optimization (BPO) to align the outputs of the small model with the general LLM. Experimental results demonstrate that BLADE significantly outperforms existing methods in legal and medical benchmarks, showcasing its effectiveness and cost-efficiency in adapting general LLMs for specific domains.

---

### 2024 DINO Direct Nash Optimization

**2024-04-04**

https://arxiv.org/pdf/2404.03715

---

### Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing

**2024-04-18**

https://arxiv.org/pdf/2404.12253

This paper introduces ALPHALLM, a framework designed to enhance the self-improvement capabilities of Large Language Models (LLMs) without requiring additional annotations. The authors address challenges such as data scarcity, vast search spaces, and subjective feedback by integrating Monte Carlo Tree Search (MCTS) with LLMs. ALPHALLM consists of three components: an imagination module for synthesizing prompts, an efficient MCTS tailored for language tasks, and a trio of critic models providing feedback. Experimental results demonstrate that ALPHALLM significantly improves LLM performance on mathematical reasoning tasks, achieving results comparable to GPT-4, thereby showcasing the potential for self-improvement in LLMs.

---

### Is In-Context Learning Sufficient for Instruction Following in LLMs?

**2024-05-30**

https://arxiv.org/pdf/2405.19874v1

This paper evaluates the effectiveness of in-context learning (ICL) for instruction following in large language models (LLMs), specifically analyzing the URIAL method proposed by Lin et al. (2024). The authors find that while URIAL achieves reasonable performance, it still underperforms compared to instruction fine-tuning on established benchmarks like MT-Bench and AlpacaEval 2.0. They demonstrate that simply adding more ICL examples does not consistently improve performance and propose a greedy selection approach for ICL examples that enhances results but does not fully bridge the gap to fine-tuning. The study highlights the importance of high-quality, correctly matched question-answer pairs for effective ICL in instruction-following tasks and suggests that LLMs require additional post-training steps beyond ICL for optimal performance.

---