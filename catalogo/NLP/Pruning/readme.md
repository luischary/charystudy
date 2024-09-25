## NLP -> Pruning



### The Unreasonable Ineffectiveness of the Deeper Layers

**2024-03-26**

https://arxiv.org/pdf/2403.17887

This study investigates a layer-pruning strategy for open-weight pretrained large language models (LLMs), revealing that significant performance degradation occurs only after removing a substantial number of layers (up to 50%). The authors propose a method to identify optimal layers for pruning based on layer similarity and employ parameter-efficient finetuning (PEFT) techniques to mitigate performance loss. Results indicate that LLMs are robust to the removal of deeper layers, suggesting that current pretraining methods may not fully utilize these layers, while shallow layers are crucial for knowledge retention. The findings advocate for combining pruning with other efficiency strategies to enhance computational resource management during finetuning and inference.

---

### Compact Language Models via Pruning and Knowledge Distillation

**2024-07-19**

https://arxiv.org/pdf/2407.14679v1

This paper investigates a method for compressing large language models (LLMs) by pruning an existing model and retraining it with a fraction of the original training data. The authors present a set of best practices for model compression that includes various pruning strategies (depth, width, attention, and MLP) combined with knowledge distillation. They demonstrate the effectiveness of their approach by deriving smaller models (MINITRON 8B and 4B) from a pretrained 15B model, achieving up to 40× fewer training tokens and 1.8× compute cost savings. The MINITRON models outperform similarly-sized community models and state-of-the-art compression techniques, showcasing improved performance on multiple language tasks. Key contributions include a thorough empirical exploration of structured pruning, practical compression strategies, and the introduction of the MINITRON model family.

---