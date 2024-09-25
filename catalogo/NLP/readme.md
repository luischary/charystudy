## NLP



### Efficient Natural Language Response Suggestion for Smart Reply

**2017-05-01**

https://arxiv.org/pdf/1705.00652v1

This paper presents a computationally efficient method for natural language response suggestion using feed-forward neural networks with n-gram embeddings. The approach encodes messages into vectors optimized for high dot-product values with response pairs, enabling effective response suggestions in Gmail's Smart Reply feature. Compared to traditional sequence-to-sequence models, this method achieves similar quality with significantly reduced computational requirements and latency. The authors discuss the transition from rule-based systems to data-driven models, the challenges of end-to-end dialog systems, and the implementation of a hierarchical structure for efficient response selection. The system was evaluated using anonymized Gmail data, demonstrating improvements in suggestion quality and user engagement.

---

### Attention Is All You Need

**2017-12-06**

https://arxiv.org/pdf/1706.03762v5

This paper introduces the Transformer, a novel neural network architecture that relies entirely on attention mechanisms, eliminating the need for recurrence and convolutions. The authors demonstrate that the Transformer outperforms existing state-of-the-art models in machine translation tasks, achieving significant improvements in translation quality and training efficiency. Specifically, the model achieves a BLEU score of 28.4 on the WMT 2014 English-to-German task and 41.8 on the English-to-French task, while requiring less training time. The paper also discusses the architecture's components, including multi-head attention, positional encoding, and feed-forward networks, and highlights its applicability to other tasks such as English constituency parsing.

---

### CTRL: A CONDITIONAL TRANSFORMER LANGUAGE MODEL FOR CONTROLLABLE GENERATION

**2019-09-20**

https://arxiv.org/pdf/1909.05858v2

The paper introduces CTRL, a 1.63 billion-parameter conditional transformer language model designed for controllable text generation. It utilizes control codes to specify various aspects of the generated text, such as style, content, and task-specific behavior. The authors highlight the model's ability to generate text from different domains and its potential for source attribution analysis. CTRL is trained on a diverse dataset, including Wikipedia, Project Gutenberg, and Amazon Reviews, and aims to enhance the controllability of language models while preserving their generality. The model and its pretrained versions are made publicly available to encourage further research in controllable generation.

---

### How to Fine-Tune BERT for Text Classification?

**2020-02-05**

https://arxiv.org/pdf/1905.05583v3

This paper investigates various fine-tuning methods for BERT (Bidirectional Encoder Representations from Transformers) specifically for text classification tasks. The authors propose a general solution that includes further pre-training on in-domain data, optional multi-task learning, and fine-tuning for the target task. They conduct extensive experiments across eight widely-studied datasets, achieving state-of-the-art results. Key findings include the effectiveness of using the top layer of BERT for classification, the importance of layer-wise learning rates to mitigate catastrophic forgetting, and the benefits of further pre-training on domain-specific data. The study emphasizes that BERT can significantly enhance performance, particularly in scenarios with limited training data.

---

### THE CURIOUS CASE OF NEURAL TEXT DeGENERATION

**2020-05-01**

https://arxiv.org/pdf/1904.09751

This paper investigates the limitations of traditional decoding strategies in neural text generation, particularly highlighting the issues of text degeneration when using maximization-based methods like beam search. The authors introduce Nucleus Sampling, a novel decoding method that dynamically samples from a subset of tokens containing the majority of probability mass, thereby improving the quality and diversity of generated text. The study demonstrates that maximization is inappropriate for open-ended generation, and that Nucleus Sampling outperforms existing methods in terms of coherence, diversity, and alignment with human-written text.

---

### Preﬁx-Tuning: Optimizing Continuous Prompts for Generation

**2021-01-01**

https://arxiv.org/pdf/2101.00190v1

This paper introduces preﬁx-tuning, a lightweight alternative to fine-tuning large pretrained language models for natural language generation tasks. Unlike fine-tuning, which modifies all model parameters, preﬁx-tuning keeps the model parameters frozen and optimizes a small continuous task-specific vector called the preﬁx. The authors demonstrate that preﬁx-tuning can achieve comparable performance to fine-tuning while requiring only 0.1% of the parameters, making it more space-efficient. The method is evaluated on table-to-text generation using GPT-2 and summarization using BART, showing superior performance in low-data settings and better extrapolation to unseen topics. The study highlights the modularity and efficiency of preﬁx-tuning, suggesting its potential for scalable applications in NLP.

---

### A Contrastive Framework for Neural Text Generation

**2022-09-26**

https://arxiv.org/pdf/2202.06417v3

This paper addresses the issue of degeneration in neural text generation, where conventional decoding methods lead to unnatural and repetitive outputs. The authors propose a contrastive training objective, SimCTG, which aims to calibrate token representations to be more isotropic and discriminative. They also introduce a novel decoding method called contrastive search, which enhances diversity while maintaining coherence in generated text. Extensive experiments demonstrate that their approach significantly outperforms existing state-of-the-art methods in both human and automatic evaluations across multiple benchmarks and languages.

---

### Foundation Transformers

**2022-10-19**

https://arxiv.org/pdf/2210.06423v2

This paper introduces MAGNETO, a new Transformer architecture designed for general-purpose modeling across various tasks and modalities, addressing the inconsistencies in existing Transformer implementations (e.g., Pre-LayerNorm vs. Post-LayerNorm). MAGNETO employs a novel Sub-LayerNorm (Sub-LN) approach, which enhances expressivity and stability during training. The authors provide a theoretical foundation for a new initialization strategy derived from DeepNet, ensuring improved training stability and scalability. Extensive experiments demonstrate that MAGNETO outperforms existing Transformer variants in language modeling, machine translation, vision pretraining, speech recognition, and multimodal tasks, while also allowing for larger learning rates without training divergence.

---

### Contrastive Search Is What You Need For Neural Text Generation

**2023-02-01**

https://arxiv.org/pdf/2210.14140

This study investigates the isotropy of autoregressive language models (LMs) and introduces the contrastive search decoding method for improved text generation. The authors find that anisotropy issues exist only in specific English GPT-2 models, while most evaluated LMs across 16 languages are isotropic. They demonstrate that contrastive search significantly outperforms existing decoding methods without additional training, achieving human-level performance in 12 out of 16 languages. The paper contributes to understanding the isotropic properties of LMs and validates the effectiveness of contrastive search across various generation tasks.

---

### Evolutionary Optimization of Model Merging Recipes

**2024-03-19**

https://arxiv.org/pdf/2403.13187v1

This paper presents a novel application of evolutionary algorithms to automate the merging of foundation models, enhancing the development of large language models (LLMs). The authors propose an evolutionary model merging approach that optimizes both parameter space and data flow space, allowing for effective combinations of diverse open-source models without extensive training data. Key contributions include the automated discovery of optimal model combinations, cross-domain merging capabilities, and the achievement of state-of-the-art performance in generating a Japanese LLM with math reasoning and a culturally-aware vision-language model (VLM). The results demonstrate that the merged models outperform existing models with significantly more parameters, showcasing the efficiency and generalizability of the proposed method. The authors emphasize the potential for this approach to democratize foundation model development and contribute to the open-source community.

---