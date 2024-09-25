## NLP -> Models



### 2023 Gemini

---

### 2023 LLAMA 2

---

### 2024 Gemini 1_5

---

### 2024 gemini_v1_5_report

---

### 202407 LLAMA 3.1

---

### Improving Language Understanding by Generative Pre-Training

This paper presents a semi-supervised approach for natural language understanding that combines unsupervised generative pre-training of a language model on a large corpus of unlabeled text with supervised fine-tuning on specific tasks. The authors demonstrate that their task-agnostic model significantly outperforms discriminatively trained models across various benchmarks, achieving state-of-the-art results in 9 out of 12 tasks, including commonsense reasoning, question answering, and textual entailment. The model utilizes a Transformer architecture and employs task-aware input transformations to facilitate effective transfer learning with minimal architectural changes. The findings highlight the potential of leveraging large unlabeled datasets to enhance performance in NLP tasks, reducing reliance on labeled data.

---

### LLaMA: Open and Efficient Foundation Language Models

The paper introduces LLaMA, a series of foundation language models ranging from 7B to 65B parameters, trained on trillions of tokens using only publicly available datasets. LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, while LLaMA-65B is competitive with leading models like Chinchilla and PaLM. The authors emphasize the importance of training efficiency and performance at various inference budgets, and they provide insights into their training methods, architecture modifications, and the biases present in their models. The release of LLaMA aims to democratize access to high-performance language models for the research community.

---

### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**2019-05-24**

https://arxiv.org/pdf/1810.04805

The paper introduces BERT (Bidirectional Encoder Representations from Transformers), a novel language representation model that pre-trains deep bidirectional representations from unlabeled text. BERT utilizes a masked language model (MLM) and next sentence prediction (NSP) tasks to jointly condition on both left and right context, significantly improving performance on various NLP tasks. The authors demonstrate that BERT achieves state-of-the-art results on eleven benchmarks, including GLUE and SQuAD, outperforming previous models by substantial margins. The study emphasizes the importance of bidirectional pre-training and shows that BERT's architecture reduces the need for complex task-specific modifications, making it a powerful tool for a wide range of language understanding tasks.

---

### RoBERTa: A Robustly Optimized BERT Pretraining Approach

**2019-07-26**

https://arxiv.org/pdf/1907.11692v1

This paper presents RoBERTa, an improved version of BERT, through a comprehensive replication study that evaluates the impact of various hyperparameters and training data size on model performance. The authors find that BERT was significantly undertrained and propose several modifications to the training process, including longer training duration, larger batch sizes, removal of the next sentence prediction objective, and dynamic masking of input data. RoBERTa achieves state-of-the-art results on multiple benchmarks, including GLUE, RACE, and SQuAD, demonstrating the importance of design choices in pretraining language models. The authors also introduce a new dataset, CC-NEWS, and release their models and code for further research.

---

### BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

**2019-10-29**

https://arxiv.org/pdf/arXiv:1910.13461

BART is a denoising autoencoder designed for pretraining sequence-to-sequence models, combining bidirectional and autoregressive transformers. It is trained by corrupting text with various noising functions and learning to reconstruct the original text. BART demonstrates strong performance across multiple NLP tasks, achieving state-of-the-art results in text generation, summarization, and machine translation, while matching or exceeding the performance of models like RoBERTa on comprehension tasks. The authors evaluate various noising approaches, finding that a combination of text in-filling and sentence permutation yields the best results. BART's architecture and flexible noising strategies allow it to generalize well across different tasks, making it a versatile tool in natural language processing.

---

### GPT3

**2020-07-22**

https://arxiv.org/pdf/2005.14165

---

### T5

**2020-07-28**

https://arxiv.org/pdf/1910.10683

---

### 2022 InstructGPT

**2022-04-03**

https://arxiv.org/pdf/2203.02155

---

### 2022 PalM

**2022-05-10**

https://arxiv.org/pdf/2204.02311

---

### Sabiá: Portuguese Large Language Models

**2023-04-16**

https://arxiv.org/pdf/2304.07880v1

This paper presents Sabiá, a series of large language models pretrained specifically on Portuguese texts, demonstrating that monolingual pretraining significantly enhances model performance compared to multilingual counterparts. The authors further pretrained GPT-J and LLaMA models on Portuguese datasets, achieving superior results on the Poeta benchmark, which includes 14 Portuguese NLP tasks. The best model, Sabiá-65B, performs comparably to GPT-3.5-turbo, with notable improvements particularly on datasets created by native Portuguese speakers. The study emphasizes the importance of domain-specific knowledge in enhancing model capabilities, advocating for specialized models tailored to individual languages.

---

### 2023 Orca

**2023-05-06**

https://arxiv.org/pdf/2306.02707

---

### 2023 Code LLaMa

**2023-08-25**

https://arxiv.org/pdf/2308.1295

---

### Textbooks Are All You Need II: phi-1.5 technical report

**2023-09-11**

https://arxiv.org/pdf/2309.05463

This technical report presents phi-1.5, a 1.3 billion parameter Transformer-based language model that demonstrates performance on common sense reasoning and natural language tasks comparable to models five times its size. The authors emphasize the importance of high-quality, synthetic 'textbook-like' training data, which enhances the model's reasoning abilities while mitigating issues like hallucinations and biases. The report discusses the architecture, training data, and benchmarks, revealing that phi-1.5 outperforms existing models in multi-step reasoning tasks and exhibits traits of larger models. The authors aim to promote further research on LLMs by open-sourcing phi-1.5, highlighting its potential for efficient and sustainable AI development.

---

### DeBERTinha: A Multistep Approach to Adapt DebertaV3 XSmall for Brazilian Portuguese Natural Language Processing Tasks

**2023-09-28**

https://arxiv.org/pdf/2309.16844v1

This paper introduces DeBERTinha, a model adapted from the DebertaV3 XSmall for Brazilian Portuguese NLP tasks. The authors employ a multi-step training process utilizing pre-trained English model weights, creating a Portuguese-specific vocabulary of 50,000 tokens. The model is fine-tuned on tasks such as named entity recognition, sentiment analysis, and sentence relatedness, outperforming the larger BERTimbau-Large in two tasks while being significantly smaller (40M parameters). The study emphasizes the effectiveness of cross-lingual adaptation and offers a methodology for resource-constrained scenarios in NLP.

---

### Textbooks Are All You Need

**2023-10-02**

https://arxiv.org/pdf/2306.11644v2

This paper introduces phi-1, a new large language model (LLM) for code generation, which has 1.3 billion parameters and is trained on a significantly smaller dataset compared to existing models. The authors emphasize the importance of high-quality training data, which they refer to as 'textbook quality,' leading to improved performance on coding benchmarks like HumanEval and MBPP. phi-1 achieves pass@1 accuracy of 50.6% on HumanEval and 55.5% on MBPP, outperforming many larger models. The study highlights the role of data quality in enhancing model performance and suggests that smaller models can achieve state-of-the-art results with less environmental impact. The authors also discuss the emergent properties observed in phi-1 post-finetuning and the implications of their findings for future LLM development.

---

### Mistral 7B

**2023-10-10**

https://arxiv.org/pdf/2310.06825v1

Mistral 7B is a 7-billion-parameter language model designed for high performance and efficiency, outperforming the best open 13B model (Llama 2) and the best released 34B model (Llama 1) in reasoning, mathematics, and code generation. It utilizes grouped-query attention (GQA) and sliding window attention (SWA) to enhance inference speed and manage longer sequences effectively. The model is also fine-tuned for instruction-following tasks, demonstrating superior performance compared to Llama 2 13B in human and automated benchmarks. Mistral 7B is released under the Apache 2.0 license, with a focus on affordability and efficiency for real-world applications.

---

### ZEPHYR: DIRECT DISTILLATION OF LM ALIGNMENT

**2023-10-25**

https://arxiv.org/pdf/2310.16944v1

This technical report presents ZEPHYR-7B, a smaller language model aligned to user intent, achieved through a novel approach combining distilled supervised fine-tuning (dSFT) and distilled direct preference optimization (dDPO) using AI Feedback (AIF) data. The authors demonstrate that ZEPHYR-7B outperforms existing 7B parameter models, including LLAMA2-CHAT-70B, on chat benchmarks like MT-Bench, without requiring human annotation. The study emphasizes the importance of preference learning in enhancing model alignment and highlights the efficiency of the training process, which can be completed in a few hours on available hardware. The report also discusses limitations, particularly regarding safety considerations and potential biases in evaluation metrics.

---

### 2023 Orca 2

**2023-11-21**

https://arxiv.org/pdf/2311.11045

---

### SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling

**2023-12-29**

https://arxiv.org/pdf/2312.15166v2

The paper introduces SOLAR 10.7B, a large language model (LLM) with 10.7 billion parameters, which demonstrates superior performance in various natural language processing tasks. The authors propose a novel method called depth up-scaling (DUS) that efficiently scales LLMs without the complexities associated with mixture-of-experts approaches. DUS involves depthwise scaling and continued pretraining, allowing for seamless integration into existing frameworks. The model outperforms existing models like Llama 2 and Mistral 7B in benchmarks. Additionally, SOLAR 10.7B-Instruct, a variant fine-tuned for instruction-following tasks, surpasses the Mixtral-8x7B-Instruct model. The authors emphasize the model's open-source availability under the Apache 2.0 license to promote collaboration and innovation in the field.

---

### OLMo: Accelerating the Science of Language Models

**2024-02-01**

https://arxiv.org/pdf/2402.00838v1

The publication introduces OLMo, a state-of-the-art open language model and framework aimed at enhancing the scientific study of language models. Unlike previous releases that provided limited access, OLMo offers comprehensive resources including model weights, training data, and evaluation tools under a permissive license. The authors emphasize the importance of open access to model details for understanding biases and risks, and they present a detailed architecture, pretraining dataset (Dolma), and evaluation framework. OLMo includes multiple model variants and aims to facilitate research on the relationship between training data and model capabilities, while also addressing the environmental impact of model training.

---

### Gemma: Open Models Based on Gemini Research and Technology

**2024-02-21**

The publication introduces Gemma, a family of lightweight open language models developed from Google's Gemini technology. Gemma includes two model sizes (2 billion and 7 billion parameters) and demonstrates superior performance on 11 out of 18 text-based tasks compared to similar open models. The models are pretrained on extensive text data and fine-tuned for various applications, including dialogue and safety. Comprehensive evaluations highlight their capabilities in language understanding, reasoning, and safety. The authors emphasize the importance of responsible model release and provide tools for developers to ensure safe deployment.

---

### Yi: Open Foundation Models by 01.AI

**2024-03-07**

https://arxiv.org/pdf/2403.04652v1

The Yi model family, consisting of 6B and 34B pretrained language models, showcases advanced language and multimodal capabilities. The models are built on a robust data engineering pipeline that includes 3.1 trillion tokens of high-quality English and Chinese corpora. Yi models excel in various benchmarks, achieving performance comparable to GPT-3.5, particularly in reasoning and instruction-following tasks. Key innovations include a focus on data quality over quantity in both pretraining and finetuning, the introduction of long context modeling (up to 200K tokens), and the integration of vision-language capabilities. The infrastructure supports efficient training and deployment, making Yi models accessible for research and commercial applications while emphasizing safety and responsible AI practices.

---

### Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone

**2024-04-22**

https://arxiv.org/pdf/2404.14219v1

The phi-3-mini model, developed by Microsoft, is a 3.8 billion parameter language model trained on 3.3 trillion tokens, achieving performance comparable to larger models like GPT-3.5 while being small enough for deployment on mobile devices. The model's success is attributed to an innovative training dataset that combines heavily filtered web data and synthetic data, allowing it to excel in reasoning and language understanding despite its compact size. The report also discusses the training methodology, parameter scaling, and safety measures implemented to enhance the model's robustness and reduce harmful outputs.

---

### Gemma 2: Improving Open Language Models at a Practical Size

**2024-06-27**

https://arxiv.org/pdf/2408.00118v1

Gemma 2 introduces a new family of lightweight open language models ranging from 2 billion to 27 billion parameters, achieving state-of-the-art performance for their size. The authors implement technical modifications to the Transformer architecture, including interleaving local-global attentions and group-query attention, and utilize knowledge distillation to enhance training efficiency. The models are trained on extensive datasets, with the 27B model achieving competitive results against larger counterparts. The paper emphasizes the importance of safety and responsible deployment, detailing extensive evaluations across various benchmarks and human assessments, while also addressing memorization and privacy concerns.

---

### Gemma 2: Improving Open Language Models at a Practical Size

**2024-06-27**

Gemma 2 introduces a new family of lightweight open language models ranging from 2 billion to 27 billion parameters, with significant architectural improvements such as interleaving local-global attentions and group-query attention. The models are trained using knowledge distillation, resulting in superior performance compared to models of similar size and competitive performance against larger models. The paper discusses the training data, architecture, pre-training and post-training processes, and evaluates the models across various benchmarks, demonstrating advancements in language understanding, reasoning, and safety measures. The authors emphasize the importance of responsible deployment and ongoing safety evaluations.

---

### 2024 DeepSeek-V2

**2024-08-05**

https://arxiv.org/pdf/2405.04434

---

### EXAONE 3.0 7.8B Instruction Tuned Language Model

**2024-08-08**

https://arxiv.org/pdf/2408.03541v2

The EXAONE 3.0 7.8B instruction-tuned language model, developed by LG AI Research, is the first open model in its family aimed at democratizing access to expert-level AI capabilities. This model excels in Korean and demonstrates competitive performance in English across various benchmarks, particularly in real-world use cases, math, coding, and reasoning tasks. The training process involved extensive pre-training on a diverse dataset, advanced post-training techniques for instruction-following, and a robust compliance system to address legal and ethical concerns. The model is available for non-commercial research purposes, promoting innovation and collaboration within the AI community.

---