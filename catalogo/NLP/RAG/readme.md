## NLP -> RAG



### MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries

**2024-01-27**

https://arxiv.org/pdf/2401.15391

This paper introduces MultiHop-RAG, a novel dataset designed for evaluating retrieval-augmented generation (RAG) systems specifically for multi-hop queries, which require reasoning over multiple pieces of evidence. The authors highlight the inadequacy of existing RAG systems in handling such complex queries and present a comprehensive dataset that includes a knowledge base, multi-hop queries, ground-truth answers, and supporting evidence. They categorize multi-hop queries into four types: Inference, Comparison, Temporal, and Null queries. Through two experiments, the authors benchmark various embedding models and state-of-the-art large language models (LLMs) to assess their retrieval and reasoning capabilities. The results indicate significant challenges in current RAG implementations, underscoring the need for improved methods. The dataset and the implemented RAG system are publicly available to facilitate further research in this area.

---

### RAFT: Adapting Language Model to Domain Specific RAG

**2024-03-15**

https://arxiv.org/pdf/2403.10131

This paper introduces Retrieval Augmented Fine Tuning (RAFT), a novel training methodology designed to enhance the performance of large language models (LLMs) in domain-specific retrieval-augmented generation (RAG) tasks. RAFT trains models to effectively utilize relevant documents while ignoring distractor documents, thereby improving their ability to reason and extract information in an 'open-book' exam setting. The authors demonstrate that RAFT outperforms traditional supervised fine-tuning and RAG methods across various datasets, including PubMed, HotpotQA, and Gorilla. Key contributions include the integration of chain-of-thought reasoning in responses, the strategic use of oracle and distractor documents during training, and insights into the optimal proportion of training data that should include oracle documents. Overall, RAFT shows significant potential for improving LLMs in specialized domains.

---

### Retrieval-Augmented Generation for Large Language Models: A Survey

**2024-03-27**

https://arxiv.org/pdf/arXiv:2312.10997v5

This survey provides a comprehensive review of Retrieval-Augmented Generation (RAG) techniques for enhancing Large Language Models (LLMs). It discusses the limitations of LLMs, such as hallucination and outdated knowledge, and presents RAG as a solution that integrates external knowledge sources to improve accuracy and credibility. The paper categorizes RAG into three paradigms: Naive RAG, Advanced RAG, and Modular RAG, detailing their evolution and core components—retrieval, generation, and augmentation. It also outlines current evaluation frameworks, benchmarks, and challenges in RAG research, while suggesting future directions for development and integration with other AI methodologies.

---

### Searching for Best Practices in Retrieval-Augmented Generation

**2024-07-01**

https://arxiv.org/pdf/2407.01219

This paper investigates optimal practices for implementing Retrieval-Augmented Generation (RAG) techniques to enhance the performance and efficiency of large language models (LLMs). The authors analyze various RAG components, including query classification, retrieval methods, reranking, repacking, and summarization, through extensive experimentation. They propose a comprehensive framework for evaluating RAG systems and demonstrate that multimodal retrieval can improve question-answering capabilities. Key contributions include identifying best practices for RAG implementation, introducing evaluation metrics, and suggesting strategies that balance performance and efficiency in RAG applications.

---

### RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs

**2024-07-02**

https://arxiv.org/pdf/2407.02485v1

This paper introduces RankRAG, a novel instruction fine-tuning framework that enhances large language models (LLMs) for both context ranking and answer generation in retrieval-augmented generation (RAG) tasks. The authors demonstrate that by integrating a small amount of ranking data into the training process, RankRAG outperforms existing expert ranking models and achieves superior performance on various knowledge-intensive benchmarks. The framework effectively addresses limitations in current RAG pipelines, such as the trade-off between recall and relevance in context selection. Experimental results show that RankRAG significantly outperforms strong baselines, including ChatQA-1.5 and GPT-4 models, particularly in challenging datasets, while also exhibiting strong generalization capabilities across different domains.

---

### Improving Retrieval Augmented Language Model with Self-Reasoning

**2024-08-02**

https://arxiv.org/pdf/2407.19813v2

This paper presents a novel self-reasoning framework to enhance the reliability and traceability of Retrieval-Augmented Language Models (RALMs). The authors identify challenges in RALMs, such as irrelevant document retrieval and lack of citations, which can lead to inaccurate outputs. The proposed framework consists of three processes: a Relevance-Aware Process (RAP) to assess document relevance, an Evidence-Aware Selective Process (EAP) for selecting and citing relevant documents, and a Trajectory Analysis Process (TAP) for generating concise analyses. The framework is evaluated on four public datasets, demonstrating superior performance compared to existing state-of-the-art models while using only 2,000 training samples. The authors emphasize the framework's efficiency and robustness in improving knowledge-intensive tasks.

---