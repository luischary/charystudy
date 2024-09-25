## NLP -> Knowledge



### Dense Passage Retrieval for Open-Domain Question Answering

**2020-09-30**

https://arxiv.org/pdf/2004.04906v3

This paper presents Dense Passage Retrieval (DPR), a method for open-domain question answering that utilizes dense representations for passage retrieval, outperforming traditional sparse models like BM25. The authors demonstrate that a dual-encoder framework, trained on a limited number of question-passage pairs, can significantly improve retrieval accuracy and end-to-end QA performance. Their experiments show that DPR achieves state-of-the-art results on multiple QA benchmarks, highlighting the effectiveness of dense representations in capturing semantic relationships and lexical variations. The study also emphasizes the importance of training strategies, such as in-batch negative sampling, and suggests that complex pretraining may not be necessary for effective retrieval.

---

### ANSWERING COMPLEX OPEN-DOMAIN QUESTIONS WITH MULTI-HOP DENSE RETRIEVAL

**2021-05-19**

https://arxiv.org/pdf/2009.12756

This paper presents a novel multi-hop dense retrieval approach for open-domain question answering, achieving state-of-the-art performance on the HotpotQA and multi-evidence FEVER datasets. Unlike previous methods, the proposed system does not rely on corpus-specific information such as hyperlinks or entity markers, making it applicable to any unstructured text corpus. The authors demonstrate that their method significantly improves the efficiency-accuracy trade-off, being ten times faster at inference while matching the best accuracy on HotpotQA. The approach utilizes a recursive framework to iteratively retrieve relevant documents based on previously retrieved information, enhancing the retrieval of complex answers that require multi-hop reasoning.

---

### Knowledge Graph Prompting for Multi-Document Question Answering

**2023-08-22**

https://arxiv.org/pdf/2308.11730

This paper introduces a Knowledge Graph Prompting (KGP) method to enhance multi-document question answering (MD-QA) using large language models (LLMs). The authors identify the challenges of MD-QA, which requires understanding logical associations across multiple documents. They propose a two-module approach: a graph construction module that creates a knowledge graph (KG) from document passages and structures, and a graph traversal module that uses an LM-guided traverser to retrieve relevant contexts. The KGP method improves retrieval quality and reduces latency by navigating the KG effectively. Experimental results demonstrate the efficacy of KGP compared to existing methods, highlighting its potential in enhancing prompt design for LLMs in MD-QA tasks.

---

### RAG VS FINE-TUNING: PIPELINES, TRADEOFFS, AND A CASE STUDY ON AGRICULTURE

**2024-01-17**

https://arxiv.org/pdf/2401.08406v2

This paper explores two methods for integrating proprietary and domain-specific data into Large Language Models (LLMs): Retrieval-Augmented Generation (RAG) and Fine-Tuning. The authors propose a comprehensive pipeline that includes data extraction, question and answer generation, and model evaluation, specifically focusing on agricultural applications. They present metrics to assess the performance of various LLMs, including Llama2-13B, GPT-3.5, and GPT-4, and demonstrate the effectiveness of their approach through an agricultural dataset. Results indicate that fine-tuning improves accuracy by over 6 percentage points, while RAG adds an additional 5 percentage points. The study highlights the potential of LLMs in providing location-specific insights for farmers and establishes a foundation for future applications in other industries.

---

### 2024 KG roadmap

**2024-01-25**

https://arxiv.org/pdf/2306.08302

---

### From Local to Global: A Graph RAG Approach to Query-Focused Summarization

**2024-04-24**

https://arxiv.org/pdf/2404.16130

This paper presents a novel Graph RAG approach that enhances query-focused summarization (QFS) over large text corpora using retrieval-augmented generation (RAG) techniques. The authors argue that traditional RAG methods struggle with global questions that require comprehensive understanding across entire datasets. The proposed method constructs a graph-based index from source documents, utilizing community detection to summarize closely-related entities and generate partial responses to user queries. The final global answer is produced by summarizing these partial responses. Experimental results demonstrate that Graph RAG significantly improves the comprehensiveness and diversity of answers compared to naive RAG methods, making it a promising solution for sensemaking tasks over extensive document collections.

---

### SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures

**2024-06-02**

https://arxiv.org/pdf/2402.03620v1

The paper introduces SELF-DISCOVER, a framework enabling large language models (LLMs) to autonomously identify and compose reasoning structures tailored to specific tasks. By leveraging a self-discovery process, LLMs select and adapt atomic reasoning modules, resulting in enhanced performance on complex reasoning benchmarks such as BigBench-Hard, grounded agent reasoning, and MATH. SELF-DISCOVER significantly outperforms traditional prompting methods like Chain of Thought (CoT) and inference-heavy approaches while requiring substantially less computational resources. The framework demonstrates universal applicability across different model families and aligns closely with human reasoning patterns, suggesting potential for improved problem-solving capabilities in LLMs.

---

### MUTUAL REASONING MAKES SMALLER LLMS STRONGER PROBLEM-SOLVERS

**2024-08-12**

https://arxiv.org/pdf/2408.06195

This paper presents rStar, a self-play mutual reasoning approach designed to enhance the reasoning capabilities of small language models (SLMs) without the need for fine-tuning or superior models. rStar employs a Monte Carlo Tree Search (MCTS) framework that integrates a rich set of human-like reasoning actions to generate high-quality reasoning trajectories. A second SLM acts as a discriminator to verify these trajectories, promoting mutual consistency among them. The authors demonstrate that rStar significantly improves reasoning accuracy across five SLMs and various reasoning tasks, achieving state-of-the-art performance, particularly on the GSM8K dataset, where accuracy increased from 12.51% to 63.91% for LLaMA2-7B. The study highlights the potential of SLMs to solve complex reasoning problems effectively when guided by structured self-play mechanisms.

---