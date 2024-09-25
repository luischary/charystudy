## NLP -> Embeddings



### Efficient Natural Language Response Suggestion for Smart Reply

**2017-05-01**

https://arxiv.org/pdf/1705.00652v1

This paper presents a computationally efficient method for natural language response suggestion using feed-forward neural networks with n-gram embeddings. The approach encodes messages into vectors optimized for high dot-product values with response pairs, enabling effective response suggestions in Gmail's Smart Reply feature. Compared to traditional sequence-to-sequence models, this method achieves similar quality with significantly reduced computational requirements and latency. The authors discuss the transition from rule-based systems to data-driven models, the challenges of end-to-end dialog systems, and the implementation of a hierarchical structure for efficient response selection. The system was evaluated using anonymized Gmail data, demonstrating improvements in suggestion quality and user engagement.

---

### Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

**2019-08-27**

https://arxiv.org/pdf/1908.10084

This paper introduces Sentence-BERT (SBERT), a modification of the BERT architecture that employs siamese and triplet networks to generate semantically meaningful sentence embeddings. SBERT significantly reduces the computational overhead for semantic similarity tasks, decreasing the time to find the most similar sentence pair from approximately 65 hours with BERT to about 5 seconds. The authors demonstrate that SBERT outperforms existing state-of-the-art sentence embedding methods on various benchmarks, including Semantic Textual Similarity (STS) tasks and transfer learning tasks. The paper also discusses the architectural choices, training strategies, and evaluation results, highlighting SBERT's efficiency and effectiveness for tasks like clustering and semantic search.

---

### REALM: Retrieval-Augmented Language Model Pre-Training

**2020-02-10**

https://arxiv.org/pdf/2002.08909v1

The paper introduces REALM, a novel framework that enhances language model pre-training by integrating a learned knowledge retriever, allowing the model to access and utilize external knowledge from large corpora like Wikipedia. This approach addresses the limitations of traditional language models that store knowledge implicitly in their parameters, making it difficult to interpret and expand their knowledge base. REALM is trained using masked language modeling and backpropagates through the retrieval step, enabling effective knowledge retrieval during inference. The authors demonstrate the effectiveness of REALM on Open-domain Question Answering tasks, achieving state-of-the-art results on multiple benchmarks, significantly outperforming previous models by 4-16% in accuracy while also providing benefits in interpretability and modularity.

---

### Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks

**2021-04-12**

https://arxiv.org/pdf/2010.08240v2

This paper introduces Augmented SBERT, a data augmentation technique aimed at enhancing the performance of bi-encoders in pairwise sentence scoring tasks. The authors highlight the limitations of bi-encoders, which require substantial training data to achieve competitive performance compared to cross-encoders. Augmented SBERT leverages a cross-encoder to label additional sentence pairs, which are then used to augment the training dataset for the bi-encoder. The study emphasizes the importance of selecting appropriate sentence pairs for labeling, as random sampling often leads to suboptimal results. The proposed method demonstrates significant performance improvements, achieving up to 6 points in in-domain tasks and up to 37 points in domain adaptation scenarios compared to traditional bi-encoder approaches. The paper also evaluates various sampling strategies for generating labeled data, concluding that BM25 sampling offers the best balance between performance and computational efficiency.

---

### Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup

**2021-06-14**

https://arxiv.org/pdf/arXiv:2101.06983v2

This paper introduces a gradient caching technique to enable large batch contrastive learning in scenarios with limited GPU memory. The authors highlight the challenges of traditional in-batch negative sampling methods, which require fitting all batch data into memory. By decoupling the backpropagation process and pre-computing gradients for a subset of the batch, the proposed method allows for almost constant memory usage regardless of batch size. Experiments demonstrate that this technique can reproduce state-of-the-art performance on a single consumer-grade GPU, significantly improving accessibility for researchers with limited resources.

---

### TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning

**2021-09-10**

https://arxiv.org/pdf/arXiv:2104.06979v3

This paper introduces TSDAE, a novel unsupervised method for learning sentence embeddings using a Transformer-based Sequential Denoising Auto-Encoder. TSDAE outperforms existing unsupervised approaches by up to 6.4 points and achieves 93.1% of the performance of supervised models. The authors highlight the limitations of previous evaluations that primarily focused on Semantic Textual Similarity (STS) tasks, arguing that these do not adequately assess generalization to domain-specific tasks. TSDAE is evaluated across four diverse datasets and demonstrates strong performance in domain adaptation and pre-training settings, significantly surpassing methods like Masked Language Model (MLM). The study emphasizes the need for broader evaluation metrics beyond STS to better understand the effectiveness of unsupervised sentence embedding methods.

---

### Large Dual Encoders Are Generalizable Retrievers

**2021-12-15**

https://arxiv.org/pdf/2112.07899

This paper presents the Generalizable T5-based dense Retrievers (GTR), a scaled-up dual encoder model that maintains a fixed-size bottleneck layer. The authors challenge the notion that dual encoders struggle with out-of-domain generalization due to their limited expressiveness. By increasing the model size while keeping the bottleneck dimension constant, they demonstrate significant improvements in retrieval performance across various tasks, particularly in out-of-domain scenarios. The GTR models outperform existing sparse and dense retrieval models on the BEIR benchmark and show remarkable data efficiency, requiring only 10% of the MS Marco supervised data to achieve optimal performance. The study emphasizes the importance of multi-stage training, combining pre-training on community question-answer pairs with fine-tuning on curated datasets to maximize model effectiveness.

---

### DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings

**2022-04-21**

https://arxiv.org/pdf/arXiv:2204.10298v1

The paper introduces DiffCSE, an unsupervised contrastive learning framework designed to learn sentence embeddings that are sensitive to differences between original and edited sentences. The authors argue that traditional contrastive learning methods often overlook the importance of certain augmentations that can alter sentence meaning. DiffCSE operationalizes equivariant contrastive learning by combining dropout-based augmentations with masked language model (MLM)-based word replacements. Experimental results demonstrate that DiffCSE outperforms the previous state-of-the-art method, SimCSE, by 2.3 absolute points on semantic textual similarity tasks. The study includes extensive ablation studies and qualitative analyses, highlighting the effectiveness of the proposed approach in producing better sentence embeddings for various NLP tasks.

---

### SimCSE: Simple Contrastive Learning of Sentence Embeddings

**2022-05-18**

https://arxiv.org/pdf/2104.08821v4

This paper introduces SimCSE, a contrastive learning framework for generating high-quality sentence embeddings. The authors propose an unsupervised method that uses dropout as a form of data augmentation to predict the input sentence itself, achieving performance comparable to supervised methods. They also present a supervised approach that utilizes natural language inference (NLI) datasets, leveraging entailment pairs as positive instances and contradiction pairs as hard negatives. SimCSE demonstrates significant improvements on standard semantic textual similarity tasks, achieving state-of-the-art results. The authors analyze the effectiveness of their approach in terms of alignment and uniformity of embeddings, showing that contrastive learning can enhance the isotropy of sentence representations.

---

### RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder

**2022-10-17**

https://arxiv.org/pdf/2205.12035v2

The paper introduces RetroMAE, a novel pre-training paradigm for retrieval-oriented language models based on a Masked Auto-Encoder (MAE). Key contributions include a unique workflow where input sentences are masked differently for the encoder and decoder, an asymmetric model structure with a full-scale BERT-like encoder and a simplified one-layer decoder, and asymmetric masking ratios that enhance the reconstruction task's difficulty. RetroMAE demonstrates significant improvements over state-of-the-art models on dense retrieval benchmarks such as BEIR and MS MARCO, showcasing its effectiveness in enhancing sentence representation capabilities for retrieval tasks.

---

### RetroMAE v2: Duplex Masked Auto-Encoder For Pre-Training Retrieval-Oriented Language Models

**2022-11-16**

https://arxiv.org/pdf/2211.08769

This paper introduces DupMAE, a novel pre-training method designed to enhance the semantic representation capabilities of retrieval-oriented language models by jointly training contextualized embeddings from both the [CLS] token and ordinary tokens. DupMAE employs two decoding tasks: reconstructing the original input sentence from the [CLS] embedding and minimizing a bag-of-words loss based on ordinary tokens' embeddings. The authors demonstrate that DupMAE achieves significant improvements in retrieval performance on MS MARCO and BEIR benchmarks, outperforming existing models while maintaining low computational costs during pre-training. The study emphasizes the importance of leveraging both [CLS] and ordinary token embeddings for better semantic representation in retrieval tasks.

---

### ANGLE-OPTIMIZED TEXT EMBEDDINGS

**2023-11-08**

https://arxiv.org/pdf/arXiv:2309.12871v6

This paper introduces AnglE, a novel angle-optimized text embedding model designed to enhance semantic textual similarity (STS) tasks by addressing the vanishing gradient problem associated with the cosine function used in existing models. AnglE operates in a complex space, optimizing both cosine similarity and angle differences to mitigate the saturation zones of the cosine function. The authors present a new long-text STS dataset collected from GitHub Issues, allowing for a comprehensive evaluation of model performance on long texts. Extensive experiments demonstrate that AnglE outperforms state-of-the-art models across various STS tasks, including short and long texts, and shows robustness in domain-specific scenarios with limited labeled data. The study also explores the effectiveness of LLM-supervised learning to enhance performance in the absence of sufficient domain-specific data.

---

### Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study

**2023-12-21**

https://arxiv.org/pdf/2304.06762v3

This study investigates the effectiveness of pretraining autoregressive language models (LMs) with retrieval mechanisms, specifically focusing on the RETRO model. The authors demonstrate that RETRO, which integrates retrieval during pretraining, significantly outperforms standard GPT models in text generation quality, factual accuracy, and performance on knowledge-intensive tasks. Key findings include reduced text degeneration, improved factual accuracy, and lower toxicity levels in generated text. The study also introduces RETRO++, a variant that enhances open-domain question answering performance. The authors argue for the future adoption of retrieval-augmented pretraining as a promising direction for developing foundation models.

---

### Improving Text Embeddings with Large Language Models

**2023-12-31**

https://arxiv.org/pdf/2401.00368v1

This paper presents a novel method for generating high-quality text embeddings using synthetic data and less than 1,000 training steps, avoiding the complex multi-stage training pipelines of existing methods. The authors leverage proprietary large language models (LLMs) to create diverse synthetic data for hundreds of thousands of text embedding tasks across nearly 100 languages. Their approach demonstrates strong performance on competitive benchmarks like BEIR and MTEB, achieving state-of-the-art results when fine-tuned with a mixture of synthetic and labeled data. The study highlights the potential of LLMs in enhancing text embeddings while streamlining the training process.

---

### Improving Text Embeddings for Smaller Language Models Using Contrastive Fine-tuning

**2024-01-08**

https://arxiv.org/pdf/arXiv:2408.00690

This paper investigates the enhancement of smaller language models' text embeddings through contrastive fine-tuning, focusing on MiniCPM, Phi-2, and Gemma. The authors demonstrate that contrastive fine-tuning on the NLI dataset significantly improves the performance of these models, with MiniCPM achieving an average performance gain of 56.33%. The study highlights the importance of text embeddings for various natural language processing tasks and presents a parameter-efficient fine-tuning method using LoRA. The results indicate that MiniCPM outperforms the other models across multiple benchmarks, showcasing its potential for resource-constrained applications. The paper also includes ablation studies examining the effects of learning rates, prompting, training data efficiency, and the impact of hard negatives in contrastive learning.

---

### Nomic Embed: Training a Reproducible Long Context Text Embedder

**2024-02-02**

https://arxiv.org/pdf/2402.01613

This technical report presents the training of nomic-embed-text-v1, the first fully reproducible, open-source long-context text embedding model with an 8192 token limit. The model outperforms OpenAI's Ada-002 and text-embedding-3-small on both short and long-context tasks. The authors provide the training code, model weights, and a curated dataset of 235 million text pairs for replication. The report discusses the architecture, training methodology, and benchmarks against existing models, highlighting the importance of auditability and compliance in AI applications.

---

### 2024 Repetition Improves Language Model Embeddings

**2024-02-23**

https://arxiv.org/pdf/2402.15449

---

### Gecko: Versatile Text Embeddings Distilled from Large Language Models

**2024-03-29**

https://arxiv.org/pdf/2403.20327v1

Gecko is a compact and versatile text embedding model that distills knowledge from large language models (LLMs) to enhance retrieval performance. The authors propose a two-step distillation process involving the generation of diverse synthetic paired data using an LLM, followed by a refinement step that retrieves and relabels positive and hard negative passages. Gecko outperforms existing models on the Massive Text Embedding Benchmark (MTEB) with fewer parameters and embedding dimensions, demonstrating strong zero-shot generalizability. The study emphasizes the importance of LLM-generated data and the effectiveness of using LLMs for identifying relevant passages, contributing to advancements in multi-task text embedding models.

---

### Scaling (Down) CLIP: A Comprehensive Analysis of Data, Architecture, and Training Strategies

**2024-04-16**

https://arxiv.org/pdf/2404.08197v2

This paper investigates the performance of the Contrastive Language-Image Pre-training (CLIP) model when scaled down under limited computational resources. The authors analyze three key dimensions: data quality and size, architecture selection, and training strategies. They demonstrate that high-quality smaller datasets can outperform larger low-quality datasets and provide guidance on selecting appropriate model architectures based on dataset size. The study compares four training strategies—SLIP, FLIP, CLIP, and CLIP+Data Augmentation—finding that CLIP+Data Augmentation can achieve comparable performance to CLIP with reduced data. The findings aim to make CLIP models more accessible and efficient for practical applications.

---

### Piccolo2: General Text Embedding with Multi-task Hybrid Loss Training

**2024-05-11**

https://arxiv.org/pdf/2405.06932v1

This paper introduces Piccolo2, a state-of-the-art text embedding model that outperforms existing models across six tasks on the CMTEB benchmark. The authors propose a multi-task hybrid loss training approach that effectively utilizes diverse textual data and labels, enhancing model performance for various downstream NLP tasks. Key innovations include scaling the embedding dimension to 1792 and employing Matryoshka Representation Learning (MRL) for flexible vector dimensions. The model's training incorporates a synthetic data generation framework and hard negative mining, resulting in significant improvements in classification, clustering, retrieval, and semantic textual similarity tasks.

---

### A Gradient Accumulation Method for Dense Retriever under Memory Constraint

**2024-06-18**

https://arxiv.org/pdf/arXiv:2406.12356v1

This paper introduces Contrastive Accumulation (CONTACCUM), a novel memory reduction method for training dense retrievers using InfoNCE loss, particularly in low-resource settings. CONTACCUM employs a dual memory bank structure to leverage previously generated query and passage representations, allowing for the use of more negative samples and achieving stable training. The authors demonstrate that CONTACCUM outperforms existing methods, including GradAccum and GradCache, across various information retrieval datasets, even surpassing high-resource scenarios. The study also provides a theoretical analysis of gradient behavior, highlighting the importance of balanced gradient norms for stable training. Key contributions include improved performance in low-resource settings, time efficiency, and insights into training instability caused by gradient norm imbalances.

---

### DEMYSTIFYING CLIP DATA

**2024-07-04**

https://arxiv.org/pdf/2309.16671v4

This paper presents MetaCLIP, a novel approach to data curation for Contrastive Language-Image Pre-training (CLIP). The authors argue that the success of CLIP is primarily due to its data quality rather than its model architecture. They reveal the curation process behind CLIP's dataset, WIT400M, and propose a method that utilizes metadata to create a balanced subset of image-text pairs from a raw data pool. MetaCLIP outperforms CLIP on various benchmarks, achieving higher accuracy in zero-shot classification tasks. The study emphasizes the importance of data curation and transparency in training data for vision-language models, making the curation code and data distribution publicly available.

---

### 2024 Matryoshka Representation Learning

**2024-08-02**

https://arxiv.org/pdf/2205.13147

---

### LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders

**2024-09-04**

https://arxiv.org/pdf/2404.05961

This paper introduces LLM2Vec, an unsupervised method to transform decoder-only large language models (LLMs) into effective text encoders. The approach consists of three steps: enabling bidirectional attention, masked next token prediction (MNTP), and unsupervised contrastive learning. The authors demonstrate that LLM2Vec significantly improves the performance of LLMs on various word- and sequence-level tasks, outperforming traditional encoder-only models. The method achieves state-of-the-art results on the Massive Text Embeddings Benchmark (MTEB) without requiring labeled data or extensive adaptation, highlighting the potential of decoder-only LLMs for text embedding tasks.

---