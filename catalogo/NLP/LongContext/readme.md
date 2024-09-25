## NLP -> LongContext



### Efficient Classification of Long Documents Using Transformers

This paper evaluates various Transformer-based models for classifying long documents, highlighting the lack of consensus on benchmarks for fair comparison. The authors assess models like BERT, Longformer, ToBERT, and CogLTX across multiple datasets and classification tasks, revealing that simpler models often outperform more complex ones. Key findings indicate that many sophisticated models do not consistently improve performance and may require significantly more computational resources. The authors recommend future research to consider diverse datasets and simpler baselines to enhance model robustness in long document classification.

---

### Revisiting Transformer-based Models for Long Document Classification

This paper addresses the challenges of applying Transformer-based models to long document classification, which has been underexplored compared to short text sequences. The authors compare various approaches, including sparse attention and hierarchical encoding methods, to improve the efficiency and effectiveness of Transformers for long documents. They find that with proper implementation, Transformer models can outperform traditional CNN-based models on the MIMIC-III dataset. Key contributions include empirical analyses of design choices, recommendations for local and global attention strategies, and the introduction of task-adaptive pre-training to enhance model performance.

---

### Generating Long Sequences with Sparse Transformers

**2019-04-23**

https://arxiv.org/pdf/1904.10509

This paper introduces Sparse Transformers, a novel architecture that reduces the computational and memory requirements of traditional Transformers, allowing them to efficiently model long sequences. The authors propose several innovations, including sparse factorizations of the attention matrix, a restructured residual block for deeper networks, and memory-saving techniques through recomputation of attention weights. Sparse Transformers achieve state-of-the-art performance in density modeling across various domains such as text, images, and audio, demonstrating the capability to handle sequences of unprecedented lengths while maintaining global coherence and diversity in generated samples.

---

### HIERARCHICAL TRANSFORMERS FOR LONG DOCUMENT CLASSIFICATION

**2019-10-23**

https://arxiv.org/pdf/1910.10781v1

This paper presents two extensions of the BERT model, named Recurrence over BERT (RoBERT) and Transformer over BERT (ToBERT), aimed at improving the classification of long documents, such as transcripts from customer call conversations. The authors address BERT's limitation in handling long sequences by segmenting input texts and using either a recurrent layer or another transformer for classification. The proposed methods demonstrate significant improvements in customer satisfaction prediction and topic classification tasks, achieving state-of-the-art results on the Fisher topic classification task and outperforming baseline models in two of the three evaluated tasks. The study highlights the effectiveness of hierarchical representations for long document classification and suggests future work on end-to-end training for long documents.

---

### CLASSIFICATION OF LONG SEQUENTIAL DATA USING CIRCULAR DILATED CONVOLUTIONAL NEURAL NETWORKS

**2022-01-06**

https://arxiv.org/pdf/2201.02143

This paper introduces the Circular Dilated Convolutional Neural Network (CDIL-CNN), a novel architecture designed for classifying long sequential data. The authors highlight the limitations of existing methods like Temporal Convolutional Networks (TCNs) in sequence classification due to their skewed connection protocols. CDIL-CNN employs symmetric multi-scale convolutions, allowing each position in the sequence to access information from all other positions, thus enhancing classification performance. The model's architecture includes circular dilated convolutions and an ensemble learning approach that aggregates classification logits from all positions. Experimental results demonstrate that CDIL-CNN outperforms several state-of-the-art models across various datasets, showcasing its effectiveness in handling both short-term and long-term dependencies in long sequences.

---

### LongT5: Efficient Text-To-Text Transformer for Long Sequences

**2022-05-03**

https://arxiv.org/pdf/2112.07916

This paper introduces LongT5, a Transformer model designed to efficiently handle long sequences by scaling both input length and model size. LongT5 incorporates a new attention mechanism called Transient Global (TGlobal), which allows tokens to attend to both local neighborhoods and dynamically constructed global tokens without requiring additional inputs. The model also adopts a PEGASUS-style pre-training strategy focused on generating principle sentences, enhancing performance across various NLP tasks. LongT5 achieves state-of-the-art results on several summarization and question answering benchmarks, demonstrating significant improvements over previous models.

---

### An Exploration of Hierarchical Attention Transformers for Efficient Long Document Classification

**2022-10-11**

https://arxiv.org/pdf/2210.05529

This paper investigates Hierarchical Attention Transformers (HATs) as an alternative to popular sparse attention models like Longformer and Big Bird for long document classification tasks. The authors develop and release fully pre-trained HAT models that utilize segment-wise and cross-segment encoders. Their experiments demonstrate that the best HAT model outperforms Longformer models in several downstream classification tasks while consuming 10-20% less GPU memory and processing documents 40-45% faster. The study also reveals that HATs benefit from cross-segment contextualization throughout the model, highlighting the advantages of end-to-end pre-training over ad-hoc approaches. The findings suggest that HATs are a promising and efficient option for handling long documents.

---

### Efficient Long-Text Understanding with Short-Text Models

**2023-03-01**

This paper introduces SLED (SLiding-Encoder and Decoder), a method for processing long text using pretrained short-text language models. SLED partitions long documents into overlapping chunks, encodes each chunk with a short-text model, and utilizes a pretrained decoder to fuse information across chunks. The authors demonstrate that SLED achieves competitive performance on the SCROLLS benchmark, outperforming specialized models with significantly larger parameter counts and without requiring expensive pretraining. The study highlights SLED's effectiveness in tasks such as question answering and summarization, and provides insights into the locality of information in long texts.

---

### Focused Transformer: Contrastive Training for Context Scaling

**2023-07-06**

https://arxiv.org/pdf/2307.03170

This paper introduces the Focused Transformer (FOT), a novel approach designed to extend the effective context length of large language models by addressing the distraction issue in multi-document scenarios. The authors propose a contrastive training method that enhances the structure of the (key, value) space, allowing for better differentiation between relevant and irrelevant information. They demonstrate the effectiveness of FOT by fine-tuning existing models, specifically the OpenLLaMA checkpoints, resulting in LONGLLAMA models capable of managing context lengths up to 256k tokens. The study highlights significant improvements in tasks requiring long contexts, such as passkey retrieval and few-shot learning, while maintaining performance on short-context tasks. Key contributions include identifying the distraction issue, developing the FOT methodology, and showcasing its applicability to existing models without architectural modifications.

---

### COLT5: Faster Long-Range Transformers with Conditional Computation

**2023-10-24**

https://arxiv.org/pdf/2303.09752v3

COLT5 is a long-input Transformer model that enhances processing efficiency by employing conditional computation, focusing computational resources on important tokens. It builds on LONGT5, achieving superior performance with faster training and inference, particularly on long-input tasks, and sets a new state-of-the-art on the SCROLLS benchmark. COLT5 utilizes a light and heavy branch mechanism for attention and feedforward layers, optimizing the processing of long documents by routing tokens based on importance. Additionally, it incorporates multi-query attention for improved inference speed and employs the UL2 pre-training objective to facilitate in-context learning. The model demonstrates significant speed and quality improvements, especially for inputs up to 64k tokens.

---

### Cached Transformers: Improving Transformers with Differentiable Memory Cache

**2023-12-20**

https://arxiv.org/pdf/arXiv:2312.12742v1

This paper introduces the Cached Transformer model, which enhances the traditional Transformer architecture by incorporating Gated Recurrent Cached (GRC) attention. This mechanism allows the model to maintain a differentiable memory cache of tokens, enabling it to attend to both past and current tokens, thereby improving the capture of long-range dependencies. The authors demonstrate that the Cached Transformer achieves significant performance improvements across six language and vision tasks, including language modeling, machine translation, image classification, and object detection. The GRC mechanism is shown to be adaptable to various Transformer architectures and outperforms previous memory-based methods, highlighting its effectiveness in both sequential and spatial contexts.

---

### LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning

**2024-01-02**

https://arxiv.org/pdf/arXiv:2401.01325v1

This paper presents Self-Extend, a method to enhance the context window of large language models (LLMs) without fine-tuning. The authors argue that LLMs possess inherent capabilities to handle long contexts, which are limited by positional out-of-distribution (O.O.D) issues during inference. Self-Extend utilizes a FLOOR operation to map unseen relative positions to known ones, allowing LLMs to maintain coherence over longer texts. The method is implemented with minimal code changes and is evaluated on various tasks, demonstrating that it can significantly improve long context understanding and outperform fine-tuning-based methods in many cases. The study highlights the potential of LLMs to effectively manage long contexts and suggests avenues for future research.

---

### Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems

**2024-01-07**

https://arxiv.org/pdf/arXiv:2407.01370v1

This paper introduces the 'Summary of a Haystack' (SummHay) task to evaluate the performance of long-context large language models (LLMs) and Retrieval Augmented Generation (RAG) systems on summarization tasks involving large document corpora. The authors synthesize 'Haystacks' of documents with repeated insights and develop an evaluation protocol focusing on Coverage and Citation metrics. Their findings reveal that current systems, even with oracle signals of document relevance, significantly underperform compared to human benchmarks, indicating that SummHay presents an open challenge for future advancements in summarization capabilities.

---

### LongAlign: A Recipe for Long Context Alignment of Large Language Models

**2024-01-31**

https://arxiv.org/pdf/2401.18058

The paper introduces LongAlign, a methodology for enhancing large language models (LLMs) to effectively manage long contexts through instruction fine-tuning. It addresses the lack of long instruction-following datasets by constructing a diverse dataset using Self-Instruct, and proposes efficient training strategies such as packing and sorted batching to improve training efficiency. The authors also develop LongBench-Chat, a benchmark for evaluating LLMs on long context tasks. Experimental results demonstrate that LongAlign significantly improves performance on long context tasks by up to 30% while maintaining proficiency in short tasks, highlighting the importance of data quantity and diversity, as well as effective training methods.

---

### Transformers Can Achieve Length Generalization But Not Robustly

**2024-02-14**

https://arxiv.org/pdf/2402.09371v1

This paper investigates the length generalization capabilities of Transformers, specifically in the context of adding two integers. The authors demonstrate that Transformers can extrapolate to sequences 2.5 times longer than those seen during training, achieving over 98% accuracy with the right combination of position encoding and data format. Key contributions include identifying the critical role of position encoding and data formatting in length generalization, revealing the fragility of this generalization influenced by factors like weight initialization and training data order, and providing a systematic evaluation of various position encodings and data formats. Despite achieving significant results, the authors note that robust length generalization remains a challenge.

---

### Data Engineering for Scaling Language Models to 128K Context

**2024-02-15**

https://arxiv.org/pdf/2402.10171

This paper investigates methods for scaling language models' context lengths to 128K tokens through continual pretraining focused on data engineering. The authors hypothesize that the ability to utilize information at arbitrary locations is largely acquired during initial pretraining, allowing for effective extension to longer contexts with lightweight continual pretraining on a carefully curated data mixture. They demonstrate that 1-5 billion tokens are sufficient for this purpose, emphasizing the importance of domain balance and length upsampling in the data mixture. Their approach outperforms existing open-source models and narrows the performance gap with frontier models like GPT-4, particularly on the Needle-in-a-Haystack benchmark, showcasing the critical role of data engineering in enhancing long-context capabilities.

---

### In Search of Needles in a 10M Haystack: Recurrent Memory Finds What LLMs Miss

**2024-02-16**

https://arxiv.org/pdf/2402.10790

This paper introduces BABILong, a benchmark for evaluating the ability of NLP models to process long documents with distributed facts. The authors demonstrate that traditional models like GPT-4 and RAG struggle with long contexts, effectively using only a fraction of their input capacity. In contrast, they show that fine-tuning GPT-2 with recurrent memory augmentations allows it to handle inputs of up to 10 million tokens, significantly outperforming existing models. The study highlights the effectiveness of recurrent memory in filtering irrelevant information and emphasizes the need for improved context processing mechanisms in large language models.

---

### In Search of Needles in a 10M Haystack: Recurrent Memory Finds What LLMs Miss

**2024-02-16**

https://arxiv.org/pdf/2402.10790

This paper introduces BABILong, a benchmark for evaluating the ability of NLP models to process long documents with distributed facts. The authors demonstrate that traditional models like GPT-4 and RAG struggle with long contexts, effectively using only a fraction of their input capacity. In contrast, they show that fine-tuning GPT-2 with recurrent memory augmentations allows it to handle inputs of up to 10 million tokens, significantly outperforming existing models. The study highlights the effectiveness of recurrent memory in filtering irrelevant information and emphasizes the need for improved context processing mechanisms in large language models.

---

### In Search of Needles in a 11M Haystack: Recurrent Memory Finds What LLMs Miss

**2024-02-21**

https://arxiv.org/pdf/2402.10790v2

This paper introduces BABILong, a benchmark for evaluating the ability of NLP models to process long documents with distributed facts. The authors demonstrate that common generative transformer models, such as GPT-4 and RAG, struggle with sequences beyond 10^4 elements, while a fine-tuned GPT-2 augmented with recurrent memory can handle inputs up to 11 million tokens. The study highlights the limitations of current models in utilizing long contexts effectively and showcases the advantages of recurrent memory mechanisms in improving performance on long-context tasks. The findings suggest that recurrent memory models can outperform larger models in reasoning tasks involving extensive text.

---

### TransformerFAM: Feedback attention is working memory

**2024-04-14**

https://arxiv.org/pdf/2404.09173v1

The paper introduces Feedback Attention Memory (FAM), a novel Transformer architecture that incorporates a feedback loop to enhance the model's ability to process indefinitely long sequences by enabling working memory. FAM allows the Transformer to attend to its own latent representations without adding new weights, facilitating seamless integration with pre-trained models. Experiments demonstrate that TransformerFAM significantly improves performance on long-context tasks across various model sizes (1B, 8B, and 24B), showcasing its potential to empower Large Language Models (LLMs) to handle unlimited input lengths effectively.

---

### Dolphin: Long Context as a New Modality for Energy-Efficient On-Device Language Models

**2024-08-28**

https://arxiv.org/pdf/2408.15518

The paper introduces Dolphin, a novel decoder-decoder architecture designed for energy-efficient processing of long contexts in on-device language models. Dolphin utilizes a compact 0.5B parameter decoder to distill extensive contextual information into memory tokens, which are then processed by a larger 7B parameter decoder. This approach significantly reduces energy consumption and latency, achieving a 10-fold improvement in energy efficiency and a 5-fold reduction in latency compared to traditional methods, while maintaining high accuracy across various tasks. The model's multi-stage training process enhances its ability to handle long contexts, making it suitable for resource-constrained environments and paving the way for advanced language processing on edge devices.

---

### Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention

**2024-10-04**

https://arxiv.org/pdf/2404.07143v1

This paper presents Infini-attention, a novel attention mechanism designed to enable Transformer-based Large Language Models (LLMs) to process infinitely long inputs while maintaining bounded memory and computation. Infini-attention integrates compressive memory with local and long-term linear attention within a single Transformer block, allowing for efficient context management. The authors demonstrate that their approach significantly outperforms existing models on long-context language modeling, passkey retrieval, and book summarization tasks, achieving a 114x memory compression ratio and setting new state-of-the-art results. The work emphasizes the practicality of continual pre-training and fine-tuning for adapting LLMs to long contexts.

---