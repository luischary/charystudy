## Multimodal



### Improved Baselines with Visual Instruction Tuning

This paper presents a systematic study on large multimodal models (LMMs) under the LLaVA framework, demonstrating that a simple fully-connected vision-language connector can achieve strong performance with minimal data. The authors introduce LLaVA-1.5, which incorporates an MLP projection and academic-task-oriented VQA data, achieving state-of-the-art results across 11 benchmarks while being data-efficient. The study explores open problems in LMMs, including scaling to higher resolutions, compositional capabilities, and model hallucination, ultimately providing a reproducible baseline for future research in multimodal instruction tuning.

---

### Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts

**2015-08-01**

https://arxiv.org/pdf/2405.11273

This paper introduces Uni-MoE, a novel unified multimodal large language model (MLLM) leveraging a Mixture of Experts (MoE) architecture to efficiently handle various modalities such as text, audio, images, and video. The authors propose a three-phase progressive training strategy that includes cross-modality alignment, training modality-specific experts, and fine-tuning with mixed multimodal instructional data. Experimental results demonstrate that Uni-MoE significantly reduces performance bias and enhances multi-expert collaboration compared to traditional dense models, achieving superior performance across a range of multimodal benchmarks. The study highlights the potential of MoE frameworks in advancing MLLMs and provides insights into effective training methodologies for multimodal tasks.

---

### Visual Instruction Tuning

**2023-04-17**

https://arxiv.org/pdf/arXiv:2304.08485

This paper presents LLaVA (Large Language and Vision Assistant), the first multimodal model that utilizes visual instruction tuning by generating language-image instruction-following data using GPT-4. The authors address the challenge of limited multimodal instruction-following data by proposing a pipeline to convert image-text pairs into instructional formats. LLaVA connects a vision encoder with a language model and demonstrates strong performance in multimodal chat and on the Science QA benchmark, achieving a state-of-the-art accuracy of 92.53%. The paper also emphasizes the importance of instruction tuning for enhancing model capabilities and provides open-source access to the generated data, model, and codebase.

---

### A Survey on Multimodal Large Language Models

**2023-06-23**

https://arxiv.org/pdf/2306.13549

This survey explores the emerging field of Multimodal Large Language Models (MLLMs), which integrate large language models with multimodal capabilities to perform tasks involving both text and visual inputs. The authors categorize MLLMs into four main techniques: Multimodal Instruction Tuning (M-IT), Multimodal In-Context Learning (M-ICL), Multimodal Chain of Thought (M-CoT), and LLM-Aided Visual Reasoning (LAVR). They discuss the formulation, key techniques, applications, and challenges faced by MLLMs, emphasizing the potential for these models to advance towards artificial general intelligence. The paper also highlights the need for improved perception capabilities, reasoning robustness, and instruction-following abilities in MLLMs, while providing a GitHub link for ongoing updates in the field.

---

### How to Bridge the Gap between Modalities: A Comprehensive Survey on Multi-modal Large Language Model

**2023-08-01**

https://arxiv.org/pdf/2311.07594v2

This survey paper reviews Multi-modal Large Language Models (MLLMs) that integrate large language models (LLMs) like GPT-4 to process multi-modal data, including text and images. The authors categorize modality alignment methods into four groups: Multi-modal Converters, Multi-modal Perceivers, Tools Assistance, and Data-Driven methods. They discuss the challenges MLLMs face, such as the semantic gap in multi-modal processing, which can lead to erroneous outputs and societal risks. The paper emphasizes the importance of selecting appropriate alignment methods to enhance performance and resource efficiency. It also outlines future research directions and the need for improved datasets and benchmarks to advance MLLMs.

---

### NExT-GPT: Any-to-Any Multimodal LLM

**2023-09-13**

https://arxiv.org/pdf/2309.05519v2

NExT-GPT is an end-to-end any-to-any multimodal Large Language Model (MM-LLM) that integrates multimodal adaptors and diffusion decoders to process and generate content across text, images, videos, and audio. Unlike existing MM-LLMs, which primarily focus on input-side multimodal understanding, NExT-GPT enables seamless input and output in arbitrary combinations of modalities. The system leverages pre-trained encoders and decoders, requiring minimal parameter adjustments (only 1%) for efficient training. Key contributions include the introduction of modality-switching instruction tuning (MosIT) and the creation of a high-quality dataset to enhance cross-modal semantic understanding and content generation. This work aims to advance the development of unified AI agents capable of human-like multimodal interactions.

---

### Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models

**2023-10-24**

https://arxiv.org/pdf/2305.15023v3

This paper introduces Mixture-of-Modality Adaptation (MMA), a novel and efficient approach for vision-language instruction tuning of large language models (LLMs). MMA utilizes lightweight adapters to connect image encoders and LLMs, enabling joint optimization without the need for extensive pre-training. The authors validate MMA by applying it to the LLaMA model, resulting in LaVIN, a multimodal LLM that demonstrates competitive performance in multimodal tasks while significantly reducing training time and storage costs. LaVIN achieves effective adaptation with only 3.8M trainable parameters and 1.4 hours of training on 8 A100 GPUs, showcasing its potential as a general-purpose chatbot and confirming the efficiency of the proposed method.

---

### VCoder: Versatile Vision Encoders for Multimodal Large Language Models

**2023-12-21**

https://arxiv.org/pdf/arXiv:2312.14233v1

This paper addresses the limitations of Multimodal Large Language Models (MLLMs) in object-level perception tasks, particularly counting and identifying objects in images. The authors propose a novel framework called Versatile vision enCoders (VCoder), which enhances MLLM performance by incorporating additional perception modalities such as segmentation and depth maps. They introduce the COCO Segmentation Text (COST) dataset, specifically designed for training and evaluating MLLMs on object perception tasks. The paper also presents new metrics for assessing object perception abilities, including count score (CS), hallucination score (HS), and depth score (DS). Experimental results demonstrate that the VCoder significantly improves object perception performance over existing MLLMs, including GPT-4V, thereby highlighting the importance of accurate perception in visual reasoning.

---

### 2024 Recent Advances in MultiModal Large Language Models

**2024-01-24**

https://arxiv.org/pdf/2401.13601

---

### Beyond Language Models: Byte Models are Digital World Simulators

**2024-02-29**

https://arxiv.org/pdf/2402.19155v1

This paper introduces bGPT, a model designed for next byte prediction to simulate the digital world, extending deep learning capabilities beyond traditional media formats like text, audio, and images to native binary data. The authors demonstrate that bGPT can effectively replicate processes such as converting ABC notation to MIDI with a low error rate and simulate CPU behavior with over 99.99% accuracy. The model's architecture incorporates a hierarchical Transformer framework to manage long byte sequences efficiently. The study highlights the potential of byte models for various applications, including cybersecurity, data compression, and algorithm simulation, while also addressing the challenges and opportunities for future research in this domain.

---

### MAGID: An Automated Pipeline for Generating Synthetic Multi-modal Datasets

**2024-03-05**

https://arxiv.org/pdf/2403.03194

The paper introduces MAGID, a framework designed to augment text-only dialogues with diverse and high-quality images, addressing the limitations of existing retrieval-based methods. MAGID employs a three-module system: an LLM-based scanner to identify suitable utterances for image augmentation, a diffusion-based image generator to create realistic images, and a quality assurance module to ensure image-text alignment and quality. The authors demonstrate that MAGID outperforms state-of-the-art baselines in both automated and human evaluations, providing a medium-sized dataset as proof of concept. The study highlights the potential of generative AI in creating high-quality multi-modal datasets while mitigating privacy and quality concerns associated with traditional methods.

---

### LAW OF VISION REPRESENTATION IN MLLMS

**2024-08-29**

https://arxiv.org/pdf/2408.16357

This paper introduces the 'Law of Vision Representation' in multimodal large language models (MLLMs), establishing a strong correlation between cross-modal alignment and correspondence in vision representation and MLLM performance. The authors quantify these factors using the Alignment and Correspondence (AC) score, demonstrating a linear relationship with model performance (R² = 95.72%). They propose an AC policy that allows for the efficient selection of optimal vision representations without the need for extensive finetuning, achieving a 99.7% reduction in computational costs. The study emphasizes the importance of understanding the underlying factors that contribute to effective vision representations in MLLMs, addressing a gap in empirical selection methods commonly used in the field.

---