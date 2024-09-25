## Multimodal -> ImageText



### Sigmoid Loss for Language Image Pre-Training

This paper introduces a novel pairwise Sigmoid loss for Language-Image Pre-training (SigLIP), which simplifies contrastive learning by eliminating the need for global normalization across batch similarities. The authors demonstrate that the sigmoid loss is more memory efficient and performs better than the traditional softmax loss, especially at smaller batch sizes. They successfully train models achieving high zero-shot accuracy on ImageNet using limited TPU resources, and find that a batch size of 32k is optimal for performance. The study also explores the effects of batch composition, the impact of negative to positive pair ratios, and robustness to data noise, ultimately contributing to more efficient and accessible language-image pre-training methods.

---

### 202406 Florence-2

**2023-10-11**

https://arxiv.org/pdf/2311.06242

---

### PALI-3 VISION LANGUAGE MODELS: SMALLER, FASTER, STRONGER

**2023-10-17**

https://arxiv.org/pdf/2310.09199

This paper introduces PaLI-3, a vision language model (VLM) that is smaller (5B parameters), faster, and stronger than existing models, achieving state-of-the-art (SOTA) performance on various benchmarks while being 10x smaller than the largest models. The authors compare classification-pretrained Vision Transformer (ViT) models with contrastively pretrained models (SigLIP), finding that the latter significantly outperforms the former in visually-situated text understanding and localization tasks. PaLI-3 sets new SOTA results on over ten diverse VLM benchmarks and demonstrates strong generalization capabilities, even achieving SOTA on video QA tasks without pretraining on video data. The model's architecture combines a contrastively pretrained image encoder with a 3B parameter UL2 language model, and the training process includes multiple stages, focusing on multimodal training and resolution increases. The findings suggest that contrastive pretraining is a more effective approach for VLMs, particularly for tasks requiring detailed visual understanding.

---

### PALI-3 VISION LANGUAGE MODELS: SMALLER, FASTER, STRONGER

**2023-10-17**

https://arxiv.org/pdf/2310.09199

This paper introduces PaLI-3, a vision language model (VLM) that is smaller (5B parameters), faster, and stronger than existing models, achieving state-of-the-art (SOTA) performance on various benchmarks while being 10x smaller than the largest models. The authors compare classification-pretrained Vision Transformer (ViT) models with contrastively pretrained models (SigLIP), finding that the latter significantly outperforms the former in visually-situated text understanding and localization tasks. PaLI-3 sets new SOTA results on over ten diverse VLM benchmarks and demonstrates strong generalization capabilities, even achieving SOTA on video QA tasks without pretraining on video data. The model's architecture combines a contrastively pretrained image encoder with a 3B parameter UL2 language model, and the training process includes multiple stages, focusing on multimodal training and resolution increases. The findings suggest that contrastive pretraining is a more effective approach for VLMs, particularly for tasks requiring detailed visual understanding.

---

### Chameleon: Mixed-Modal Early-Fusion Foundation Models

**2024-05-17**

https://arxiv.org/pdf/2405.09818

Chameleon introduces a family of early-fusion token-based mixed-modal models designed to understand and generate interleaved sequences of images and text. The authors present a stable training approach, architectural innovations, and a comprehensive evaluation across various tasks, including visual question answering, image captioning, and mixed-modal generation. Chameleon achieves state-of-the-art performance in image captioning, outperforms Llama-2 in text-only tasks, and competes with larger models like Gemini Pro and GPT-4V. The model's unified architecture allows for seamless reasoning across modalities, marking a significant advancement in multimodal foundation models.

---

### 2024 An Introduction to Vision-Language Modeling

**2024-05-27**

https://arxiv.org/pdf/2405.17247

---