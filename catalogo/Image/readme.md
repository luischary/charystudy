## Image



### AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

**2021-05-06**

https://arxiv.org/pdf/2010.11929

This paper introduces the Vision Transformer (ViT), which applies a pure Transformer architecture directly to sequences of image patches for image classification tasks, challenging the dominance of convolutional neural networks (CNNs) in computer vision. The authors demonstrate that when pre-trained on large datasets, ViT achieves state-of-the-art performance on various benchmarks (e.g., ImageNet, CIFAR-100) while requiring significantly fewer computational resources compared to traditional CNNs. They emphasize the importance of large-scale pre-training, showing that ViT can outperform CNNs when trained on datasets with millions of images. The paper also discusses the model's architecture, training methodology, and the effects of different pre-training datasets on performance, concluding that Transformers can effectively learn visual representations without the inductive biases inherent in CNNs.

---

### VISION TRANSFORMERS NEED REGISTERS

**2024-04-12**

https://arxiv.org/pdf/arXiv:2309.16588v2

This paper identifies and characterizes artifacts in feature maps of Vision Transformer (ViT) networks, particularly high-norm tokens that arise during inference in low-informative background areas. The authors propose a solution by introducing additional learnable tokens, termed 'registers', to the input sequence, which effectively mitigates these artifacts. This approach not only enhances the interpretability of attention maps but also improves performance on dense visual prediction tasks and object discovery methods. The findings demonstrate that the proposed method generalizes across various models, including supervised and self-supervised frameworks, setting new state-of-the-art results for self-supervised visual models.

---