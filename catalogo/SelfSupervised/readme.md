## SelfSupervised



### Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language

https://arxiv.org/pdf/2202.03555

The paper presents data2vec 2.0, an efficient self-supervised learning framework that generalizes across vision, speech, and language modalities. It improves training efficiency by utilizing contextualized target representations, a fast convolutional decoder, and a multi-mask training strategy, allowing for significant reductions in pre-training time while maintaining competitive accuracy. Experiments demonstrate that data2vec 2.0 achieves comparable performance to existing models like Masked Autoencoders and wav2vec 2.0 with up to 16x faster training speeds across various tasks, including image classification, speech recognition, and natural language understanding.

---

### LARGE BATCH TRAINING OF CONVOLUTIONAL NETWORKS

**2017-09-13**

https://arxiv.org/pdf/1708.03888v3

This technical report discusses the challenges of training large convolutional neural networks (CNNs) using large batch sizes, which can lead to decreased model accuracy. The authors critique the existing method of linear learning rate scaling with warm-up, proposing a new algorithm called Layer-wise Adaptive Rate Scaling (LARS). LARS adapts the learning rate for each layer based on the ratio of the layer's weight norm to the gradient norm, improving training stability and allowing for larger batch sizes without accuracy loss. The authors successfully trained AlexNet and ResNet-50 with batch sizes up to 32K using LARS, demonstrating its effectiveness compared to traditional methods.

---

### Deep Clustering for Unsupervised Learning of Visual Features

**2019-03-18**

https://arxiv.org/pdf/1807.05520v2

This paper presents DeepCluster, a novel unsupervised learning method that integrates clustering with the end-to-end training of convolutional neural networks (CNNs). The approach iteratively clusters features using k-means and uses the resulting cluster assignments as pseudo-labels to optimize the network's parameters. DeepCluster demonstrates significant improvements over existing unsupervised methods on large-scale datasets like ImageNet and YFCC100M, achieving state-of-the-art performance across various benchmarks. The authors also explore the robustness of their method against changes in architecture and training data distribution, showing that it can effectively learn general-purpose visual features without requiring extensive domain knowledge.

---

### A Simple Framework for Contrastive Learning of Visual Representations

**2020-07-01**

https://arxiv.org/pdf/2002.05709

This paper introduces SimCLR, a straightforward framework for contrastive learning of visual representations that simplifies existing self-supervised learning methods by eliminating the need for specialized architectures and memory banks. The authors identify key components that enhance representation learning, including the importance of data augmentation composition, the benefits of a learnable nonlinear transformation, and the advantages of larger batch sizes and extended training durations. SimCLR achieves state-of-the-art performance on ImageNet, with a top-1 accuracy of 76.5% using a linear classifier on self-supervised representations, outperforming previous methods and matching supervised models. The findings emphasize the effectiveness of contrastive learning in generating high-quality visual representations.

---

### Bootstrap Your Own Latent_A New Approach to Self-Supervised Learning

**2020-10-09**

https://arxiv.org/pdf/2006.07733

---

### Unsupervised Learning of Visual Features by Contrasting Cluster Assignments

**2021-01-08**

https://arxiv.org/pdf/2006.09882v5

This paper introduces SwAV, an online algorithm for unsupervised visual representation learning that leverages contrastive methods without requiring explicit pairwise comparisons. SwAV simultaneously clusters data while ensuring consistency between cluster assignments from different augmentations of the same image. The authors propose a 'swapped' prediction mechanism and a new multi-crop data augmentation strategy, which enhances the number of views without increasing computational costs. The method achieves a top-1 accuracy of 75.3% on ImageNet with ResNet-50, surpassing previous contrastive methods and even outperforming supervised pretraining on various transfer tasks. Key contributions include a scalable online clustering loss, the multi-crop strategy, and improved performance on self-supervised benchmarks.

---

### Barlow Twins: Self-Supervised Learning via Redundancy Reduction

**2021-06-14**

https://arxiv.org/pdf/arXiv:2103.03230v3

The paper introduces Barlow Twins, a self-supervised learning (SSL) method that aims to learn embeddings invariant to input distortions while avoiding trivial constant solutions. The authors propose an objective function that minimizes redundancy in the output embeddings of two identical networks processing distorted versions of the same input, by making their cross-correlation matrix close to the identity matrix. This approach does not require large batch sizes or asymmetric network structures, making it conceptually simpler than existing methods. Barlow Twins demonstrates competitive performance on ImageNet for semi-supervised classification and transfer tasks, outperforming previous methods in low-data regimes and achieving state-of-the-art results with linear classifiers.

---

### Masked Autoencoders Are Scalable Vision Learners

**2021-12-19**

https://arxiv.org/pdf/2111.06377v3

This paper introduces Masked Autoencoders (MAE) as a scalable self-supervised learning approach for computer vision. The MAE framework involves masking a significant portion of input image patches (up to 75%) and reconstructing the missing pixels using an asymmetric encoder-decoder architecture. The encoder processes only the visible patches, while a lightweight decoder reconstructs the full image from the latent representation and mask tokens. The authors demonstrate that this method accelerates training by over 3× and achieves state-of-the-art accuracy on ImageNet-1K, outperforming previous self-supervised methods. Additionally, MAE shows strong transfer performance on various downstream tasks, indicating its potential for scaling in visual representation learning.

---

### VICREG: Variance-Invariance-Covariance Regularization for Self-Supervised Learning

**2022-05-28**

https://arxiv.org/pdf/2105.04906

The paper introduces VICReg, a self-supervised learning method that addresses the collapse problem in joint embedding architectures by employing three regularization terms: variance, invariance, and covariance. VICReg maintains the variance of each embedding dimension above a threshold, decorrelates embedding variables, and minimizes the distance between embeddings of different views of the same image. Unlike many existing methods, VICReg does not require weight sharing, large batch sizes, or normalization techniques, making it more flexible and applicable to multi-modal signals. The authors demonstrate that VICReg achieves competitive performance on various downstream tasks, stabilizes training for other methods, and provides a modular approach to self-supervised learning.

---

### 2023 A Cookbook of Self-Supervised Learning

**2023-04-24**

https://arxiv.org/pdf/2304.1221

---