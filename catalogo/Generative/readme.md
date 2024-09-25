## Generative



### Reverse Variational Autoencoder for Visual Attribute Manipulation and Anomaly Detection

This paper introduces the Reverse Variational Autoencoder (Reverse-VAE), a generative network designed for visual attribute manipulation and anomaly detection. The model effectively combines the capabilities of a generator and an encoder, trained adversarially to minimize the Kullback-Leibler divergence between their joint distributions. The Reverse-VAE successfully manipulates visual attributes in CelebA images and demonstrates competitive anomaly detection performance on the MNIST dataset. Key contributions include a novel training approach that enhances image generation and reconstruction quality, a simple architecture that scales to high-resolution images, and the ability to manipulate visual attributes without auxiliary information. The model's scalability and efficiency make it suitable for practical applications in data augmentation and robust deep learning systems.

---

### Scaling up GANs for Text-to-Image Synthesis

This paper introduces GigaGAN, a novel GAN architecture designed for text-to-image synthesis, addressing the challenges of scaling GANs to large datasets. The authors demonstrate that simply increasing the capacity of existing GAN models like StyleGAN leads to instability. GigaGAN achieves stable training with a billion parameters, significantly faster inference times (0.13 seconds for 512px images), and the ability to generate high-resolution images (up to 16 megapixels in 3.66 seconds). It supports various latent space editing applications, such as style mixing and prompt interpolation. The results show that GigaGAN can compete with state-of-the-art diffusion and autoregressive models while maintaining the advantages of GANs, such as speed and controllability.

---

### Fader Networks: Manipulating Images by Sliding Attributes

**2018-01-28**

https://arxiv.org/pdf/1706.00409v2

This paper presents Fader Networks, a novel encoder-decoder architecture designed for image manipulation by controlling specific attributes. The model disentangles salient information and attribute values in the latent space, allowing users to generate realistic variations of images by adjusting continuous attribute values. Unlike traditional adversarial networks that operate in pixel space, Fader Networks simplify training and effectively scale to multiple attributes. The authors demonstrate that their approach preserves image naturalness while enabling significant changes in perceived attributes, outperforming existing methods in both reconstruction quality and human evaluation.

---

### Learning Latent Subspaces in Variational Autoencoders

**2018-12-01**

This paper introduces the Conditional Subspace Variational Autoencoder (CSVAE), a novel generative model that effectively learns interpretable latent representations correlated with specific labels in data. The authors address the challenge of unsupervised feature extraction by minimizing mutual information between latent variables and labels, allowing for the creation of low-dimensional subspaces for each label. The CSVAE demonstrates improved performance in attribute manipulation tasks on the Toronto Face and CelebA datasets compared to baseline models, showcasing its ability to capture intra-class variation and facilitate controllable data generation.

---

### Generating Diverse High-Fidelity Images with VQ-VAE-2

**2019-06-02**

https://arxiv.org/pdf/1906.00446v1

This paper presents an enhanced version of the Vector Quantized Variational AutoEncoder (VQ-VAE) for large-scale image generation. The authors introduce a multi-scale hierarchical approach that improves the coherence and fidelity of generated images, achieving quality comparable to state-of-the-art Generative Adversarial Networks (GANs) while avoiding issues like mode collapse. The model employs simple feed-forward networks for encoding and decoding, making it efficient for applications requiring fast processing. By leveraging a powerful autoregressive model as a prior over the latent space, the method effectively captures both global and local image features, resulting in diverse and high-resolution image outputs. The paper also discusses the advantages of likelihood-based models over GANs in terms of generalization and diversity.

---

### Latent Space Factorisation and Manipulation via Matrix Subspace Projection

**2020-08-14**

https://arxiv.org/pdf/1907.12385v3

This paper presents a novel method called Matrix Subspace Projection (MSP) for disentangling the latent space of autoencoders, enabling the manipulation of labeled attributes while preserving other characteristics. Unlike previous approaches that rely on complex adversarial training and multiple discriminators, MSP simplifies the process by directly separating attribute information from non-attribute information. The authors demonstrate the effectiveness of their method across various domains, including images and text, achieving competitive results in both human evaluations and automated metrics. Key contributions include a universal plugin for conditional generation, strong performance in learning disentangled representations of multiple attributes, and a principled strategy for loss term weighting during training.

---

### DeepFaceLab: Integrated, flexible and extensible face-swapping framework

**2021-06-29**

https://arxiv.org/pdf/2005.05535v5

DeepFaceLab (DFL) is a state-of-the-art open-source framework for face-swapping that addresses the challenges of existing deepfake methods, such as obscure workflows and poor performance. It offers a user-friendly pipeline consisting of three main phases: extraction, training, and conversion, allowing for high-quality, cinema-grade results. Key features include convenience of use, wide engineering support, extensibility for customization, and scalability for handling large datasets. DFL employs advanced techniques such as Generative Adversarial Networks (GANs) and various face segmentation methods to enhance the realism of generated faces. The framework has gained popularity in the VFX community and contributes to both entertainment and deepfake detection research.

---

### PARTI

**2022-06-22**

https://arxiv.org/pdf/2206.10789

---

### Towards Robust Blind Face Restoration with Codebook Lookup Transformer

**2022-11-01**

https://arxiv.org/pdf/2206.11253v2

This paper addresses the challenges of blind face restoration, which is complicated by various degradations such as compression and noise. The authors propose a novel approach called CodeFormer, a Transformer-based prediction network that utilizes a learned discrete codebook prior to reduce uncertainty in the restoration process. By framing face restoration as a code prediction task, CodeFormer enhances the mapping from low-quality to high-quality images while preserving facial details. The method includes a controllable feature transformation module that allows for a flexible balance between restoration quality and fidelity. Experimental results demonstrate that CodeFormer outperforms state-of-the-art methods in both synthetic and real-world datasets, showcasing its robustness against severe degradation and its effectiveness in related tasks like face inpainting and color enhancement.

---

### Consistency models

**2023-02-03**

https://arxiv.org/pdf/2303.01469

---