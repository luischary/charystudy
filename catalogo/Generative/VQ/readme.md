## Generative -> VQ



### Neural Discrete Representation Learning

**2018-05-30**

https://arxiv.org/pdf/1711.00937v2

This paper introduces the Vector Quantised-Variational AutoEncoder (VQ-VAE), a generative model that learns discrete representations without supervision. The VQ-VAE model differs from traditional VAEs by outputting discrete codes from the encoder and learning a dynamic prior. It effectively addresses the 'posterior collapse' issue common in VAEs, allowing for high-quality generation of images, videos, and speech. The authors demonstrate that VQ-VAE can learn meaningful representations in various domains, including unsupervised learning of phonemes and speaker conversion, while achieving log-likelihood performance comparable to continuous latent variable models. The contributions include the introduction of VQ-VAE, validation of discrete latent variables, and applications showcasing its effectiveness in generative tasks.

---

### Taming Transformers for High-Resolution Image Synthesis

**2021-06-23**

https://arxiv.org/pdf/2012.09841v3

This paper presents a novel approach to high-resolution image synthesis by combining convolutional neural networks (CNNs) and transformers. The authors propose a two-stage model where CNNs learn a context-rich codebook of image constituents, which are then composed using transformers to capture long-range interactions. This method allows for efficient modeling of high-resolution images, achieving state-of-the-art results in class-conditional image synthesis on ImageNet. The approach retains the advantages of transformers while leveraging the inductive biases of CNNs, enabling the generation of megapixel images and demonstrating versatility across various synthesis tasks.

---

### UWA-LIP: Language Guided Image Inpainting with Defect-free VQGAN

**2022-02-10**

https://arxiv.org/pdf/2202.05009

This paper presents N ¨ UWA-LIP, a novel approach for language-guided image inpainting that addresses issues of receptive spreading and information loss in existing models. The authors introduce a defect-free VQGAN (DF-VQGAN) that utilizes relative estimation and symmetrical connections to enhance the encoding process, ensuring non-defective regions remain unchanged. Additionally, a multi-perspective sequence to sequence (MP-S2S) framework is proposed to integrate visual information from both low-level pixels and high-level tokens guided by text. The model outperforms state-of-the-art baselines on three open-domain benchmarks and includes comprehensive ablation studies to validate its components. Key contributions include the development of DF-VQGAN, MP-S2S, and the establishment of evaluation datasets for language-guided image inpainting.

---

### Autoregressive Image Generation using Residual Quantization

**2022-03-09**

https://arxiv.org/pdf/arXiv:2203.01941v2

This paper presents a two-stage framework for autoregressive (AR) modeling of high-resolution images, consisting of Residual-Quantized VAE (RQ-VAE) and RQ-Transformer. The authors argue that traditional vector quantization (VQ) methods struggle to balance code sequence length and image fidelity. RQ-VAE addresses this by using a fixed-size codebook to recursively quantize feature maps, allowing for a significant reduction in spatial resolution while maintaining high-quality reconstructions. RQ-Transformer then autoregressively predicts the quantized codes, benefiting from the reduced computational costs and improved long-range interactions. The proposed framework outperforms existing AR models in terms of image quality, computational efficiency, and sampling speed across various benchmarks, including unconditional and conditional image generation.

---

### Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors

**2022-03-24**

https://arxiv.org/pdf/2203.13131

This paper presents a novel text-to-image generation method that enhances controllability, quality, and alignment with human perception. The authors introduce a scene-based control mechanism alongside text inputs, improving the generation of complex scenes and allowing for scene and text editing. They employ domain-specific knowledge in the tokenization process, focusing on key image regions like faces and salient objects. The method utilizes an autoregressive transformer and classiﬁer-free guidance, achieving state-of-the-art results in image fidelity and human evaluation metrics, with the capability to generate high-resolution images (512x512 pixels). The contributions include improved structural consistency, overcoming out-of-distribution text prompts, and facilitating creative applications such as story illustration.

---

### VECTOR-QUANTIZED IMAGE MODELING WITH IMPROVED VQGAN

**2022-05-06**

https://arxiv.org/pdf/2110.04627

This paper presents a novel approach called Vector-quantized Image Modeling (VIM) that enhances image generation and understanding by pretraining a Transformer to predict rasterized image tokens autoregressively. The authors introduce an improved Vision-Transformer-based VQGAN (ViT-VQGAN) that optimizes architecture and codebook learning, resulting in better efficiency and reconstruction fidelity. The proposed method achieves significant improvements in image generation metrics, such as Inception Score (IS) and Fréchet Inception Distance (FID), outperforming previous models like vanilla VQGAN and iGPT. The study also demonstrates the effectiveness of VIM in unsupervised representation learning, achieving competitive linear-probe accuracy on ImageNet. Overall, the paper highlights the importance of efficient image quantization for enhancing both generative and discriminative tasks in computer vision.

---