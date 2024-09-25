## GAN



### A Style-Based Generator Architecture for Generative Adversarial Networks

**2019-03-29**

https://arxiv.org/pdf/1812.04948

This paper introduces a novel generator architecture for Generative Adversarial Networks (GANs) that leverages concepts from style transfer. The proposed style-based generator enables automatic, unsupervised separation of high-level attributes and stochastic variations in generated images, allowing for intuitive control over image synthesis at different scales. The authors present two new metrics for quantifying interpolation quality and disentanglement in latent spaces. Additionally, they introduce the Flickr-Faces-HQ (FFHQ) dataset, which offers high-quality, diverse human face images. Experimental results demonstrate that the style-based generator significantly improves image quality and disentangles latent factors compared to traditional GAN architectures.

---

### Analyzing and Improving the Image Quality of StyleGAN

**2020-03-23**

https://arxiv.org/pdf/arXiv:1912.04958v2

This paper presents an analysis of the StyleGAN architecture, identifying characteristic artifacts in generated images and proposing architectural and training modifications to enhance image quality. Key contributions include redesigning the generator normalization to eliminate blob-like artifacts, revisiting progressive growing to stabilize training without changing network topology, and introducing a path length regularizer to improve the mapping from latent codes to images. The authors demonstrate that these changes lead to significant improvements in image quality metrics, including FID and perceptual path length, and facilitate easier attribution of generated images to their respective networks. The improved model, termed StyleGAN2, sets a new state-of-the-art in unconditional image modeling.

---

### StyleGAN_2_ADA

**2020-07-10**

https://arxiv.org/pdf/2006.06676

---

### Projected GANs Converge Faster

**2021-11-01**

https://arxiv.org/pdf/2111.01007

This paper presents Projected GANs, a novel approach to training Generative Adversarial Networks (GANs) that significantly improves image quality, sample efficiency, and convergence speed. By projecting generated and real samples into a fixed, pretrained feature space, the authors address common challenges in GAN training, such as the difficulty of balancing the generator and discriminator. The proposed method utilizes multi-scale feedback from multiple discriminators and random projections to enhance the discriminator's ability to utilize deeper features. Experimental results demonstrate that Projected GANs achieve state-of-the-art Fréchet Inception Distance (FID) scores across various datasets, converging up to 40 times faster than previous methods, reducing training time from days to hours. The findings indicate that this approach not only improves performance but also reduces the need for extensive hyperparameter tuning.

---

### State-of-the-Art in the Architecture, Methods and Applications of_StyleGAN

**2022-02-28**

https://arxiv.org/pdf/2202.1402

---

### StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis

**2023-01-23**

https://arxiv.org/pdf/2301.09515

This paper introduces StyleGAN-T, a generative adversarial network (GAN) designed for efficient large-scale text-to-image synthesis. The authors address the limitations of existing GANs in this domain, focusing on enhancing model capacity, stable training on diverse datasets, and strong text alignment. StyleGAN-T outperforms previous GAN models and distilled diffusion models in terms of sample quality and speed, achieving significant improvements in zero-shot FID scores. The architecture builds on StyleGAN-XL, incorporating modifications to both the generator and discriminator, and introduces a novel guidance mechanism using CLIP for better text alignment. The model demonstrates fast inference speeds while maintaining high-quality outputs, making it competitive with state-of-the-art diffusion models, particularly at lower resolutions.

---