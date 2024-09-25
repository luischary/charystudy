## Diffusion Models



### IMAGEN VIDEO: HIGH DEFINITION VIDEO GENERATION WITH DIFFUSION MODELS

The paper presents Imagen Video, a text-conditional video generation system utilizing a cascade of video diffusion models. It generates high-definition videos from text prompts by employing a base video generation model alongside interleaved spatial and temporal super-resolution models. Key contributions include the effective scaling of diffusion models for high-resolution outputs, the transfer of successful techniques from text-to-image generation to video, and the introduction of progressive distillation for efficient sampling. The system demonstrates high fidelity, temporal consistency, and the ability to generate diverse videos with artistic styles and 3D object understanding.

---

### Imagine Flash: Accelerating Emu Diffusion Models with Backward Distillation

The paper presents Imagine Flash, a novel distillation framework aimed at accelerating the inference of diffusion models, specifically the Emu model, while maintaining high image quality. The authors introduce three key components: Backward Distillation, which aligns the training and inference processes to prevent data leakage; Shifted Reconstruction Loss (SRL), which adapts knowledge transfer based on the time step to enhance structural and detail rendering; and Noise Correction, a technique that improves sample quality during inference. Extensive experiments demonstrate that Imagine Flash achieves comparable performance to the original model using only 1-3 denoising steps, outperforming existing methods in both quantitative metrics and human evaluations, thus enabling efficient high-quality image generation suitable for real-time applications.

---

### RePaint: Inpainting using Denoising Diffusion Probabilistic Models

This paper presents RePaint, an innovative image inpainting method utilizing Denoising Diffusion Probabilistic Models (DDPM). Unlike traditional approaches that require training on specific mask distributions, RePaint is mask-agnostic, allowing it to generalize across various mask types, including extreme cases. The authors condition the generation process by sampling known regions from the input image during the reverse diffusion iterations, enhancing the semantic coherence of the inpainted areas. The method demonstrates superior performance compared to state-of-the-art GAN and autoregressive models in both qualitative and quantitative evaluations on datasets such as CelebA-HQ and ImageNet. Key contributions include a novel conditioning strategy that leverages the generative capabilities of pretrained DDPMs and an improved denoising process that harmonizes generated content with existing image information.

---

### Resolution-robust Large Mask Inpainting with Fourier Convolutions

This paper presents a novel image inpainting method called Large Mask Inpainting (LaMa), which addresses challenges in filling large missing areas and complex structures in high-resolution images. The authors introduce a new network architecture utilizing fast Fourier convolutions (FFCs) to achieve an image-wide receptive field, a high receptive field perceptual loss, and an aggressive training mask generation strategy. LaMa demonstrates state-of-the-art performance across various datasets, effectively generalizing to higher resolutions while maintaining lower parameter and time costs compared to existing methods. The study emphasizes the importance of receptive fields in image inpainting and shows that training with wider masks enhances performance on both narrow and wide inpainting tasks.

---

### PIXELCNN++: IMPROVING THE PIXELCNN WITH DISCRETIZED LOGISTIC MIXTURE LIKELIHOOD AND OTHER MODIFICATIONS

**2017-01-19**

https://arxiv.org/pdf/1701.05517v1

This paper presents PixelCNN++, an enhanced version of the PixelCNN generative model, which incorporates several modifications to improve performance and simplify the model structure. Key contributions include the use of a discretized logistic mixture likelihood for pixel values, conditioning on whole pixels instead of sub-pixels, and employing downsampling for multi-resolution processing. The authors also introduce short-cut connections to recover information lost during downsampling and apply dropout for regularization. Experimental results demonstrate that PixelCNN++ achieves state-of-the-art log-likelihood scores on the CIFAR-10 dataset, showcasing the effectiveness of these modifications.

---

### Denoising Diffusion Probabilistic Models

**2020-12-16**

https://arxiv.org/pdf/2006.11239v2

This paper introduces diffusion probabilistic models for high-quality image synthesis, leveraging concepts from nonequilibrium thermodynamics. The authors establish a novel connection between diffusion models and denoising score matching with Langevin dynamics, leading to a weighted variational bound for training. They demonstrate state-of-the-art results on the CIFAR10 dataset with an Inception score of 9.46 and an FID score of 3.17, as well as comparable quality to ProgressiveGAN on LSUN datasets. The paper also discusses the models' capabilities in progressive lossy compression and their potential applications in various generative modeling tasks.

---

### Improved Denoising Diffusion Probabilistic Models

**2021-02-18**

https://arxiv.org/pdf/arXiv:2102.09672v1

This paper presents enhancements to Denoising Diffusion Probabilistic Models (DDPMs), demonstrating that with modifications, DDPMs can achieve competitive log-likelihoods while maintaining high sample quality. The authors introduce a hybrid learning objective that combines variational lower-bound optimization with a simpler objective, allowing for improved log-likelihoods. They also show that learning variances in the reverse diffusion process enables sampling with significantly fewer forward passes, enhancing practical deployment. The study compares DDPMs with GANs using precision and recall metrics, revealing that DDPMs cover a larger portion of the target distribution. Furthermore, the authors explore the scalability of DDPMs with increased model capacity and training compute, indicating predictable performance improvements. Overall, the findings suggest that DDPMs are a promising avenue for generative modeling, combining good log-likelihoods, high-quality samples, and efficient sampling.

---

### Zero-Shot Text-to-Image Generation

**2021-02-26**

https://arxiv.org/pdf/arXiv:2102.12092v2

This paper presents a novel approach to text-to-image generation using a 12-billion parameter autoregressive transformer that models text and image tokens as a single data stream. The authors demonstrate that their method, trained on 250 million image-text pairs, achieves competitive performance in a zero-shot setting on the MS-COCO dataset, outperforming previous domain-specific models in human evaluations. The approach involves a two-stage training process: first, a discrete variational autoencoder (dVAE) compresses images into a grid of tokens, and second, a transformer learns to model the joint distribution of text and image tokens. The study highlights the model's ability to generalize across diverse tasks, including image-to-image translation, without requiring specific training for those tasks.

---

### DENOISING DIFFUSION IMPLICIT MODELS

**2021-05-01**

https://arxiv.org/pdf/2010.02502

This paper introduces Denoising Diffusion Implicit Models (DDIMs), a novel class of generative models that enhance the efficiency of Denoising Diffusion Probabilistic Models (DDPMs) by utilizing non-Markovian diffusion processes. DDIMs maintain the same training objective as DDPMs but allow for faster sampling, achieving speedups of 10× to 50× while preserving high sample quality. The authors demonstrate that DDIMs can perform semantically meaningful image interpolation in latent space and reconstruct observations with low error. The paper also discusses the theoretical foundations of DDIMs, their empirical advantages over DDPMs, and their relation to neural ODEs.

---

### SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models

**2021-05-18**

https://arxiv.org/pdf/arXiv:2104.14951v2

This paper introduces SRDiff, the first diffusion-based model for single image super-resolution (SISR), addressing issues such as over-smoothing, mode collapse, and large model footprint found in traditional methods. SRDiff utilizes a Markov chain to transform Gaussian noise into high-resolution images conditioned on low-resolution inputs. The model incorporates a residual prediction mechanism to enhance convergence speed and stability during training. Extensive experiments on facial and general benchmarks demonstrate that SRDiff generates diverse, high-quality super-resolution results, is easy to train with a small footprint, and allows for flexible image manipulation, including latent space interpolation and content fusion.

---

### Diffusion Models Beat GANs on Image Synthesis

**2021-06-01**

https://arxiv.org/pdf/2105.05233v4

This paper demonstrates that diffusion models can achieve superior image sample quality compared to state-of-the-art GANs in unconditional and conditional image synthesis tasks. The authors improve the architecture of diffusion models through various ablations and introduce a method called classifier guidance, which allows for a trade-off between diversity and fidelity in generated samples. They report achieving FID scores of 2.97, 4.59, and 7.72 on ImageNet at different resolutions, outperforming GANs while maintaining better distribution coverage. The study also highlights the effectiveness of combining classifier guidance with upsampling diffusion models, leading to further improvements in sample quality.

---

### Taming Transformers for High-Resolution Image Synthesis

**2021-06-23**

https://arxiv.org/pdf/2012.09841v3

This paper presents a novel approach to high-resolution image synthesis by combining convolutional neural networks (CNNs) and transformers. The authors propose a two-stage model where CNNs learn a context-rich codebook of image constituents, which are then composed using transformers to capture long-range interactions. This method allows for efficient modeling of high-resolution images, achieving state-of-the-art results in class-conditional image synthesis on ImageNet. The approach retains the advantages of transformers while leveraging the inductive biases of CNNs, enabling the generation of megapixel images and demonstrating versatility across various synthesis tasks.

---

### Cascaded Diﬀusion Models for High Fidelity Image Generation

**2021-12-17**

https://arxiv.org/pdf/2106.15282v3

This paper presents Cascaded Diffusion Models (CDM) that generate high-fidelity images on the ImageNet benchmark without relying on auxiliary classifiers. The authors introduce a pipeline of multiple diffusion models that progressively generate images at increasing resolutions, starting from a low-resolution base model and followed by super-resolution models. A key innovation is 'conditioning augmentation,' which enhances the conditioning inputs for the super-resolution models, effectively reducing compounding errors during sampling. The results show that CDM achieves superior FID scores compared to BigGAN-deep and VQ-VAE-2 across various resolutions, demonstrating the effectiveness of the proposed methods in improving sample quality in generative models.

---

### GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models

**2022-03-08**

https://arxiv.org/pdf/arXiv:2112.10741v3

This paper presents GLIDE, a text-conditional image synthesis model that utilizes guided diffusion techniques to generate high-quality, photorealistic images from textual descriptions. The authors compare two guidance strategies: CLIP guidance and classiﬁer-free guidance, finding that classiﬁer-free guidance yields superior results in terms of photorealism and caption similarity. The model, which has 3.5 billion parameters, is capable of generating diverse images and performing image inpainting, allowing for iterative editing based on natural language prompts. The authors also address safety concerns by filtering training data to mitigate the risks of misuse, such as generating disinformation or deepfakes. The code and weights for a smaller, filtered version of the model are made publicly available to support further research.

---

### StableDiffusion

**2022-04-13**

https://arxiv.org/pdf/2112.10752

---

### SDEDIT: GUIDED IMAGE SYNTHESIS AND EDITING WITH STOCHASTIC DIFFERENTIAL EQUATIONS

**2022-05-01**

https://arxiv.org/pdf/arXiv:2108.01073v2

This paper introduces SDEdit, a novel framework for guided image synthesis and editing that leverages stochastic differential equations (SDEs). Unlike existing GAN-based methods, SDEdit does not require task-specific training or inversions, allowing it to balance realism and faithfulness effectively. The method operates by adding noise to user-provided guides (e.g., stroke paintings) and iteratively denoising the images to enhance realism while maintaining fidelity to the input. SDEdit outperforms state-of-the-art GAN approaches in various tasks, achieving significant improvements in realism and overall satisfaction scores based on human evaluations. The authors demonstrate its effectiveness in stroke-based image synthesis, editing, and compositing, highlighting its versatility and efficiency in generating high-quality images from minimal user input.

---

### Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding

**2022-05-23**

https://arxiv.org/pdf/arXiv:2205.11487

The paper presents Imagen, a text-to-image diffusion model that combines large transformer language models with high-fidelity image generation techniques. Key findings include the effectiveness of large frozen language models as text encoders, the introduction of dynamic thresholding for improved image quality, and the development of a new benchmark called DrawBench for evaluating text-to-image models. Imagen achieves a state-of-the-art FID score of 7.27 on the COCO dataset, outperforming existing models like DALL-E 2 and GLIDE in both image fidelity and alignment with text prompts. The authors emphasize the importance of scaling text encoder size over image diffusion model size and highlight the need for responsible AI practices in the deployment of generative models.

---

### CLASSIFIER-FREE DIFFUSION GUIDANCE

**2022-07-26**

https://arxiv.org/pdf/2207.12598

This paper introduces classiﬁer-free guidance, a novel method for enhancing sample quality in diffusion models without relying on a separate image classiﬁer. The authors demonstrate that by jointly training conditional and unconditional diffusion models, they can achieve a trade-off between sample fidelity and diversity similar to that of classiﬁer guidance. The paper discusses the limitations of classiﬁer guidance, such as the need for additional training and potential adversarial effects on classiﬁers. Through experiments on ImageNet, the authors show that classiﬁer-free guidance can effectively improve Inception and FID scores, confirming that high-quality samples can be generated using pure generative models.

---

### RePaint: Inpainting using Denoising Diffusion Probabilistic Models

**2022-08-31**

https://arxiv.org/pdf/2201.09865v4

This paper presents RePaint, an innovative image inpainting method utilizing Denoising Diffusion Probabilistic Models (DDPM). Unlike existing approaches that are trained on specific mask distributions, RePaint is mask-agnostic, allowing it to generalize across various inpainting scenarios, including extreme masks. The authors condition the DDPM's reverse diffusion process by sampling from known regions of the image, which enhances the semantic coherence of the generated content. Experimental results demonstrate that RePaint outperforms state-of-the-art methods, particularly in producing high-quality and semantically meaningful inpainted images across diverse mask types. The paper also discusses the advantages of their resampling strategy over traditional methods, highlighting its effectiveness in harmonizing generated content with known regions.

---

### Scalable Diffusion Models with Transformers

**2023-03-02**

https://arxiv.org/pdf/2212.09748v2

This paper introduces Diffusion Transformers (DiTs), a novel class of diffusion models utilizing transformer architectures instead of the traditional U-Net backbone. The authors demonstrate that DiTs exhibit strong scalability, achieving lower Fréchet Inception Distance (FID) scores as model complexity (measured in Gflops) increases. DiTs outperform existing diffusion models on ImageNet benchmarks, achieving state-of-the-art results with an FID of 2.27 at 256×256 resolution. The study emphasizes the potential of transformers in generative modeling and provides empirical baselines for future research, highlighting the importance of architectural choices in improving sample quality.

---

### Imagic: Text-Based Real Image Editing with Diffusion Models

**2023-03-20**

https://arxiv.org/pdf/2210.09276

The paper presents Imagic, a novel method for text-based image editing that allows complex non-rigid edits on a single real image using only a text prompt. Unlike previous methods, Imagic does not require additional inputs such as image masks or multiple images. It leverages a pre-trained text-to-image diffusion model to optimize a text embedding that aligns with the input image and target text. The method consists of three main stages: optimizing the text embedding, fine-tuning the diffusion model, and interpolating between embeddings to generate edited images. The authors introduce TEdBench, a benchmark for evaluating text-based image editing methods, and demonstrate that Imagic outperforms existing techniques in user studies, showcasing its versatility and quality in producing high-fidelity edits.

---

### DIFFUSION-GAN: TRAINING GANS WITH DIFFUSION

**2023-04-24**

https://arxiv.org/pdf/2206.02262

This paper introduces Diffusion-GAN, a novel framework for training Generative Adversarial Networks (GANs) that utilizes a forward diffusion process to generate Gaussian-mixture distributed instance noise. The framework consists of an adaptive diffusion process, a timestep-dependent discriminator, and a generator. By diffusing both real and generated data, the discriminator learns to distinguish between them at various noise levels, which stabilizes training and improves data efficiency. The authors provide theoretical analysis showing that this approach offers consistent gradients for the generator, enabling better convergence to the true data distribution. Extensive experiments demonstrate that Diffusion-GAN outperforms strong GAN baselines across multiple datasets, achieving higher stability, realism, and data efficiency in image generation.

---

### Emu: Enhancing Image Generation Models Using Photogenic Needles in a Haystack

**2023-09-27**

https://arxiv.org/pdf/2309.15807

This paper introduces Emu, a quality-tuned latent diffusion model designed to enhance the aesthetic quality of images generated from text prompts. The authors propose a two-stage training approach: a knowledge learning stage involving pre-training on 1.1 billion image-text pairs, followed by a quality-tuning stage that utilizes a small dataset of a few thousand carefully selected high-quality images. The results demonstrate that Emu significantly outperforms its pre-trained counterpart and the state-of-the-art model SDXLv1.0 in visual appeal, achieving an 82.9% preference rate in human evaluations. The study emphasizes the importance of prioritizing high-quality data over quantity in fine-tuning generative models and shows that the quality-tuning method is applicable to various architectures, including pixel diffusion and masked generative transformers.

---

### PIXART-δ: FAST AND CONTROLLABLE IMAGE GENERATION WITH LATENT CONSISTENCY MODELS

**2024-01-10**

https://arxiv.org/pdf/2401.05252v1

This technical report presents PIXART-δ, a text-to-image synthesis framework that enhances the PIXART-α model by integrating the Latent Consistency Model (LCM) and ControlNet. PIXART-δ achieves high-quality image generation at 1024px resolution with significantly improved inference speed, producing images in just 0.5 seconds using only 2-4 steps, a 7× improvement over its predecessor. The model is designed for efficient training on consumer-grade GPUs and supports 8-bit inference, allowing operation within 8GB memory constraints. Additionally, the introduction of a novel ControlNet-Transformer architecture enables fine-grained control over image generation, marking a substantial advancement in text-to-image synthesis capabilities.

---

### Zero-shot Identity-Preserving Generation in Seconds

**2024-01-15**

https://arxiv.org/pdf/2401.07519

The paper introduces InstantID, a novel diffusion model-based solution for personalized image synthesis that addresses the limitations of existing methods requiring extensive fine-tuning and multiple reference images. InstantID utilizes a lightweight plug-and-play module that enables high-fidelity image personalization using only a single facial image. Key innovations include the design of IdentityNet, which integrates facial and landmark images with textual prompts to enhance identity preservation, and a decoupled cross-attention mechanism for improved image prompting. The method demonstrates superior performance compared to traditional approaches, maintaining identity consistency while allowing for stylistic flexibility, making it highly applicable in real-world scenarios.

---

### Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers

**2024-01-21**

https://arxiv.org/pdf/2401.11605v1

This paper introduces the Hourglass Diffusion Transformer (HDiT), a novel image generative model that efficiently scales to high resolutions directly in pixel space. The HDiT architecture combines the strengths of Transformers and convolutional U-Nets, achieving linear computational scaling with pixel count, which is a significant improvement over traditional diffusion models that exhibit quadratic scaling. The authors demonstrate that HDiT can generate high-quality images at resolutions up to 1024x1024 without relying on common high-resolution training techniques. The model outperforms existing diffusion models on benchmarks like FFHQ-10242 and ImageNet-2562, setting a new state-of-the-art in the field. Key contributions include a hierarchical architecture that adapts to varying resolutions, a novel loss weighting strategy, and an efficient attention mechanism that enhances detail while maintaining coherence.

---

### Make a Cheap Scaling: A Self-Cascade Diffusion Model for Higher-Resolution Adaptation

**2024-02-16**

https://arxiv.org/pdf/2402.10491

This paper introduces a self-cascade diffusion model designed to facilitate rapid adaptation of pre-trained low-resolution diffusion models for higher-resolution image and video generation. The proposed method employs a pivot-guided noise re-scheduling strategy for tuning-free adaptation and incorporates lightweight, time-aware feature upsampler modules for fine-tuning. The approach achieves over 5x training speed-up with only 0.002M additional parameters and maintains efficient inference times. Experimental results demonstrate state-of-the-art performance in both tuning-free and fine-tuning settings across various scale adaptations, addressing challenges such as composition issues and resource demands associated with high-resolution synthesis.

---

### Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

**2024-03-05**

https://arxiv.org/pdf/2403.03206v1

This paper presents advancements in rectified flow models for high-resolution text-to-image synthesis. The authors introduce improved noise sampling techniques that enhance the performance of rectified flow models over traditional diffusion formulations. They propose a novel transformer-based architecture, MM-DiT, which allows for bidirectional information flow between text and image tokens, leading to better text comprehension and image generation quality. The study includes a large-scale analysis demonstrating predictable scaling trends, where larger models correlate with lower validation loss and superior performance in various metrics and human evaluations. The authors also make their experimental data, code, and model weights publicly available.

---

### PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation

**2024-03-07**

https://arxiv.org/pdf/2403.04692v1

PixArt-Σ is a novel Diffusion Transformer model designed for high-quality 4K text-to-image generation. It builds upon the pre-trained PixArt-α model, employing a 'weak-to-strong training' approach that enhances training efficiency through the use of higher-quality data and a new attention module for token compression. Key advancements include the incorporation of 33 million high-resolution images with detailed captions and a reduction in model size to 0.6 billion parameters, significantly smaller than competitors. The model demonstrates superior image fidelity and alignment with text prompts, enabling direct generation of 4K images suitable for various applications in the film and gaming industries.

---

### Multistep Consistency Models

**2024-03-11**

https://arxiv.org/pdf/2403.06807

This paper introduces Multistep Consistency Models, which unify Consistency Models and TRACT to balance sampling speed and quality in generative modeling. The authors demonstrate that by allowing multiple sampling steps (2-8), they can achieve higher quality samples while retaining the efficiency benefits of consistency models. Notable results include achieving 1.4 FID on ImageNet64 and 2.1 FID on ImageNet128 with only 8 steps. The paper also presents a new deterministic sampler, Adjusted DDIM (aDDIM), which improves sample quality by correcting integration errors in the sampling process. Overall, the proposed models show competitive performance compared to standard diffusion models with significantly fewer sampling steps.

---

### LiteVAE: Lightweight and Efficient Variational Autoencoders for Latent Diffusion Models

**2024-05-23**

https://arxiv.org/pdf/2405.14477

This paper introduces LiteVAE, a family of lightweight autoencoders designed for latent diffusion models (LDMs) that utilize the 2D discrete wavelet transform to enhance computational efficiency without compromising output quality. The authors demonstrate that LiteVAE achieves a six-fold reduction in encoder parameters compared to standard variational autoencoders (VAEs), resulting in faster training and lower GPU memory usage. They explore various training methodologies and decoder architectures, proposing enhancements that improve training dynamics and reconstruction quality. Extensive experiments show that LiteVAE matches or exceeds the performance of existing VAEs across multiple metrics while significantly reducing computational costs, making it a promising approach for high-resolution image generation.

---