## Diffusion Models -> SuperResolution



### Image Super-Resolution via Iterative Refinement

**2021-06-30**

https://arxiv.org/pdf/2104.07636v2

This paper presents SR3, a novel approach to image super-resolution that utilizes denoising diffusion probabilistic models for conditional image generation. SR3 refines images through a stochastic iterative denoising process, starting from Gaussian noise and progressively improving the output using a U-Net model trained at various noise levels. The authors demonstrate that SR3 outperforms state-of-the-art GAN methods in human evaluations, achieving a fool rate close to 50% on an 8× face super-resolution task, while GANs do not exceed 34%. Additionally, SR3 is effective for both faces and natural images across different magnification factors and can be used in cascaded models for high-resolution image generation. The paper emphasizes the limitations of traditional automated metrics like PSNR and SSIM in reflecting human preferences, advocating for human evaluation as a more reliable measure of image quality.

---