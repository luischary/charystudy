## GAN -> Models



### PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION

**2018-05-01**

https://arxiv.org/pdf/1710.10196

This paper introduces a novel training methodology for Generative Adversarial Networks (GANs) that progressively grows both the generator and discriminator from low to high resolutions. This approach enhances training speed and stability, enabling the generation of high-quality images, such as 1024x1024 CELEBA images. The authors also propose a method to increase variation in generated images, achieving a record inception score of 8.80 on CIFAR10. Key contributions include implementation details to mitigate unhealthy competition between networks, a new metric for evaluating GAN outputs, and the creation of a higher-quality CELEBA dataset.

---

### Alias-Free Generative Adversarial Networks

**2021-10-18**

https://arxiv.org/pdf/2106.12423v4

This paper introduces Alias-Free Generative Adversarial Networks (StyleGAN3), addressing the issue of 'texture sticking' in traditional GANs, where fine details are improperly anchored to pixel coordinates rather than object surfaces. The authors identify aliasing as a key problem stemming from inadequate signal processing in generator networks. They propose architectural modifications that ensure continuous equivariance to translation and rotation, improving the hierarchical synthesis process. The resulting networks match the Fréchet Inception Distance (FID) of StyleGAN2 while offering enhanced internal representations, making them more suitable for applications in video and animation generation.

---