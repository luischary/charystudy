## Deep Learning -> Optimizers



### ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION

**2015-05-01**

https://arxiv.org/pdf/1412.6980v9

The paper introduces Adam, an efficient algorithm for first-order gradient-based optimization of stochastic objective functions. Adam combines the advantages of AdaGrad and RMSProp, providing adaptive learning rates for different parameters based on estimates of first and second moments of the gradients. The algorithm is computationally efficient, requires little memory, and is suitable for large-scale problems with noisy or sparse gradients. The authors analyze Adam's convergence properties, demonstrating a regret bound comparable to the best-known results in online convex optimization. Empirical results show that Adam outperforms other stochastic optimization methods across various machine learning models and datasets.

---

### DECOUPLED WEIGHT DECAY REGULARIZATION

**2019-04-24**

https://arxiv.org/pdf/1711.05101

This paper investigates the differences between L2 regularization and weight decay regularization in the context of adaptive gradient methods like Adam. The authors demonstrate that L2 regularization is less effective for Adam compared to SGD with momentum, leading to poorer generalization performance. They propose a modification called decoupled weight decay, which separates the weight decay from the gradient updates, allowing for independent tuning of the weight decay factor and learning rate. Empirical results show that this approach significantly improves Adam's generalization performance, making it competitive with SGD. The authors also provide insights into the optimal settings for weight decay and learning rate, and their findings have been adopted in popular deep learning frameworks.

---

### Symbolic Discovery of Optimization Algorithms

**2023-04-26**

https://arxiv.org/pdf/2302.06675v3

This paper presents a novel method for discovering optimization algorithms through program search, specifically targeting optimizers for deep neural network training. The authors introduce 'Lion' (EvoLved Sign Momentum), an efficient optimizer that tracks only momentum and utilizes the sign operation for updates, resulting in lower memory usage compared to traditional adaptive optimizers like Adam. Lion demonstrates superior performance across various tasks, including image classification and language modeling, achieving significant improvements in accuracy and computational efficiency. The study also addresses challenges in bridging the generalization gap between proxy and target tasks through program selection and simplification techniques, ultimately contributing to the field of automated machine learning and optimization.

---

### 202409 THE ADEMAMIX OPTIMIZER

**2024-05-09**

https://arxiv.org/pdf/2409.03137

---