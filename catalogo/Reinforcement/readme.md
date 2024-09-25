## Reinforcement



### Deep Reinforcement Learning with Double Q-learning

**2015-12-08**

https://arxiv.org/pdf/1509.06461v3

This paper addresses the issue of overestimation in Q-learning, particularly in the context of the DQN algorithm applied to Atari 2600 games. The authors demonstrate that DQN often produces overly optimistic action value estimates, which can negatively impact policy performance. They propose Double DQN, an adaptation of Double Q-learning that decouples action selection from evaluation to mitigate overestimations. The empirical results show that Double DQN not only reduces overestimations but also improves performance across multiple games, achieving state-of-the-art results in the Atari domain.

---

### DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION

**2020-05-01**

https://arxiv.org/pdf/1912.01603

The paper introduces Dreamer, a reinforcement learning agent that learns long-horizon behaviors from images through latent imagination. Dreamer utilizes a learned world model to efficiently predict future states and rewards, allowing for the backpropagation of analytic gradients through imagined trajectories. The authors demonstrate that Dreamer outperforms existing model-based and model-free approaches in terms of data efficiency, computation time, and final performance across 20 challenging visual control tasks. Key contributions include the ability to learn behaviors in a compact latent space and the integration of value and action models to optimize performance beyond a fixed imagination horizon.

---

### Phasic Policy Gradient

**2020-09-09**

https://arxiv.org/pdf/2009.04416

The paper introduces Phasic Policy Gradient (PPG), a reinforcement learning framework that enhances traditional on-policy actor-critic methods by separating the training of policy and value functions into distinct phases. This approach mitigates interference between the two objectives while allowing for feature sharing. PPG operates in two alternating phases: a policy phase using Proximal Policy Optimization (PPO) and an auxiliary phase that distills features from the value function into the policy network. The authors demonstrate that PPG significantly improves sample efficiency on the Procgen Benchmark compared to PPO, highlighting the benefits of decoupling training and optimizing each objective with appropriate levels of sample reuse. The framework also allows for the incorporation of arbitrary auxiliary losses alongside reinforcement learning training.

---

### MASTERING ATARI WITH DISCRETE WORLD MODELS

**2021-05-03**

https://arxiv.org/pdf/2010.02193

This paper introduces DreamerV2, a reinforcement learning agent that achieves human-level performance on the Atari benchmark by learning behaviors solely within a separately trained world model. The authors demonstrate that DreamerV2, utilizing discrete representations and a compact latent space, surpasses top model-free agents like Rainbow and IQN while using the same computational resources. The paper highlights the effectiveness of world models in facilitating generalization and planning, and presents various modifications to the original Dreamer agent, including the use of categorical latents and KL balancing, which significantly enhance performance. DreamerV2 also shows applicability to continuous action tasks, successfully learning to control a humanoid robot from pixel inputs.

---

### Grandmaster-Level Chess Without Search

**2024-02-07**

https://arxiv.org/pdf/2402.04494v1

This paper presents a novel approach to chess AI by training a 270M parameter transformer model using supervised learning on a dataset of 10 million chess games, annotated with action-values from Stockfish 16. The model achieves a Lichess blitz Elo rating of 2895, demonstrating grandmaster-level play without relying on explicit search algorithms. The authors show that strong chess performance is contingent on sufficient model and dataset scale, and their model outperforms previous systems like AlphaZero's policy and value networks. Key contributions include the successful distillation of Stockfish's capabilities into a neural predictor and extensive ablation studies confirming the importance of model size and data diversity for robust generalization.

---

### Stop Regressing: Training Value Functions via Classification for Scalable Deep RL

**2024-03-06**

https://arxiv.org/pdf/2403.03950v1

This paper investigates the use of categorical cross-entropy loss as a replacement for mean squared error (MSE) regression in training value functions for deep reinforcement learning (RL). The authors demonstrate that this shift significantly enhances performance and scalability across various domains, including Atari games, robotic manipulation, chess, and language tasks. The study reveals that categorical cross-entropy mitigates issues like noisy targets and non-stationarity, leading to improved robustness and representation learning. The findings suggest that framing regression as classification can yield substantial benefits in value-based RL, particularly when scaling to large neural network architectures.

---

### Learning H-Infinity Locomotion Control

**2024-04-22**

https://arxiv.org/pdf/2404.14405v1

This paper presents a novel approach to enhance the robustness of quadruped robots in locomotion tasks by modeling the learning process as an adversarial interaction between an actor (the robot's policy) and a disturber (external forces). The authors introduce an H-Infinity constraint to ensure stable optimization, allowing the actor to adapt to increasingly complex disturbances while maintaining performance. The method is validated through simulations and real-world experiments on Unitree Aliengo and A1 robots, demonstrating significant improvements in robustness against various disturbances across different terrains. The proposed framework aims to inspire further research on enhancing the resilience of legged robots.

---