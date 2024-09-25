## NLP -> Finetune -> Reinforcement



### Deep Reinforcement Learning from Human Preferences

**2017-07-13**

https://arxiv.org/pdf/1706.03741v3

This paper presents a novel approach to reinforcement learning (RL) that utilizes human preferences to define complex goals, enabling RL agents to learn effectively without a predefined reward function. The authors demonstrate that by asking non-expert humans to compare pairs of trajectory segments, they can train agents to perform complex tasks in environments like Atari games and simulated robotics with minimal human feedback (less than 1% of interactions). The method significantly reduces the cost of human oversight and allows for the training of complex behaviors, such as performing backflips or driving in traffic, using only about an hour of human input. The paper highlights the scalability of learning from human feedback in deep RL systems and addresses challenges related to misalignment between human values and RL objectives.

---

### Reinforced Self-Training (ReST) for Language Modeling

**2023-08-22**

https://arxiv.org/pdf/2308.08998v2

The paper introduces Reinforced Self-Training (ReST), a novel algorithm for aligning large language models (LLMs) with human preferences using offline reinforcement learning from human feedback (RLHF). ReST operates through two main steps: 'Grow', where the model generates a dataset from its current policy, and 'Improve', where the model is fine-tuned using this dataset. This approach enhances computational efficiency by allowing data reuse and reduces the risk of reward hacking. The authors demonstrate ReST's effectiveness in machine translation tasks, showing significant improvements in translation quality over traditional supervised learning methods and other RL approaches. The study emphasizes the importance of dataset quality and the potential for ReST to be applied across various generative tasks in natural language processing.

---

### RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback

**2023-09-01**

https://arxiv.org/pdf/2309.00267

This paper presents Reinforcement Learning from AI Feedback (RLAIF), a method that uses preferences labeled by an off-the-shelf large language model (LLM) instead of human annotators to enhance the scalability of reinforcement learning from human feedback (RLHF). The authors conduct a comparative study showing that RLAIF achieves performance comparable to RLHF on the summarization task, with human evaluators preferring both methods over a supervised fine-tuned baseline. The study also explores various techniques for generating AI labels, revealing that detailed prompting and chain-of-thought reasoning improve alignment with human preferences, while few-shot in-context learning does not yield benefits. The findings suggest that RLAIF can effectively address the limitations of RLHF, providing a viable alternative that does not rely on human annotation.

---

### Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs

**2024-02-26**

https://arxiv.org/pdf/2402.14740v2

This paper critiques the use of Proximal Policy Optimization (PPO) in Reinforcement Learning from Human Feedback (RLHF) for large language models (LLMs), arguing that its complexity and computational cost are unnecessary. The authors propose simpler REINFORCE-style optimization methods, including Vanilla Policy Gradient and REINFORCE Leave-One-Out (RLOO), which they demonstrate outperform PPO and other recent RL-free methods in terms of performance and efficiency. They highlight that modeling full sequences as single actions rather than partial completions simplifies the process and enhances learning. The findings suggest that a careful adaptation of RL techniques can lead to effective and low-cost alignment of LLMs with human preferences.

---

### Teaching Large Language Models to Reason with Reinforcement Learning

**2024-03-08**

https://arxiv.org/pdf/2403.04642

This paper investigates the effectiveness of various reinforcement learning algorithms, particularly Expert Iteration (EI) and Proximal Policy Optimization (PPO), in enhancing the reasoning capabilities of large language models (LLMs) through Reinforcement Learning from Human Feedback (RLHF). The authors find that EI generally outperforms PPO across multiple metrics, with both algorithms requiring similar sample complexity to converge. They also highlight that RL fine-tuning can improve both major accuracy and pass rates simultaneously, unlike supervised fine-tuning, which often leads to overfitting. The study concludes that exploration limitations and deterministic task dynamics significantly influence the performance of RL algorithms in LLM fine-tuning, suggesting a need for more sophisticated exploration strategies in future research.

---

### The Importance of Online Data: Understanding Preference Fine-tuning via Coverage

**2024-07-16**

https://arxiv.org/pdf/arXiv:2406.01462v2

This paper investigates the differences between online reinforcement learning (RL) methods and offline contrastive methods for fine-tuning large language models (LLMs) using human preference data. The authors establish that a global coverage condition is necessary for offline methods like Direct Preference Optimization (DPO) to converge to the optimal policy, while a weaker local coverage condition suffices for online RL methods. They propose a new hybrid algorithm, Hybrid Preference Optimization (HyPO), which combines offline data for contrastive optimization with online unlabeled data for KL regularization, demonstrating improved performance over DPO in empirical evaluations. The study emphasizes the critical role of dataset coverage in the convergence properties of preference learning algorithms.

---

### BOND: Aligning LLMs with Best-of-N Distillation

**2024-07-19**

https://arxiv.org/pdf/2407.14622v1

This paper introduces BOND (Best-of-N Distillation), a novel reinforcement learning from human feedback (RLHF) algorithm designed to emulate the Best-of-N sampling strategy without incurring its computational overhead during inference. BOND utilizes a distribution matching approach, specifically minimizing the Jeffreys divergence to align the policy with the Best-of-N distribution. The authors demonstrate that BOND outperforms existing RLHF methods in various benchmarks, particularly in abstractive summarization and Gemma models. The paper also presents J-BOND, a practical implementation of BOND that incorporates an iterative process and an exponential moving average anchor for improved stability and performance.

---

### 202407 A General_Framework for Steerable Multi-Objective_Finetuning

**2024-07-22**

https://arxiv.org/pdf/2407.15762

---