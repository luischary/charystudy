## NLP -> Finetune -> Preference



### Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**2023-05-29**

https://arxiv.org/pdf/2305.18290

This paper introduces Direct Preference Optimization (DPO), a novel algorithm for fine-tuning large unsupervised language models (LMs) to align with human preferences without the complexities of reinforcement learning from human feedback (RLHF). DPO simplifies the training process by directly optimizing the policy based on human preference data, effectively solving a classification problem rather than fitting a reward model. The authors demonstrate that DPO achieves performance comparable to or better than existing RLHF methods, particularly in tasks like sentiment modulation, summarization, and dialogue, while being more stable and computationally efficient. The study highlights the importance of controlling model behavior to ensure safe and effective AI systems.

---

### ORPO: Monolithic Preference Optimization without Reference Model

**2024-03-14**

https://arxiv.org/pdf/2403.07691v2

This paper introduces Odds Ratio Preference Optimization (ORPO), a novel algorithm for preference alignment in language models that eliminates the need for a reference model and a separate preference alignment phase. The authors emphasize the importance of supervised fine-tuning (SFT) in achieving effective preference alignment and demonstrate that a minor penalty for disfavored generation styles is sufficient for this process. ORPO is shown to outperform existing state-of-the-art models across various benchmarks, achieving significant improvements in instruction-following tasks with models ranging from 125M to 7B parameters. The authors provide empirical and theoretical evidence supporting the effectiveness of ORPO and release code and model checkpoints for further research.

---

### sDPO: Don’t Use Your Data All at Once

**2024-03-28**

https://arxiv.org/pdf/2403.19270

The paper introduces stepwise Direct Preference Optimization (sDPO), an enhancement of direct preference optimization (DPO) for aligning large language models (LLMs) with human preferences. sDPO employs a stepwise approach to utilize preference datasets, allowing for more aligned reference models at each training step. The authors demonstrate that this method improves model performance, achieving higher scores than conventional DPO and outperforming larger models with more parameters. The study emphasizes the importance of using well-aligned reference models and presents empirical results showing the effectiveness of sDPO in alignment tuning.

---

### SimPO: Simple Preference Optimization with a Reference-Free Reward

**2024-05-23**

https://arxiv.org/pdf/2405.14734

This paper introduces SimPO, a novel offline preference optimization algorithm that enhances the simplicity and effectiveness of Direct Preference Optimization (DPO). SimPO utilizes the average log probability of a sequence as an implicit reward, aligning better with model generation and eliminating the need for a reference model, thus improving computational efficiency. The authors incorporate a target reward margin into the Bradley-Terry objective to strengthen the distinction between winning and losing responses. Extensive experiments demonstrate that SimPO consistently outperforms DPO and its variants across various benchmarks, achieving significant improvements in performance without increasing response length. The study highlights the importance of length normalization and the target reward margin in optimizing model performance.

---