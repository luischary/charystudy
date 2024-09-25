## NLP -> Finetune -> Dataset



### The Flan Collection: Designing Data and Methods for Effective Instruction Tuning

**2023-02-14**

https://arxiv.org/pdf/2301.13688v2

This paper investigates the design decisions behind instruction tuning methods, focusing on the Flan 2022 models. The authors conduct ablation studies to identify critical factors that contribute to the improved performance of Flan-T5 over previous models, achieving enhancements of 3-17% across various benchmarks. Key findings include the importance of task balancing, enrichment techniques, and the effectiveness of mixed prompt settings (zero-shot, few-shot, and chain-of-thought) in training. The authors also demonstrate that Flan-T5 converges faster and requires less fine-tuning than T5 for single downstream tasks, advocating for the use of instruction-tuned models as efficient starting points for new tasks. The Flan 2022 collection of datasets, templates, and methods is made publicly available to facilitate further research in instruction tuning.

---

### Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models

**2024-02-20**

https://arxiv.org/pdf/2402.13064v1

This paper introduces Generalized Instruction Tuning (GLAN), a novel method for instruction tuning of Large Language Models (LLMs) that generates synthetic instruction data without relying on existing datasets. GLAN constructs a pre-curated taxonomy of human knowledge and capabilities, which is then used to systematically generate diverse instruction data across various disciplines. The authors demonstrate that GLAN excels in multiple tasks, including mathematical reasoning, coding, and academic exams, without using task-specific training data. The method is scalable and customizable, allowing for easy incorporation of new fields into the taxonomy, thus enhancing the generalization capabilities of LLMs.

---

### 2024 RewardBench

**2024-03-20**

https://arxiv.org/pdf/2403.13787

---