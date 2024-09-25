## NLP -> Reasoning -> Finetune



### Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

**2024-03-14**

https://arxiv.org/pdf/2403.09629

The paper introduces Quiet-STaR, an extension of the Self-Taught Reasoner (STaR), which enables language models (LMs) to generate rationales at each token to enhance their predictions. The authors address challenges such as computational costs and the initial lack of internal thought generation in LMs. By employing a tokenwise parallel sampling algorithm and an extended teacher-forcing technique, Quiet-STaR allows LMs to learn reasoning from diverse unstructured text data rather than curated datasets. The results show significant zero-shot improvements on reasoning tasks like GSM8K and CommonsenseQA, demonstrating that longer rationales lead to better performance. The study contributes to the understanding of how LMs can learn to reason in a scalable manner, paving the way for more robust language models.

---

### Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

**2024-03-18**

https://arxiv.org/pdf/2403.09629v2

This paper introduces Quiet-STaR, an extension of the Self-Taught Reasoner (STaR) framework, enabling language models (LMs) to generate rationales for predicting future text. The authors argue that reasoning is implicit in all text and propose a method for LMs to learn from diverse unstructured text data rather than curated datasets. Key contributions include a tokenwise parallel sampling algorithm for efficient rationale generation, the introduction of meta-tokens to signal thought generation, and a non-myopic loss function that improves predictions. The results demonstrate significant zero-shot performance improvements on reasoning tasks like GSM8K and CommonsenseQA, indicating that Quiet-STaR enhances the LM's reasoning capabilities without task-specific fine-tuning.

---

### Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning

**2024-06-17**

https://arxiv.org/pdf/2406.12050

This paper introduces a novel training technique called reflective augmentation (RefAug) aimed at enhancing the mathematical reasoning capabilities of language models (LMs). Unlike traditional data augmentation methods that focus on increasing the quantity of training instances, RefAug embeds reflective sections into each training instance, encouraging models to engage in deeper reasoning by considering alternative approaches and follow-up scenarios. The authors demonstrate that RefAug significantly improves performance in both standard single-round question-answering and more complex reflective reasoning tasks, achieving a notable accuracy gain of +7.2 in standard settings and substantial improvements in reflective scenarios. The method is shown to be complementary to existing augmentation techniques, providing unique advantages in fostering a deeper understanding of mathematical concepts. Extensive experiments validate the effectiveness of RefAug across various datasets and tasks, including code generation.

---