## NLP -> Benchmarks



### Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge

**2018-03-14**

https://arxiv.org/pdf/1803.05457

The authors introduce the AI2 Reasoning Challenge (ARC), a new dataset aimed at advancing question answering (QA) research by focusing on complex reasoning tasks. The ARC dataset consists of 7,787 natural science questions, divided into a Challenge Set (2,590 difficult questions) and an Easy Set (5,197 easier questions). The Challenge Set is specifically designed to be difficult for retrieval-based and word co-occurrence algorithms, as none of the tested baseline models significantly outperform random guessing on this set. The authors also release the ARC Corpus, containing 14 million science-related sentences, and three neural baseline models to facilitate further research. The paper emphasizes the need for more sophisticated QA methods that can handle deeper text comprehension and reasoning, posing ARC as a challenge to the AI community.

---

### HellaSwag: Can a Machine Really Finish Your Sentence?

**2019-05-19**

https://arxiv.org/pdf/1905.07830

This paper introduces HellaSwag, a challenging dataset for commonsense natural language inference (NLI) that highlights the limitations of state-of-the-art models like BERT in performing commonsense reasoning. Despite achieving near-human performance on the SWAG dataset, models struggle with HellaSwag, achieving less than 50% accuracy compared to over 95% for humans. The authors utilize Adversarial Filtering to create a dataset that is easy for humans but difficult for machines by generating nonsensical endings that are misclassified by models. The findings suggest that current models primarily learn distributional biases rather than robust commonsense reasoning, indicating that the task of commonsense NLI remains unsolved. The paper advocates for evolving benchmarks that adapt to advancements in model capabilities to ensure continued progress in NLP research.

---

### MEASURING MASSIVE MULTITASK LANGUAGE UNDERSTANDING

**2021-01-12**

https://arxiv.org/pdf/2009.03300v3

This paper introduces a new benchmark for evaluating the multitask accuracy of text models across 57 diverse subjects, including STEM, humanities, and social sciences. The authors find that while the largest GPT-3 model performs better than random chance, it still falls short of expert-level accuracy across all tasks, particularly in areas requiring calculation and understanding of human values. The benchmark aims to assess models' knowledge and problem-solving abilities in zero-shot and few-shot settings, revealing significant gaps in their performance and calibration. The findings highlight the need for improvements in model understanding and alignment with human values.

---

### TruthfulQA: Measuring How Models Mimic Human Falsehoods

**2022-05-08**

https://arxiv.org/pdf/arXiv:2109.07958v2

The authors introduce TruthfulQA, a benchmark designed to evaluate the truthfulness of language models in generating answers to questions across various categories, including health, law, finance, and politics. The benchmark consists of 817 questions crafted to elicit falsehoods that some humans might also produce due to misconceptions. Testing models like GPT-3, GPT-Neo/J, GPT-2, and a T5-based model revealed that the best-performing model was truthful on only 58% of questions, compared to 94% for humans. The study highlights a concerning trend where larger models tend to be less truthful, a phenomenon termed 'inverse scaling.' The authors suggest that improving truthfulness may require methods beyond merely scaling up models, such as fine-tuning with alternative training objectives. The benchmark and evaluation methods are made publicly available to facilitate further research in this area.

---