## NLP -> Datasets -> Syntetic



### Best Practices and Lessons Learned on Synthetic Data for Language Models

**2024-04-10**

https://arxiv.org/pdf/2404.07503v1

This paper discusses the role of synthetic data in enhancing the training and evaluation of language models, addressing challenges such as data scarcity, privacy concerns, and high costs. It outlines various applications of synthetic data across domains like reasoning, tool usage, and multilingual tasks, emphasizing its potential to improve model performance and mitigate biases. The authors highlight the importance of ensuring the factuality, fidelity, and unbiasedness of synthetic data, while also identifying challenges such as the risk of misinformation and difficulties in evaluation. The paper concludes with directions for future research, including scaling synthetic data generation and improving its quality and diversity.

---

### MAGPIE: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing

**2024-06-12**

https://arxiv.org/pdf/2406.08464

The paper introduces MAGPIE, a self-synthesis method for generating large-scale alignment data for fine-tuning large language models (LLMs) like Llama-3. The authors highlight the challenges of existing data creation methods, which are often limited by human labor and predefined prompts. MAGPIE leverages the auto-regressive nature of aligned LLMs to autonomously generate high-quality instruction data without the need for seed questions or extensive prompt engineering. The authors generated 4 million instructions and responses, ultimately selecting 300K high-quality instances. Experimental results demonstrate that models fine-tuned with MAGPIE data perform comparably to those fine-tuned with significantly larger datasets, showcasing the effectiveness and scalability of the MAGPIE approach in enhancing LLM alignment.

---