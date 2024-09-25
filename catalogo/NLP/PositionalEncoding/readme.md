## NLP -> PositionalEncoding



### Self-Attention with Relative Position Representations

**2018-04-12**

https://arxiv.org/pdf/arXiv:1803.02155v2

This paper presents an extension to the self-attention mechanism of the Transformer model, incorporating relative position representations to enhance translation quality in machine translation tasks. The authors demonstrate that using relative position representations leads to significant improvements in BLEU scores on the WMT 2014 English-to-German and English-to-French translation tasks, outperforming models that rely solely on absolute position encodings. The study also reveals that combining relative and absolute position representations does not yield further benefits. The proposed method is efficient and can generalize to arbitrary graph-labeled inputs, suggesting potential for future research in modeling complex relationships in data.

---

### Improve Transformer Models with Better Relative Position Embeddings

**2020-09-28**

https://arxiv.org/pdf/2009.13658

This paper discusses the limitations of existing position embeddings in Transformer models, particularly focusing on the relative position embeddings. The authors propose new techniques that enhance the interaction between query, key, and relative position embeddings in the self-attention mechanism. They introduce four novel relative position embedding methods, demonstrating that their best method outperforms previous approaches on the SQuAD1.1 dataset. Additionally, the paper explores the inductive properties of these embeddings, showing that they can generalize well to longer sequences. The proposed methods serve as effective drop-in replacements for improving the performance of large models with minimal computational cost.

---

### Improve Transformer Models with Better Relative Position Embeddings

**2020-09-28**

https://arxiv.org/pdf/2009.13658

This paper discusses the limitations of existing position embeddings in Transformer models, particularly focusing on the relative position embeddings. The authors propose new techniques that enhance the interaction between query, key, and relative position embeddings in the self-attention mechanism. They introduce four novel relative position embedding methods, demonstrating that their best method outperforms previous approaches on the SQuAD1.1 dataset. Additionally, the paper explores the inductive properties of these embeddings, showing that they can generalize well to longer sequences. The proposed methods serve as effective drop-in replacements for improving the performance of large models with minimal computational cost.

---

### TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION

**2022-04-22**

https://arxiv.org/pdf/2108.12409

This paper introduces Attention with Linear Biases (ALiBi), a novel position representation method for transformer models that enhances their ability to extrapolate to longer input sequences during inference. The authors demonstrate that traditional sinusoidal position embeddings limit extrapolation capabilities, while ALiBi allows models trained on shorter sequences to perform effectively on longer sequences without additional computational costs. They show that a 1.3 billion parameter model using ALiBi can achieve comparable perplexity to sinusoidal models trained on longer sequences, while being faster and more memory-efficient. The results indicate that ALiBi not only improves performance on standard benchmarks but also mitigates the early token curse, making it a valuable alternative for language modeling tasks.

---

### RoFormer: Enhanced Transformer with Rotary Position Embedding

**2022-08-10**

https://arxiv.org/pdf/2104.09864v4

This paper introduces Rotary Position Embedding (RoPE), a novel method for integrating positional information into transformer-based language models. RoPE utilizes a rotation matrix to encode absolute positions while incorporating relative position dependencies in self-attention mechanisms. The authors argue that RoPE offers advantages such as flexibility in sequence length, decaying inter-token dependencies with increasing distances, and compatibility with linear self-attention. Experimental results demonstrate that RoFormer, the enhanced transformer model utilizing RoPE, consistently outperforms existing alternatives on various long text classification benchmarks. The paper also provides theoretical analyses to support the effectiveness of RoPE.

---

### The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction

**2023-12-21**

https://arxiv.org/pdf/2312.13558

This paper introduces LAyer-SElective Rank reduction (LASER), a method that enhances the performance of Transformer-based Large Language Models (LLMs) by selectively removing higher-order components of weight matrices after training. The authors demonstrate that significant performance improvements can be achieved without additional parameters or data, particularly in later layers of the model. Extensive experiments across various models and datasets reveal that LASER not only boosts accuracy on reasoning tasks but also increases robustness to paraphrases. The findings suggest that higher-order components often introduce noise, leading to incorrect responses, and that their removal acts as a denoising technique, making less frequent but correct information more accessible. The study contributes to understanding model compression and the internal representations of LLMs.

---

### LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

**2024-02-21**

https://arxiv.org/pdf/2402.13753

This paper presents LongRoPE, a novel method that extends the context window of large language models (LLMs) to 2048k tokens, significantly surpassing the previous limit of around 128k tokens. The authors address challenges such as high fine-tuning costs and the introduction of catastrophic values due to untrained token positions. LongRoPE employs three key innovations: (1) it utilizes an efficient search to exploit non-uniformities in positional interpolation, (2) it implements a progressive extension strategy that fine-tunes a 256k length LLM before extending to 2048k, and (3) it readjusts positional embeddings to recover performance on shorter context windows. Extensive experiments demonstrate that LongRoPE maintains performance across various tasks and can be applied to any LLM using RoPE embeddings.

---

### Resonance RoPE: Improving Context Length Generalization of Large Language Models

**2024-02-29**

https://arxiv.org/pdf/2403.00071

This paper introduces RESONANCE ROPE, a novel technique aimed at improving the performance of Large Language Models (LLMs) in train-short-test-long (TSTL) scenarios, particularly those utilizing Rotary Position Embedding (RoPE). The authors identify that models pre-trained on shorter sequences struggle with out-of-distribution (OOD) token positions in longer sequences. RESONANCE ROPE refines the interpolation of RoPE features for OOD positions, significantly enhancing model performance without additional computational costs. Additionally, the paper presents POSGEN, a synthetic benchmark designed for fine-grained analysis of TSTL performance, allowing researchers to isolate difficulties in token generation from challenges in recognizing new token positions. Experimental results demonstrate that RESONANCE ROPE improves OOD position recognition and overall performance in both synthetic tasks and real-world applications, surpassing existing methods.

---