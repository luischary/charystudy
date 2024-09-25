## NLP -> LongContext -> Attention



### Longformer: The Long-Document Transformer

**2020-12-02**

https://arxiv.org/pdf/2004.05150v2

The Longformer introduces a modified Transformer architecture that addresses the limitations of traditional self-attention mechanisms, which scale quadratically with sequence length. By employing a linear attention mechanism that combines local windowed attention with global attention, Longformer can efficiently process long documents of thousands of tokens. The authors demonstrate that Longformer achieves state-of-the-art results on character-level language modeling tasks and outperforms RoBERTa on various long document NLP tasks, including question answering and coreference resolution. Additionally, they introduce the Longformer-Encoder-Decoder (LED) variant for generative tasks, achieving strong performance on the arXiv summarization dataset.

---

### LONGNET: Scaling Transformers to 1,000,000,000 Tokens

**2023-07-05**

https://arxiv.org/pdf/2307.02486v1

LONGNET is a novel Transformer variant designed to scale sequence lengths to over 1 billion tokens while maintaining performance on shorter sequences. The key innovation is 'dilated attention,' which reduces computational complexity from quadratic to linear, enabling efficient long-sequence modeling. LONGNET can be integrated seamlessly with existing Transformer optimizations and supports distributed training across multiple GPUs. Experimental results demonstrate its superior performance in language modeling tasks compared to traditional Transformers and other sparse attention models, highlighting its potential for processing extensive datasets like entire corpora or the Internet.

---

### LM-INFINITE: SIMPLE ON-THE-FLY LENGTH GENERALIZATION FOR LARGE LANGUAGE MODELS

**2023-08-30**

https://arxiv.org/pdf/2308.16137

This paper addresses the length generalization failures of Transformer-based Large Language Models (LLMs) when handling long sequences. The authors propose LM-Infinite, a simple and efficient solution that utilizes a Λ-shaped attention mask and a distance limit, enabling LLMs to generate fluent text and perform downstream tasks on sequences up to 32k tokens without requiring parameter updates or extensive fine-tuning. The study identifies three out-of-distribution factors contributing to length generalization failures and demonstrates that LM-Infinite maintains generation quality comparable to fine-tuned models while achieving significant computational efficiency.

---