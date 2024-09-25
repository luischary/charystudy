## NLP -> Deploy



### Efficient Memory Management for Large Language Model Serving with PagedAttention

**2023-10-23**

https://arxiv.org/pdf/2309.06180

This paper introduces PagedAttention, a novel attention algorithm designed to optimize memory management for large language models (LLMs) during serving. The authors present vLLM, a high-throughput LLM serving system that minimizes waste in key-value (KV) cache memory and enables flexible sharing of KV cache among requests. By employing techniques inspired by operating systems' virtual memory and paging, vLLM achieves 2-4 times higher throughput compared to state-of-the-art systems like FasterTransformer and Orca, particularly benefiting from longer sequences and more complex decoding algorithms. The paper highlights the challenges of memory allocation in LLM serving and demonstrates how PagedAttention addresses these issues, leading to significant performance improvements.

---