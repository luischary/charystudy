## NLP -> RNNs



### Learning to (Learn at Test Time): RNNs with Expressive Hidden States

**2024-07-05**

https://arxiv.org/pdf/2407.04620

This paper introduces Test-Time Training (TTT) layers, a new class of sequence modeling layers that enhance the expressive power of RNNs by making the hidden state a machine learning model and the update rule a step of self-supervised learning. The authors present two instantiations, TTT-Linear and TTT-MLP, which demonstrate linear complexity while outperforming strong baselines like Transformers and Mamba in perplexity across various contexts. The TTT layers can be integrated into existing architectures and optimized end-to-end. The study highlights the efficiency of TTT-Linear in terms of FLOPs and wall-clock time, especially in long contexts, and suggests promising directions for future research in optimizing TTT layers and exploring more complex models.

---