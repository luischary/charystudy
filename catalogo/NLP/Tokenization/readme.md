## NLP -> Tokenization



### Neural Machine Translation of Rare Words with Subword Units

**2016-06-10**

https://arxiv.org/pdf/1508.07909v5

This paper presents a novel approach to neural machine translation (NMT) that addresses the challenge of translating rare and unknown words by representing them as sequences of subword units. The authors argue that traditional NMT models, which rely on a fixed vocabulary, struggle with open-vocabulary translation, particularly for languages with complex morphological structures. They introduce byte pair encoding (BPE) as a segmentation technique that allows for a compact representation of an open vocabulary, improving translation accuracy for rare words. Empirical results demonstrate that subword models outperform back-off dictionary methods, achieving significant gains in BLEU scores for English-to-German and English-to-Russian translation tasks. The study concludes that subword representations enhance the NMT system's ability to generalize and produce unseen words, making it a simpler and more effective solution for open-vocabulary translation.

---

### SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing

**2018-08-19**

https://arxiv.org/pdf/1808.06226

This paper introduces SentencePiece, an open-source, language-independent subword tokenizer and detokenizer designed for neural text processing, particularly for neural machine translation (NMT). Unlike existing tools that require pre-tokenized input, SentencePiece can train subword models directly from raw sentences, facilitating an end-to-end system. The authors validate SentencePiece's effectiveness through experiments on English-Japanese translation, demonstrating comparable accuracy to traditional methods. Key features include lossless tokenization, efficient subword training, customizable character normalization, and a self-contained model design that ensures reproducibility. The system supports on-the-fly processing, making it suitable for integration into various NMT frameworks.

---