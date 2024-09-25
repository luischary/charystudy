## Audio -> Translation



### Direct speech-to-speech translation with a sequence-to-sequence model

**2019-06-25**

https://arxiv.org/pdf/1904.06037v2

This paper presents Translatotron, an attention-based sequence-to-sequence neural network designed for direct speech-to-speech translation (S2ST) without relying on intermediate text representations. The model is trained end-to-end to map speech spectrograms from one language to another, while also enabling voice synthesis in the source speaker's voice. Experiments on Spanish-to-English datasets reveal that while Translatotron slightly underperforms compared to a baseline cascade system, it demonstrates the feasibility of direct S2ST. The study highlights the challenges of data collection and alignment in training, and emphasizes the importance of multitask training with auxiliary phoneme recognition tasks to improve performance. The authors suggest future research directions to enhance translation quality and voice transfer capabilities.

---

### Translatotron 2: High-quality direct speech-to-speech translation with voice preservation

**2022-05-17**

https://arxiv.org/pdf/2107.08661v5

Translatotron 2 is an advanced neural model for direct speech-to-speech translation (S2ST) that operates end-to-end, significantly improving translation quality and speech generation compared to its predecessor, Translatotron. It introduces a novel architecture comprising a speech encoder, linguistic decoder, acoustic synthesizer, and a unified attention module, achieving up to +15.5 BLEU score improvements. A key innovation is its method for preserving speaker voices during translation without requiring speaker segmentation, enhancing privacy and reducing misuse risks associated with voice cloning. The model demonstrates robust performance across multiple datasets, including multilingual S2ST, and employs data augmentation techniques to handle speaker turns effectively.

---

### TRANSLATOTRON 3: SPEECH TO SPEECH TRANSLATION WITH MONOLINGUAL DATA

**2024-01-16**

https://arxiv.org/pdf/2305.17547v3

This paper introduces Translatotron 3, an unsupervised model for direct speech-to-speech translation (S2ST) that utilizes monolingual speech-text datasets. The model combines techniques such as masked autoencoding, unsupervised embedding mapping, and back-translation to outperform traditional cascade systems, achieving significant improvements in BLEU scores and naturalness (MOS) in English-Spanish translation tasks. Key contributions include the first fully unsupervised end-to-end S2ST model trained on real speech data, effective retention of para-/non-linguistic characteristics, and competitive performance compared to supervised models, highlighting its potential for low-resource language support.

---