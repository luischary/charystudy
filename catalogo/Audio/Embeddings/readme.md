## Audio -> Embeddings



### WAVENET: A GENERATIVE MODEL FOR RAW AUDIO

**2016-09-19**

https://arxiv.org/pdf/1609.03499v2

This paper presents WaveNet, a deep generative model for producing raw audio waveforms, which operates autoregressively by modeling the joint probability of audio samples. WaveNet demonstrates state-of-the-art performance in text-to-speech (TTS) synthesis, generating highly natural-sounding speech in English and Mandarin, and can capture characteristics of multiple speakers. The model employs dilated causal convolutions to manage long-range temporal dependencies, allowing for efficient training on high-resolution audio data. Additionally, WaveNet can be adapted for music generation and phoneme recognition, showcasing its versatility across various audio applications.

---

### Conformer: Convolution-augmented Transformer for Speech Recognition

**2020-05-16**

https://arxiv.org/pdf/2005.08100

The paper presents the Conformer, a novel architecture that integrates convolutional neural networks (CNNs) and transformers for automatic speech recognition (ASR). The authors argue that while transformers excel at capturing long-range dependencies, CNNs are effective for local feature extraction. By combining both, the Conformer achieves state-of-the-art performance on the LibriSpeech benchmark, with a word error rate (WER) of 2.1%/4.3% without a language model and 1.9%/3.9% with one. The study also explores the impact of various architectural choices, such as the number of attention heads and convolution kernel sizes, demonstrating that the inclusion of convolution modules significantly enhances model accuracy while maintaining parameter efficiency.

---

### wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

**2020-10-22**

https://arxiv.org/pdf/2006.11477v3

The paper presents wav2vec 2.0, a self-supervised learning framework for speech representations that outperforms existing semi-supervised methods. It employs a multi-layer convolutional neural network to encode raw audio, masks parts of the latent representations, and utilizes a contrastive task to learn discrete speech units. The model demonstrates significant improvements in word error rates (WER) on the Librispeech dataset, achieving state-of-the-art results with minimal labeled data, including a WER of 4.8/8.2 with just 10 minutes of labeled data. The authors highlight the potential for effective speech recognition in low-resource settings and the feasibility of training models with limited labeled data, thereby making speech technology more accessible across diverse languages.

---

### HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

**2021-06-14**

https://arxiv.org/pdf/2106.07447

This paper introduces HuBERT, a self-supervised learning approach for speech representation that addresses challenges in speech data, such as the absence of a lexicon and variable lengths of sound units. HuBERT employs an offline clustering step to generate aligned target labels for a BERT-like masked prediction loss, focusing on learning both acoustic and language models from continuous inputs. The model shows significant improvements over existing methods, matching or surpassing the state-of-the-art wav2vec 2.0 performance on various benchmarks, particularly in low-resource settings. The authors highlight the importance of iterative refinement of cluster assignments and demonstrate the effectiveness of their approach through extensive experiments, achieving up to 19% relative WER reduction in challenging evaluation subsets.

---

### W2V-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training

**2021-09-13**

https://arxiv.org/pdf/2108.06209v2

The paper introduces w2v-BERT, a novel framework for self-supervised speech representation learning that integrates contrastive learning and masked language modeling (MLM). Unlike existing frameworks, w2v-BERT allows for end-to-end optimization of both tasks simultaneously, improving the discretization of speech signals and contextualized representation learning. Experiments demonstrate that w2v-BERT achieves state-of-the-art performance on the LibriSpeech benchmarks, showing a 5% to 10% relative reduction in word error rates compared to models like wav2vec 2.0 and HuBERT. Additionally, it significantly outperforms a conformer-based wav2vec 2.0 model on Google’s Voice Search dataset by over 30%. The authors emphasize the importance of the contrastive learning component in enabling effective MLM and provide insights into the architecture and training processes of w2v-BERT.

---

### High Fidelity Neural Audio Compression

**2022-10-24**

https://arxiv.org/pdf/2210.13438

This paper presents EnCodec, a state-of-the-art real-time neural audio codec that achieves high-fidelity audio compression using a streaming encoder-decoder architecture with quantized latent space. Key contributions include a novel loss balancer for stabilizing training, the use of a multiscale spectrogram adversary to reduce artifacts, and the integration of lightweight Transformer models for further compression. The authors conduct extensive subjective evaluations (MUSHRA tests) and ablation studies across various audio domains, demonstrating that EnCodec outperforms traditional codecs like Opus and EVS at low bitrates, achieving superior audio quality at 1.5 to 24 kbps for both monophonic and stereophonic audio.

---

### ZIPFORMER: A FASTER AND BETTER ENCODER FOR AUTOMATIC SPEECH RECOGNITION

**2023-06-12**

https://arxiv.org/pdf/arXiv:2310.11230v2

This paper introduces Zipformer, a novel encoder model for automatic speech recognition (ASR) that enhances performance, speed, and memory efficiency compared to existing models like Conformer. Key innovations include a U-Net-like encoder structure for temporal downsampling, a redesigned block structure that reuses attention weights, a new normalization method called BiasNorm, and new activation functions (SwooshR and SwooshL). Additionally, the authors propose ScaledAdam, an optimizer that improves convergence speed and performance. Extensive experiments on multiple datasets (LibriSpeech, Aishell-1, WenetSpeech) demonstrate that Zipformer achieves state-of-the-art results while requiring less computational resources.

---