## Audio -> TTS



### TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS

**2017-04-06**

https://arxiv.org/pdf/1703.10135v2

The paper presents Tacotron, an end-to-end generative text-to-speech (TTS) model that synthesizes speech directly from characters, trained on <text, audio> pairs. Tacotron simplifies the traditional multi-stage TTS pipeline by eliminating the need for extensive feature engineering and independent training of components. It employs a sequence-to-sequence framework with attention mechanisms, achieving a mean opinion score of 3.82 in naturalness, outperforming existing parametric systems. Key innovations include the CBHG module for robust representation extraction, a pre-net for improved convergence, and a post-processing net for enhanced waveform synthesis. The model demonstrates faster inference compared to sample-level autoregressive methods and can be trained from scratch without requiring phoneme-level alignment.

---

### NATURAL TTS SYNTHESIS BY CONDITIONING WAVENET ON MEL SPECTROGRAM PREDICTIONS

**2018-02-16**

https://arxiv.org/pdf/1712.05884v2

This paper presents Tacotron 2, a neural network architecture for text-to-speech synthesis that generates mel spectrograms from text and uses a modified WaveNet vocoder to synthesize audio waveforms. The system achieves a mean opinion score (MOS) of 4.53, comparable to 4.58 for professional recordings. The authors validate their design through ablation studies, demonstrating that using mel spectrograms as input significantly reduces the complexity of the WaveNet architecture while maintaining high audio quality. The paper highlights the advantages of an end-to-end neural approach, eliminating the need for complex feature engineering and achieving state-of-the-art sound quality close to natural human speech.

---

### EFFICIENTLY TRAINABLE TEXT-TO-SPEECH SYSTEM BASED ON DEEP CONVOLUTIONAL NETWORKS WITH GUIDED ATTENTION

**2018-04-15**

https://arxiv.org/pdf/1710.08969v2

This paper presents a novel text-to-speech (TTS) system called Deep Convolutional TTS (DCTTS), which utilizes deep convolutional neural networks (CNN) instead of recurrent neural networks (RNN) to improve training efficiency. The authors argue that CNNs can be trained significantly faster due to their parallelizability, achieving acceptable speech quality in just 15 hours on a standard gaming PC with two GPUs. The paper also introduces a 'guided attention' mechanism to enhance the training of the attention module, leading to improved focus on relevant text during synthesis. The results indicate that DCTTS can be trained rapidly while maintaining competitive performance compared to existing RNN-based systems like Tacotron.

---

### Neural Speech Synthesis with Transformer Network

**2019-01-30**

https://arxiv.org/pdf/1809.08895v3

This paper introduces a novel end-to-end text-to-speech (TTS) model that integrates the Transformer architecture with the Tacotron2 framework. The authors replace recurrent neural networks (RNNs) with a multi-head attention mechanism, enhancing training efficiency and effectively modeling long-range dependencies. The proposed Transformer TTS network generates mel spectrograms from phoneme sequences, which are then synthesized into audio using a WaveNet vocoder. Experimental results demonstrate that the new model is approximately 4.25 times faster in training compared to Tacotron2 and achieves state-of-the-art performance, closely matching human quality in audio synthesis. The study highlights the advantages of parallel processing and improved prosody in synthesized speech.

---

### HIGH FIDELITY SPEECH SYNTHESIS WITH ADVERSARIAL NETWORKS

**2019-09-26**

https://arxiv.org/pdf/1909.11646v2

This paper introduces GAN-TTS, a Generative Adversarial Network for Text-to-Speech (TTS) that generates high-fidelity speech audio. The architecture features a conditional feed-forward generator and an ensemble of discriminators that evaluate audio realism and alignment with linguistic features. The authors propose new quantitative metrics for evaluating speech generation, including Fréchet DeepSpeech Distance and Kernel DeepSpeech Distance, which correlate well with human evaluations. GAN-TTS achieves a Mean Opinion Score (MOS) of 4.2, comparable to the state-of-the-art WaveNet, while offering efficient parallelization in audio generation.

---

### FastSpeech: Fast, Robust and Controllable Text to Speech

**2019-11-20**

https://arxiv.org/pdf/1905.09263v5

FastSpeech is a novel text-to-speech (TTS) model that addresses the limitations of traditional autoregressive TTS systems, such as slow inference speed, robustness issues, and lack of controllability. By employing a feed-forward network based on Transformer architecture, FastSpeech generates mel-spectrograms in parallel, achieving a 270x speedup in mel-spectrogram generation and a 38x speedup in end-to-end speech synthesis compared to autoregressive models. The model incorporates a phoneme duration predictor and a length regulator, enabling it to avoid word skipping and repeating while allowing for smooth adjustments in voice speed and prosody. Experiments on the LJSpeech dataset demonstrate that FastSpeech matches the speech quality of autoregressive models while significantly improving synthesis speed and robustness.

---

### HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

**2020-10-23**

https://arxiv.org/pdf/2010.05646v2

HiFi-GAN introduces a novel approach to speech synthesis using generative adversarial networks (GANs) that achieves both high fidelity and efficient audio generation. The authors emphasize the importance of modeling periodic patterns in audio signals, leading to improved sample quality. HiFi-GAN outperforms existing models like WaveNet and WaveGlow in terms of mean opinion score (MOS) and synthesis speed, generating high-quality audio 167.9 times faster than real-time on a single GPU. The model demonstrates versatility in mel-spectrogram inversion for unseen speakers and end-to-end speech synthesis. A compact version of HiFi-GAN also shows competitive quality with significantly reduced computational requirements. The study provides open-source implementation for reproducibility and future research.

---

### END-TO-END ADVERSARIAL TEXT-TO-SPEECH

**2021-05-01**

https://arxiv.org/pdf/2006.03575

This paper presents EATS (End-to-end Adversarial Text-to-Speech), a novel approach to text-to-speech synthesis that operates directly on character or phoneme input sequences to generate raw audio outputs in an end-to-end manner. The authors propose a feed-forward generator that includes an aligner and a decoder, utilizing adversarial feedback and prediction losses to produce high-fidelity audio. Key contributions include a differentiable aligner architecture that predicts token durations, the use of dynamic time warping to capture temporal variations, and achieving a mean opinion score of 4.083, comparable to state-of-the-art models that require more complex training setups. The model demonstrates efficiency in both training and inference, generating speech significantly faster than real-time.

---

### FASTSPEECH 2: FAST AND HIGH-QUALITY END-TO-END TEXT TO SPEECH

**2022-08-08**

https://arxiv.org/pdf/2006.04558v8

FastSpeech 2 is an advanced text-to-speech (TTS) model that addresses limitations of its predecessor, FastSpeech, by simplifying the training pipeline and improving voice quality. Key innovations include training directly with ground-truth mel-spectrograms, introducing additional variance information (pitch, energy, and accurate duration) as conditional inputs, and developing FastSpeech 2s, which generates speech waveforms directly from text in a fully end-to-end manner. Experimental results demonstrate that FastSpeech 2 achieves a 3x training speed-up and outperforms FastSpeech and autoregressive models in voice quality, while FastSpeech 2s offers even faster inference speed.

---

### NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers

**2023-05-30**

https://arxiv.org/pdf/2304.09116v3

NaturalSpeech 2 is a text-to-speech (TTS) system developed to enhance speech synthesis quality and diversity by leveraging latent diffusion models and a neural audio codec. The system addresses limitations of previous autoregressive models, such as unstable prosody and poor voice quality, by utilizing continuous latent vectors instead of discrete tokens. It incorporates a speech prompting mechanism for in-context learning, enabling strong zero-shot capabilities for diverse speaker identities and styles, including singing. Evaluations show that NaturalSpeech 2 significantly outperforms prior TTS systems in terms of prosody similarity, robustness, and voice quality, achieving human-level synthesis quality and novel zero-shot singing synthesis with only a speech prompt.

---

### E3 TTS: EASY END-TO-END DIFFUSION-BASED TEXT TO SPEECH

**2023-11-02**

https://arxiv.org/pdf/arXiv:2311.00945v1

The paper presents E3 TTS, an innovative end-to-end text-to-speech model that utilizes diffusion processes to generate audio waveforms directly from text input, without relying on intermediate representations like spectrograms or phonemes. The model employs a pretrained BERT for text representation and a UNet architecture for waveform prediction, enabling high-fidelity audio generation comparable to state-of-the-art systems. E3 TTS simplifies the TTS pipeline, supports zero-shot tasks such as speech editing and prompt-based generation, and demonstrates improved audio diversity and speaker similarity. Experimental results indicate that E3 TTS achieves competitive performance on a proprietary dataset, showcasing its potential for broader applications in TTS technology.

---

### NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models

**2024-04-23**

https://arxiv.org/pdf/2403.03100v3

NaturalSpeech 3 introduces a novel text-to-speech (TTS) system that leverages factorized diffusion models and a new neural codec called FACodec to enhance speech synthesis quality, similarity, and prosody in a zero-shot manner. The system disentangles speech into distinct attributes—content, prosody, timbre, and acoustic details—allowing for more effective modeling and generation. Experimental results demonstrate that NaturalSpeech 3 outperforms state-of-the-art TTS systems in various metrics, achieving human-level naturalness on multi-speaker datasets. The work highlights the scalability of the model, achieving significant improvements with increased parameters and training data, while also enabling speech attribute manipulation through customizable prompts.

---

### FlashSpeech: Efficient Zero-Shot Speech Synthesis

**2024-04-25**

https://arxiv.org/pdf/2404.14700

FlashSpeech is a novel zero-shot speech synthesis system that significantly reduces inference time to approximately 5% of previous methods while maintaining high audio quality and speaker similarity. It employs a latent consistency model and introduces a novel adversarial consistency training approach, allowing it to be trained from scratch without a pre-trained model. Additionally, a new prosody generator enhances the diversity of speech prosody. Experimental results demonstrate that FlashSpeech is about 20 times faster than existing systems and excels in various tasks, including voice conversion and speech editing.

---