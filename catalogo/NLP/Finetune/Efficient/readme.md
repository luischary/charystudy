## NLP -> Finetune -> Efficient



### Parameter-Efficient Transfer Learning for NLP

**2019-06-13**

https://arxiv.org/pdf/1902.00751v2

The paper introduces a novel transfer learning approach for natural language processing (NLP) using adapter modules, which allows for parameter-efficient fine-tuning of large pre-trained models like BERT. Unlike traditional fine-tuning that requires retraining all model parameters for each task, adapter tuning adds only a small number of trainable parameters per task while keeping the original model's parameters fixed. This method enables the model to be extended to new tasks without losing performance on previously learned tasks. The authors demonstrate that adapter tuning achieves near state-of-the-art performance on 26 diverse text classification tasks, including the GLUE benchmark, while using significantly fewer parameters compared to full fine-tuning. The proposed architecture is particularly beneficial for applications requiring incremental learning and efficient resource utilization.

---

### LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

**2021-10-16**

https://arxiv.org/pdf/2106.09685v2

This paper introduces Low-Rank Adaptation (LoRA), a method for adapting large pre-trained language models to specific tasks without the need for full fine-tuning. LoRA freezes the pre-trained model weights and injects trainable low-rank matrices into each layer of the Transformer architecture, significantly reducing the number of trainable parameters and memory requirements. The authors demonstrate that LoRA can reduce trainable parameters by up to 10,000 times compared to traditional fine-tuning while maintaining or improving model performance across various benchmarks. They provide empirical evidence supporting the low intrinsic rank of weight updates during adaptation and release a package for integrating LoRA with PyTorch models.

---

### QLORA: Efficient Finetuning of Quantized LLMs

**2023-05-23**

https://arxiv.org/pdf/arXiv:2305.14314v1

QLORA introduces an efficient finetuning method that enables the finetuning of a 65B parameter language model on a single 48GB GPU while maintaining full 16-bit performance. The method employs a frozen, 4-bit quantized pretrained model and Low Rank Adapters (LoRA) for gradient backpropagation. Key innovations include the 4-bit NormalFloat data type, Double Quantization for reduced memory usage, and Paged Optimizers to manage memory spikes. The resulting Guanaco model family outperforms existing models on the Vicuna benchmark, achieving 99.3% of ChatGPT's performance. The study emphasizes the importance of data quality over size for model performance and provides a comprehensive analysis of chatbot capabilities, revealing discrepancies in current evaluation benchmarks. All models and code are made publicly available.

---

### LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention

**2023-06-14**

https://arxiv.org/pdf/arXiv:2303.16199v2

LLaMA-Adapter is a lightweight adaptation method designed to efficiently fine-tune the LLaMA model into an instruction-following model using only 1.2M learnable parameters. It leverages 52K self-instruct demonstrations and can be fine-tuned in less than one hour on 8 A100 GPUs. The method introduces learnable adaptation prompts at higher transformer layers and employs a zero-initialized attention mechanism with gating to preserve pre-trained knowledge while incorporating new instructional cues. The approach demonstrates competitive performance in generating high-quality responses, comparable to fully fine-tuned models like Alpaca, and extends to multi-modal instructions for improved reasoning in tasks such as ScienceQA and COCO Caption. Additionally, the zero-initialized attention mechanism is shown to enhance fine-tuning for other models like ViT and RoBERTa, indicating its generalization capacity across various vision and language tasks.

---

### DoRA: Weight-Decomposed Low-Rank Adaptation

**2024-03-05**

https://arxiv.org/pdf/2402.09353v3

This paper introduces Weight-Decomposed Low-Rank Adaptation (DoRA), a novel parameter-efficient fine-tuning method that enhances the learning capacity and stability of LoRA while maintaining inference efficiency. The authors conduct a weight decomposition analysis to reveal the distinct learning patterns between full fine-tuning (FT) and LoRA, leading to the development of DoRA, which decomposes pre-trained weights into magnitude and direction components. DoRA consistently outperforms LoRA across various tasks, including commonsense reasoning and visual instruction tuning, without incurring additional inference costs. The study also demonstrates DoRA's compatibility with other LoRA variants, showcasing its robustness and effectiveness in fine-tuning large language models.

---

### LoRA Learns Less and Forgets Less

**2024-05-15**

https://arxiv.org/pdf/2405.09673

This study investigates the performance of Low-Rank Adaptation (LoRA) compared to full finetuning in large language models, specifically in programming and mathematics domains. The authors find that while LoRA generally underperforms full finetuning in terms of accuracy and sample efficiency, it exhibits stronger regularization properties, resulting in less forgetting of the source domain and maintaining diversity in generated outputs. The research highlights that full finetuning learns high-rank weight perturbations, whereas LoRA's perturbations are typically low-rank. The authors propose best practices for using LoRA, emphasizing its sensitivity to hyperparameters such as learning rates and target modules. Overall, the findings suggest that LoRA is effective for certain applications but does not match the performance of full finetuning in challenging domains.

---