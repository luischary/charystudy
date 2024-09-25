## NLP -> Sampling -> Search



### Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

**2024-08-07**

https://arxiv.org/pdf/2408.03314v1

This paper investigates the scaling of inference-time computation in large language models (LLMs) to enhance their performance on challenging prompts. The authors propose a 'compute-optimal' strategy that adaptively allocates test-time compute based on the difficulty of the prompt, demonstrating that this approach can improve efficiency by over 4x compared to traditional methods. They analyze two main mechanisms for scaling test-time computation: searching against dense verifier reward models and adaptively updating the model's response distribution. The findings suggest that, in many cases, leveraging additional test-time compute can outperform using larger pre-trained models, particularly for easier and medium-difficulty questions. The study emphasizes the importance of understanding question difficulty and tailoring compute strategies accordingly, paving the way for future research on optimizing LLM performance through adaptive test-time computation.

---