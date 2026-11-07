# Inference Techniques Survey

In Chapters 11 to 14, we implemented four basic features of model inference: sampling strategies, stopping criteria, the KV cache, and batch inference. They answer two questions: how to control what
the model generates, and how to make it generate faster.

But there is still a big gap between these basic features and industrial inference systems. Today's LLM services handle thousands of requests at the same time, models have tens or even hundreds of
billions of parameters, and every step of inference has been heavily optimized.

In this chapter, we survey the mainstream techniques in model inference.

> LLM technology evolves quickly. The content of this chapter is current as of November 2026.

## Why inference is slow

Each inference request has two distinct phases:

* **Prefill**: process all the prompt tokens at once, compute their keys and values, and generate the first token. The matrix multiplications are large, so the bottleneck is compute (compute-bound);
* **Decode**: from then on, process only one new token per step, generating serially. Every step has to read all the model parameters and the KV cache from GPU memory, but does very little
  computation, so the bottleneck is memory bandwidth (memory-bound).

Almost all inference optimizations revolve around three goals:

* **Read less data**: quantization, GQA, MLA;
* **Do more in parallel**: continuous batching, speculative decoding;
* **Do less repeated work**: KV cache, prefix caching.

### Memory footprint of the KV cache

The KV cache trades space for time, but space is not free. The cache size per token is:

$$
\text{size} = 2 \times L \times H_{kv} \times d_{head} \times \text{bytes}
$$

Where:

* $2$: one copy each for keys and values;
* $L$: number of layers;
* $H_{kv}$: number of key-value heads;
* $d_{head}$: embedding size per head;
* $\text{bytes}$: number of bytes per value, 2 for FP16.

Take Llama-2-7B (32 layers, 32 key-value heads, 128 dimensions per head, FP16) as an example. Each token takes 512 KB, so a sequence of 4096 tokens takes 2 GB. Serving 32 requests at the same time
needs 64 GB for the cache alone, more than the model itself (about 14 GB).

Llama-3-8B, on the other hand, uses GQA and has only 8 key-value heads. Each token needs only 128 KB, and the same sequence takes only 0.5 GB. This is why GQA and MLA became mainstream: they shrink
the cache directly.

### Metrics

An inference system should be evaluated from two angles: user experience and system efficiency:

* **Time to first token** (TTFT): the time from sending a request to receiving the first token, mainly determined by queuing and prefill;
* **Time per output token** (TPOT, also called inter-token latency): the interval between subsequent tokens, which determines how smooth the "typing" feels;
* **Throughput**: the total number of tokens the system generates per second, which determines the cost per unit;
* **Goodput**: the throughput contributed only by requests that meet the latency requirements;
* **Tail latency**: look at percentiles such as P99 rather than the average, because a few very slow requests seriously hurt the experience.

There are trade-offs among these metrics: the larger the batch size, the higher the throughput, but the higher the latency of each request. Designing an inference system means maximizing throughput
under the constraint of latency targets (SLOs).

## Model compression

The bottleneck of decoding is reading data, so the most direct approach is to make the model smaller.

### Quantization

**Quantization** uses low-precision integers (INT8, INT4) instead of FP16 or FP32 to represent parameters. Obviously, quantization reduces memory usage and speeds up decoding. Common quantization
schemes:

* **Weight-only quantization**: quantize the weights and dequantize them back to FP16 at compute time. GPTQ compensates for quantization error layer by layer using second-order information, and AWQ
  protects the few weight channels that matter most for the activations;
* **Weight and activation quantization**: quantize both weights and activations, which speeds up both prefill and decode. The difficulty is that activations contain a few extremely large **outliers**:
  LLM.int8() computes the outlier dimensions separately in FP16, and SmoothQuant shifts the quantization difficulty from the activations to the weights;
* **Low-precision floating point**: FP8 (Hopper architecture), FP4 (Blackwell architecture), and so on. They keep the dynamic range of floating point, which handles outliers more easily than integers.
  For example, gpt-oss released its MoE weights in MXFP4, so that a large model can fit on a single GPU;
* **KV cache quantization**: quantize the KV cache as well, which directly shrinks the cache;
* **GGUF**: the file format used by llama.cpp. It quantizes block by block, so that models can run on CPUs and consumer devices.

Quantization is lossy. Usually INT8 is almost lossless, while INT4 causes a slight loss of accuracy. Post-training quantization (PTQ) is the most common. When higher accuracy is needed, quantization
can also be introduced during training (QAT).

### Pruning

**Pruning** removes the unimportant parts of a model:

* **Unstructured pruning**: set weights at arbitrary positions to 0. The compression ratio is high, but it is hard to accelerate on ordinary GPUs. SparseGPT can prune a large model by 50% in one shot
  with almost no loss of accuracy;
* **Semi-structured pruning**: in every 4 consecutive weights, 2 are zero. The NVIDIA Ampere architecture has hardware support for it;
* **Structured pruning**: remove whole attention heads, channels, or even whole layers, getting a smaller dense model. It needs no special hardware, but the accuracy loss is large, and it usually
  requires retraining or knowledge distillation (see the fine-tuning survey).

## Attention optimization

Attention is the most special part of inference: its computation grows with sequence length, and so does its cache.

### Reducing key-value heads

The principles of MQA, GQA, and MLA were introduced in the pre-training survey. From the inference point of view, their value is that they shrink the KV cache directly: GQA shrinks the cache to a
fraction of MHA's, and MLA caches a low-dimensional latent vector, which can compress it by more than an order of magnitude. A smaller cache means the same memory can serve more requests at the same
time, and less data is read at each step.

### FlashAttention

FlashAttention computes attention in blocks and avoids writing the full attention matrix to GPU memory. It applies to the prefill phase of inference as well.

The decode phase has a special problem: each step has only one query token, and on a long sequence, parallelizing only along the query direction leaves the GPU poorly utilized. Flash-Decoding splits
the keys and values along the sequence-length direction, computes the parts separately, and then merges them, so that decoding with a long context can also make full use of the GPU.

### PagedAttention

The most naive implementation pre-allocates a contiguous block of cache for each request, big enough for the maximum length. But the actual length of a request is unknown in advance, which causes
serious memory waste and fragmentation.

PagedAttention (proposed by vLLM) borrows from the virtual memory of operating systems: it cuts the KV cache into fixed-size **blocks**, uses a block table to record the mapping between each request's
logical blocks and physical blocks, and allocates on demand. This leaves almost no fragmentation. It also lets multiple requests share the same blocks (for example, a common prompt), and copies a
block only when it needs to be modified (copy-on-write).

### Prefix caching

Many requests start with the same content: system prompts, few-shot examples, the history of a multi-turn conversation, and the unchanged context of repeated agent calls. **Prefix caching** saves the
KV cache of these common prefixes. A new request that hits the cache can skip the prefill of that part, which significantly reduces TTFT and cost.

In implementations, SGLang's RadixAttention manages the cache with a radix tree, and vLLM also provides automatic prefix caching. Many model-serving APIs also offer "prompt caching", where input
tokens that hit the cache are cheaper.

### KV cache compression

When the cache is still too large, it can be compressed further:

* **Quantization**: store the cache in INT8 or lower precision;
* **Eviction**: keep only the important tokens. StreamingLLM found that the first few tokens (attention sinks) must be kept, and together with a sliding window it can handle very long inputs. H2O,
  SnapKV, and similar methods pick important tokens based on historical attention scores;
* **Offloading**: move the rarely used cache to CPU memory or even SSD, and load it back when needed.

Eviction methods are lossy, so they require a trade-off between quality and memory.

### Long context

A longer context brings three kinds of pressure: the computation of prefill (which grows with the square of the length), the memory of the KV cache, and the bandwidth for reading the cache during
decoding. The responses include position encoding extrapolation (such as YaRN), sparse attention and linear attention (see the pre-training survey), the cache compression introduced above, and context
parallelism, which splits a long sequence across multiple GPUs.

Note that a model "supporting" a long context does not mean it can use it effectively. Evaluations such as RULER found that a model's effective context size is often shorter than the advertised
number, and research on "Lost in the Middle" also found that models do worse on information located in the middle of the text.

## Request scheduling

### Continuous batching

The batch inference we implemented is **static batching**: a batch of requests starts together and ends together, and the short requests have to wait for the longest one to finish. The GPU is wasted
while waiting, and newly arrived requests must queue for the next batch.

**Continuous batching** (also called iteration-level scheduling) schedules at the granularity of each decode step: as soon as a request finishes generating, it is swapped out and a waiting request is
swapped in. The composition of the batch can change at every step, and the GPU stays fully loaded. This is the foundation of modern inference engines.

### Chunked prefill

Prefill of a very long prompt takes a long time. If it is put in the same batch as the decoding of other requests, everyone's decoding is slowed down by it, and users notice an obvious stutter.

**Chunked prefill** cuts a long prompt into several small chunks, processes only one chunk per step, and mixes it with decoding requests in the same batch. This keeps GPU utilization high while
preventing decoding from being blocked by a long prefill, so TPOT is steadier.

### Prefill-decode disaggregation

Prefill is limited by compute, and decode is limited by bandwidth, so they need different hardware. Mixing them on the same GPUs makes them interfere with each other, and TTFT and TPOT cannot be
optimized separately.

**Prefill-decode disaggregation** puts them in different GPU pools: after the prefill pool computes the KV cache, it transfers it to the decode pool over a high-speed network. The two pools can scale
independently and choose different parallelism strategies and hardware. The cost is that the KV cache must be transferred, which demands high network bandwidth.

Today, mainstream engines such as vLLM, SGLang, and TensorRT-LLM all support this deployment mode. Systems such as NVIDIA Dynamo and llm-d provide multi-node orchestration on top, and route requests
based on KV cache hits.

## Distributed parallelism

When a single GPU cannot hold the model, or lower latency is needed, the model must be distributed across multiple GPUs:

* **Tensor parallelism**: split the matrices of each layer across multiple GPUs. It lowers the latency of a single request, but every layer needs communication, so it requires a high-speed
  interconnect such as NVLink and is usually limited to a single machine;
* **Pipeline parallelism**: put different layers on different GPUs. It needs less communication and suits multiple machines;
* **Data parallelism**: multiple complete copies of the model, each handling different requests, used to increase total throughput.

### Mixture of experts

Inference for MoE models has its own characteristics. Each token activates only a few experts, so the computation is small, but the weights of all the experts must stay in GPU memory. At each decode
step, only the selected experts are read. When the batch size is small, almost every expert processes very few tokens, and the bandwidth is poorly used; only when the batch size is large enough does
each expert get enough tokens to amortize the read.

So MoE usually uses **expert parallelism**: put different experts on different GPUs, and send tokens to the corresponding experts through all-to-all communication. This has to deal with load imbalance
among experts, communication overhead, and similar problems. Large-scale MoE services such as DeepSeek's often use dozens or even hundreds of GPUs for expert parallelism, together with prefill-decode
disaggregation.

## Decoding acceleration

Autoregressive decoding generates only one token at a time, which is the fundamental bottleneck of inference latency. Can we generate multiple tokens in each step?

### Speculative decoding

The core idea of **speculative decoding** is to use a cheap **draft** mechanism to guess the next several tokens first, and then let the large model **verify these guesses in parallel in one pass**.
Processing multiple tokens costs the large model almost the same as processing one (because decoding is limited by bandwidth and the compute is idle), so if the guesses are right, one forward pass
yields multiple tokens.

Verification uses a special rejection-sampling rule: keep the prefix that agrees with the large model's distribution, and resample from a corrected distribution at the first rejected position. This
rule guarantees that the distribution of the final output is exactly the same as generating token by token with the large model, so the speedup is lossless.

The drafts can come from several sources:

* **A separate small model**: a smaller model from the same family, which must use the same tokenizer as the large model;
* **n-gram or prompt lookup**: copy fragments directly from the prompt, which suits tasks such as summarization and code editing where the output highly overlaps with the input;
* **Extra prediction heads**: Medusa adds multiple prediction heads on the large model to predict several following positions at once;
* **The EAGLE family**: train a lightweight draft head on the large model's internal features, currently the most widely used;
* **Multi-token prediction**: the model carries MTP heads from pre-training (see the pre-training survey), which can be used directly as drafts at inference time.

The speedup depends on the **acceptance rate** (the fraction of drafts that are accepted) and the overhead of the draft itself. When the batch size is large, the GPU compute is no longer idle, so the
benefit of speculative decoding shrinks and can even become negative. Therefore, measure the acceptance rate under real traffic before going live.

### Non-autoregressive generation

Another direction is to give up left-to-right, token-by-token generation altogether. **Diffusion language models** (such as LLaDA, Mercury, and Gemini Diffusion) start from a sequence of masked
tokens, predict multiple positions in parallel in each round, and get the complete text after several iterations.

Their advantage is natural parallelism, and their disadvantages are also obvious: the overall quality usually lags behind autoregressive models of the same size, they need multiple rounds of denoising
to converge, and they cannot directly use the standard KV cache. For now, a more practical approach is to use a diffusion model as the draft generator for speculative decoding.

## Generation control

### More sampling strategies

We implemented temperature, top-k, and top-p. There are other common strategies:

* **Min-p**: keep only the tokens whose probability is at least "the maximum probability × a ratio". The threshold changes automatically with how sharp the distribution is, and it is more robust at
  high temperatures;
* **Typical sampling**: prefer tokens whose "information content is close to the average", avoiding tokens that are too ordinary or too rare;
* **Repetition, frequency, and presence penalties**: lower the probability of tokens that have already appeared, to reduce repetition;
* **Beam search**: keep multiple candidate sequences at the same time and choose the one with the highest overall probability. It suits tasks with a clear "standard answer", such as translation and
  summarization, but tends to produce dull, repetitive text in open-ended generation;
* **Logit bias**: adjust the probability of certain tokens by hand, to encourage or forbid specific tokens.

There is also an often overlooked problem about "determinism": even with the temperature set to 0, inference results can differ slightly across batch sizes. The reason is that GPU kernels are not "
invariant" to batch size: different batch sizes change the order in which floating-point operations are accumulated. In scenarios that need strict reproducibility, such as comparing training and
inference probabilities in reinforcement learning, batch-invariant kernels must be used.

### Constrained decoding

Many applications need the model output to strictly follow some format: JSON, a specific function-call format, or code that conforms to a grammar. A prompt can only raise the probability, not
guarantee it.

**Constrained decoding**, at each generation step, computes the set of "valid" next tokens from what has been generated so far, sets the logits of all other tokens to negative infinity, and then
samples. This guarantees that the output meets the requirements.

The constraints are usually described with regular expressions, context-free grammars, or JSON Schema, compiled into an automaton, and aligned with the vocabulary to get the valid tokens at each step.
Common tools include llama.cpp's GBNF grammars, Outlines, and XGrammar, and vLLM and SGLang also have built-in structured output.

Constrained decoding guarantees the format, but not the correctness of the content, and constraints that are too strong may hurt the quality of the content.

## Test-time scaling

Traditionally, the way to improve a model's capability was to train a bigger model. **Test-time scaling** offers another path: spend more compute at inference time to get better answers.

* **Chain-of-thought**: let the model write out its reasoning first and then give the answer. Reasoning models (such as OpenAI o1, DeepSeek-R1, and the thinking mode of Qwen3) internalize this ability
  through reinforcement learning and generate very long thinking processes;
* **Multiple samples with voting**: sample several answers to the same question and take the most frequent one (self-consistency);
* **Best-of-N**: sample N answers and pick the best with a reward model or a verifier;
* **Search**: use tree search and similar methods to explore multiple branches at each step of reasoning;
* **Thinking budget**: let users control how much the model thinks. Qwen3 supports "thinking / non-thinking" modes and can limit the length of thinking. DeepSeek-V4 also has thinking and non-thinking
  modes, and in thinking mode you can choose the reasoning effort (`reasoning_effort`).

Test-time scaling changes the load profile of inference systems: chains of thought greatly increase the output length of each request, so the KV cache and decode throughput become more prominent
bottlenecks. This is also why all the inference optimizations above are becoming more and more important.

## Inference engines

The techniques above are integrated into various **inference engines**, so users do not need to implement them from scratch:

| Engine               | Features                                                                                                                                                                                                           |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| vLLM                 | The most widely used open-source engine. PagedAttention, continuous batching, prefix caching, speculative decoding, many quantization schemes, and the most complete ecosystem                                     |
| SGLang               | Emphasizes RadixAttention prefix caching and structured output, and stands out in multi-turn conversation and agent scenarios                                                                                      |
| TensorRT-LLM         | NVIDIA's official engine, deeply optimized for its own GPUs. High peak performance, but more complex to deploy                                                                                                     |
| llama.cpp            | A C/C++ implementation that supports CPUs, Apple silicon, and all kinds of consumer devices. The origin of the GGUF format                                                                                         |
| Ollama, LM Studio    | Local tools built on top of llama.cpp and other engines, ready to use out of the box                                                                                                                               |
| MLX                  | Apple's array framework for its own chips, suitable for inference and fine-tuning on a Mac                                                                                                                         |
| NVIDIA Dynamo, llm-d | Multi-node orchestration layers on top of the engines, handling prefill-decode disaggregation and KV-aware routing. Dynamo can work with vLLM, SGLang, and TensorRT-LLM, and llm-d is built on vLLM and Kubernetes |

Hugging Face's TGI has entered maintenance mode, and the official recommendation for new deployments is vLLM, SGLang, or local engines such as llama.cpp and MLX.

As a general rule: to run locally on a personal computer, choose llama.cpp, Ollama, or MLX; to serve on GPU servers, choose vLLM or SGLang; to get the highest performance on NVIDIA hardware, consider
TensorRT-LLM.

---

In this chapter, we surveyed the mainstream techniques in inference:

* **Read less data**: quantization, pruning, GQA and MLA, KV cache compression;
* **Do less repeated work**: KV cache, PagedAttention, prefix caching;
* **Do more in parallel**: continuous batching, chunked prefill, prefill-decode disaggregation, speculative decoding;
* **Control generation**: more sampling strategies, constrained decoding;
* **Trade compute for quality**: test-time scaling.

The KV cache and batch inference we implemented are the starting point of these techniques: the cache removes repeated computation, and batching raises parallelism. Industrial inference systems are
built on this foundation, optimized layer by layer around the core bottleneck of memory bandwidth.

## References

* Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, 2023.
* Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022.
* Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022.
* Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, 2023.
* Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*, 2022.
* Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, 2022.
* OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, 2025.
* Frantar and Alistarh, *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*, 2023.
* Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, 2023.
* Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*, 2023.
* Li et al., *SnapKV: LLM Knows What You are Looking for Before Generation*, 2024.
* Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, 2023.
* Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?*, 2024.
* Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, 2023.
* Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, 2022.
* Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, 2024.
* Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, 2024.
* Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, 2022.
* Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, 2024.
* Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, 2024.
* Nie et al., *Large Language Diffusion Models*, 2025.
* Nguyen et al., *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs*, 2024.
* Dong et al., *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models*, 2024.
* Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, 2022.
* Snell et al., *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*, 2024.
* He et al., *Defeating Nondeterminism in LLM Inference*, Thinking Machines Lab, 2025.
