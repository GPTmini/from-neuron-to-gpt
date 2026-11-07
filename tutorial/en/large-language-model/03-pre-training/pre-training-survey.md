# A Survey of Pre-training Techniques

Over the previous ten chapters, we built and pre-trained a GPT model step by step:

* Embedding layers (token embedding and position embedding, Chapters 1 to 2): turn text into vectors;
* Self-attention (causal, weighted, and multi-head, Chapters 3 to 6): let tokens understand each other;
* Pre-training (residual connection, layer normalization, and dropout, Chapters 7 to 10): stack the model deeper and train it stably.

Overall, what we implemented is basically the GPT-2 architecture (GPT-3 mainly scaled it up and used sparse attention in some layers). But compared with today's mainstream large models (Llama, Qwen,
DeepSeek, Gemma, and others), almost every part has changed a great deal, from tokenization to attention, and from module details to training data.

These new architectures are hard to demonstrate with small code and small data. In this chapter we survey the cutting-edge techniques of the pre-training stage and draw a complete technology map.
Interested readers can dig deeper from here.

> Large model technology evolves quickly; the content of this chapter is current as of November 2026.

## Technology comparison

A comparison table shows how the GPT we implemented differs from modern mainstream large models:

| Component                   | Our GPT                                             | Modern mainstream large models                                                                                                                                 |
|-----------------------------|-----------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Tokenization                | Character-level, vocabulary of about 65             | Byte-level BPE, vocabulary of 30,000 to 250,000                                                                                                                |
| Position encoding           | Trainable absolute position embedding               | Rotary position embedding (RoPE)                                                                                                                               |
| Attention                   | Multi-head attention (MHA), all global              | Grouped-query attention (GQA) or multi-head latent attention (MLA); alternating local and global layers, sparse attention, and in some models linear attention |
| Normalization               | Layer normalization (LayerNorm), pre-norm           | Root mean square normalization (RMSNorm), pre-norm, and QK-Norm in some models                                                                                 |
| Feed-forward network        | GELU, hidden layer 4×                               | SwiGLU, hidden layer about 2.7× to 3.5×, or mixture of experts (MoE)                                                                                           |
| Linear layer bias           | Yes                                                 | Usually none                                                                                                                                                   |
| Dropout                     | Yes                                                 | Usually none                                                                                                                                                   |
| Input and output embeddings | Independent                                         | Often shared in small models, usually independent in large ones                                                                                                |
| Numerical precision         | FP32                                                | BF16 mixed precision, FP8, FP4 explored in some cases                                                                                                          |
| Optimizer                   | AdamW                                               | AdamW, Muon in some models                                                                                                                                     |
| Learning rate schedule      | Warmup + cosine decay                               | Warmup + cosine decay, or warmup-stable-decay (WSD)                                                                                                            |
| Context size                | 256                                                 | 128K to 1M                                                                                                                                                     |
| Training data               | About 1.1 million characters of Shakespeare's plays | Trillions of tokens, cleaned, deduplicated, and mixed                                                                                                          |

## Tokenization

### Character-level tokenization

We use **character-level tokenization**: each character is a token, and the vocabulary has only a few dozen entries. It is simple, but the cost is high:

* **Sequences too long**: the encoded token sequence is very long, while the computation of attention is proportional to the square of the sequence length;
* **Low semantic density**: a single token carries almost no semantic information, and the model has to spend many layers of learning to assemble characters into words correctly.

The other extreme is **word-level tokenization**: each word is a token. Sequences get shorter, but the vocabulary becomes huge, and words that were never seen (out of vocabulary, OOV) cannot be
handled, such as newly coined words, misspellings, and personal names.

### Subword tokenization

**Subword tokenization** is the compromise: common words are kept whole, and rare words are split into smaller pieces. There are three main algorithms:

* **BPE** (Byte Pair Encoding): start from the smallest units and keep merging the most frequent adjacent pair of tokens in the corpus into a new token, until the vocabulary reaches a preset size.
  Used by GPT, Llama, Qwen, and others;
* **WordPiece**: similar to BPE, but when merging it picks the pair that increases the likelihood of the language model the most. Used by BERT;
* **Unigram**: the other way around: start from a very large candidate vocabulary and keep deleting the token that contributes least to the likelihood. Used by T5 and others.

Tokenizers are commonly trained with SentencePiece (which trains directly on raw text and treats spaces as ordinary characters, suitable for languages without spaces such as Chinese and Japanese) and
Hugging Face's tokenizers library.

### Byte-level BPE

The most mainstream scheme today is **byte-level BPE**: merging starts from the 256 bytes instead of from characters. Any text can be represented as a byte sequence, so OOV never occurs, and there is
no need to reserve room for the characters of various languages. It has been used since GPT-2.

Before BPE tokenization there is usually also a **pre-tokenization** step: a regular expression first roughly splits the text into words, numbers, punctuation, and so on, and BPE is then applied
inside each piece, which avoids merges that cross the natural boundaries of words. OpenAI's tiktoken is a commonly used fast encoding library, but it only encodes and does not train tokenizers.

### Vocabulary size

Vocabulary size is a trade-off:

* **Larger vocabulary**: the same text is encoded into a shorter token sequence, which lowers the cost of attention and inference. The advantage is especially clear for Chinese, code, and other
  non-English text. But the embedding layer and the output layer have more parameters, and low-frequency tokens appear less often and are trained insufficiently;
* **Smaller vocabulary**: exactly the opposite: the embedding layer and the output layer have fewer parameters, but sequences are longer.

Typical vocabulary sizes:

* GPT-2: about 50,000
* Llama 2: about 32,000
* Llama 3: about 128,000
* Qwen3: about 150,000
* Gemma: about 256,000

The trend is toward ever larger vocabularies, especially to support more languages better.

### Special tokens

Besides text pieces, the vocabulary also contains some **special tokens**: tokens marking the beginning and end of a sequence, padding tokens, and tokens that distinguish the system, user, and
assistant roles in a conversation. The end token is especially important: the model relies on it to learn "when to stop". We will see how they are used in the fine-tuning chapters.

### Multimodal tokens

Multimodal models likewise need to turn images and audio into token sequences:

* **Continuous patch embeddings**: for example ViT, which cuts an image into patches, applies a linear projection to each patch, and uses the results directly as input vectors;
* **Discrete visual tokens**: for example VQ-VAE, which quantizes an image into discrete codes and puts them in the same vocabulary as text tokens.

### Embedding sharing

In our model, the weights of the **token embedding** in the embedding layer and the weights of the linear layer in the output layer are two independent matrices. But their shapes happen to be the
same: both are `(vocabulary size, embedding size)` (the weights of our linear layer are stored as `(output, input)`).

**Weight tying** makes the two use the same set of parameters. GPT-2 does exactly this. When the vocabulary is large and the model is small, the embedding and output layers take a considerable share
of the parameters, so small models (such as Gemma, the small sizes of Llama 3.2, and the small sizes of Qwen) often adopt it; once the model gets larger, this part is a small share and is usually no
longer shared.

## Position encoding

### Absolute position embedding

We use a **trainable absolute position embedding**: each position has its own position vector, which is added to the word vector. It has two limitations:

* **Fixed length**: positions beyond the context size have no corresponding position vector, so text longer than the context size cannot be handled;
* **Absolute position**: what the model learns is the absolute position, whereas language cares more about the relative distance between tokens.

### The evolution of position encoding

* **Sinusoidal encoding**: the original scheme of the Transformer paper. It generates position vectors with sine and cosine functions of different frequencies and needs no training;
* **Trainable position embedding**: the scheme of GPT-2, and also the one we implemented;
* **ALiBi**: directly subtracts a value proportional to the token distance from the attention scores, with a different slope for each head. It needs no training and can handle longer sequences;
* **RoPE**: rotary position embedding, the most mainstream scheme today;
* **NoPE**: no position encoding at all, relying on the position information implied by the causal mask. Some models (for example SmolLM3) use no position encoding in some of their layers.

### Rotary position embedding

**Rotary position embedding** (RoPE) does not add a position vector to the word vector, but expresses position by rotating vectors.

RoPE divides the dimensions of the query vector ($q$) and the key vector ($k$) into groups of two, treats each group as a vector in a two-dimensional plane, and rotates it by an angle $m\theta_i$
proportional to the token position $m$, with different groups using different frequencies $\theta_i$. After the rotation, when the query vector at position $m$ and the key vector at position $n$ take
a dot product, the result depends only on their relative distance and has nothing to do with their absolute positions $m$ and $n$:

$$
(R_m q)^\top (R_n k) = q^\top R_{n-m} k
$$

Where:

* $R_m$, $R_n$: the rotation matrices for positions $m$ and $n$, and $R_{n-m}$ is the rotation matrix for the relative distance $n - m$;
* $q$, $k$: the query vector at position $m$ and the key vector at position $n$;
* $\top$: matrix transpose.

This design gives the model a natural sense of relative position, and there is no need to add a position vector to the word vector any more. The vast majority of mainstream open-source large models
adopt RoPE, including Llama, Qwen, DeepSeek, Gemma, and Mistral.

### Context extension

The context size during training is often far smaller than the final supported size, because training on long sequences is expensive. The common practice is:

* use shorter sequences (for example 4K) for most of the pre-training, and finally continue training with a small amount of long text to extend the context to 32K, 128K, or even longer;
* adjust the frequencies of RoPE at the same time: **position interpolation** compresses the positions into the range seen during training; **NTK-aware** scaling and **YaRN** treat different
  frequencies differently; one can also directly increase the base frequency of RoPE (Qwen3 raised it from 10,000 to 1,000,000).

## Attention

Both the computation of self-attention and the size of the attention score matrix are proportional to the square of the sequence length $n$: $O(n^2d)$. When the context grows from 4K to 128K, the
computation of the attention part increases about 1000 times.

This is the core driver of attention research in pre-training: how to make attention faster and cheaper, or to bypass the quadratic complexity altogether.

### Reducing key-value heads

In standard multi-head attention (MHA), each query head has its own set of keys and values. It is expressive, but at inference time too many keys and values need to be cached, which costs a lot of GPU
memory and bandwidth. Letting several heads share keys and values shrinks the cache by a multiple:

* **MQA** (Multi-Query Attention): all query heads share one set of keys and values;
* **GQA** (Grouped-Query Attention): query heads are divided into groups, and each group shares one set of keys and values;
* **MLA** (Multi-head Latent Attention): compresses keys and values into low-dimensional **latent vectors**, and the latent vectors are cached.

The main motivation of these algorithms is to reduce the key-value cache (KV cache) at inference time, but they all change the structure of the model, and are usually fixed during pre-training. There
are also approaches that start from an MHA model and convert it to GQA with a small amount of continued training. Most mainstream large models today adopt GQA, while DeepSeek, Kimi, and others adopt
MLA.

### Efficient attention computation

Standard attention has to write the complete attention score matrix to GPU memory and read it back for the Softmax and the weighted sum, so a lot of time is spent moving data.

**FlashAttention** uses **tiling**: it cuts the queries, keys, and values into small blocks and completes the Softmax and the weighted sum in the fast cache, without writing the complete attention
matrix to GPU memory. This reduces the memory use of this part from $O(n^2)$ to $O(n)$ and also speeds things up significantly. Its output is identical to standard attention: it is exact computation,
not an approximation. Improved versions for newer GPU architectures have followed FlashAttention. Today the pre-training of mainstream large models almost always enables it by default.

### Sparse attention

Most of the weights in the attention matrix are actually close to 0, so we can choose to compute only the most important positions:

* **Sliding window attention**: each token attends only to the nearest several tokens;
* **Global tokens**: a few tokens can attend to all tokens and be attended to by all tokens;
* **Alternating local and global layers**: most layers use sliding windows, and a few use full attention. For example, Gemma 3 pairs every 5 local attention layers with 1 global layer;
* **Trainable sparse patterns**: let the model learn which tokens to attend to. DeepSeek Sparse Attention (DSA) uses a lightweight indexer to pick the most important positions for each token;
  DeepSeek-V4 combines it with compression along the token dimension and makes a context of 1 million tokens its default configuration.

### Linear attention and state space models

Another path is to **bypass** the quadratic complexity entirely. **Linear attention** and **state space models** (SSM, such as Mamba) rewrite attention in a recurrent form: they summarize the history
with a fixed-size state instead of an ever-growing KV cache. The computation grows linearly with the sequence length, and the memory use at inference time is constant.

But compressing into a fixed-size state has a price: on tasks that need to recall a precise detail, pure linear models are weaker than full attention. So since 2025 a **hybrid architecture** has
become the more common practice: most layers use linear attention or state space layers, and a few keep full attention. Examples are Qwen3-Next (Gated DeltaNet and gated attention), Kimi Linear (
linear attention and MLA), and Nemotron 3 (Mamba layers, attention layers, and MoE).

### Attention stability

**QK-Norm** normalizes the query and the key before computing the attention scores, which prevents scores that are too large from saturating the Softmax and makes training more stable. Qwen3, Gemma 3,
OLMo 2, and other models all adopt it.

## The evolution of the Transformer block

The Transformer block we implemented is: normalization → attention → dropout → residual connection; normalization → feed-forward network → dropout → residual connection. Modern large models have
improved every submodule.

### Normalization

* Root mean square normalization (RMSNorm) is a simplified version of layer normalization: it omits the step of "subtracting the mean", scales with only the root mean square, and keeps only the
  learnable scaling factor $\gamma$ while dropping the shifting factor $\beta$. It is cheaper to compute and works as well as layer normalization, and it is the choice of almost all mainstream large
  models;
* **Pre-norm**: the Transformer paper places the normalization after the residual connection. GPT-2 moved it before the sublayer (which is also what we adopted), which makes training more stable
  because the residual channel stays clean. Some models (such as OLMo 2) place the normalization after the output of the sublayer, also for training stability.

### Feed-forward network

The **gated linear unit** (GLU): the feed-forward network is no longer "linear → activation → linear", but adds a gating mechanism:

$$
\text{FFN}(x) = \left(\text{Swish}(x W_1) \odot x W_3\right) W_2
$$

Where:

* $W_1$, $W_3$: the two weight matrices that project the input to the hidden layer;
* $W_2$: the weight matrix that projects the hidden layer back to the embedding size;
* $\odot$: element-wise multiplication;
* $\text{Swish}(x) = x \cdot \sigma(x)$, where $\sigma$ is the Sigmoid function.

One path, after the activation, "gates" the other, linear path. This structure is called **SwiGLU**; at the same parameter count it performs consistently better than GELU and has become standard in
most mainstream large models.

The feed-forward network has two more improvements:

* **Hidden layer size**: SwiGLU has 3 weight matrices, so to keep the parameter count unchanged the hidden layer shrinks from 4× to about 8/3×, and real models adjust it from there according to
  factors such as hardware alignment;
* **Mixture of experts** (MoE): replaces the feed-forward network with multiple **experts** and a **router**. The router selects the $k$ highest-scoring experts for each token, and only the selected
  experts take part in the computation. MoE lets the capacity of the model be determined by the total parameters, while the computation per token is determined only by the activated parameters.

Several design points of MoE:

* **Fine-grained experts and shared experts**: cut the experts smaller and make them more numerous, and keep one or two shared experts that every token passes through;
* **Load balancing**: a few experts must be prevented from being overused; an auxiliary loss is commonly used, and there are also methods that need no auxiliary loss and adjust the routing bias
  directly;
* **Scale comparison**: DeepSeek-V3 has 671B total parameters and activates 37B per token; Qwen3-235B-A22B has 128 experts per layer and activates 8; DeepSeek-V4-Pro has 1.6T total parameters and
  activates 49B.

### Residual connection

A standard residual connection has only one "main road". **Hyper-connections** widen the residual stream into several parallel channels and let them mix with each other, which strengthens
expressiveness. But unconstrained mixing amplifies the signal in deep networks and makes training unstable. The **mHC** (Manifold-Constrained Hyper-Connections) proposed by DeepSeek constrains the
mixing matrix to a specific manifold, restoring the identity mapping property of the residual connection. DeepSeek-V4 has adopted mHC. This is still a direction under exploration.

### Dropout

We use dropout in our model to prevent overfitting. But the pre-training of modern large models usually does not use it: the training data usually goes through only once, the model is underfitting
rather than overfitting, and dropout would only slow down convergence.

Only when the amount of data is insufficient and several epochs of repeated training are needed (as with our model) does dropout become worthwhile. The weight decay used in large-model pre-training
also does not mainly serve to prevent overfitting, but to work with the learning rate to shape the dynamics of optimization.

### Linear layer bias

The linear layers of modern large models usually drop the bias, which reduces parameters and simplifies the implementation. Some models once kept the bias in the QKV projections of attention (for
example Qwen2) and later dropped it as well (Qwen3 switched to QK-Norm to stabilize training).

## Training objectives

What we train is the most basic **next-token prediction**: each position predicts the next token. Modern large models extend it in a few ways:

* **Fill-in-the-middle** (FIM): cut out a middle segment of the text and let the model complete it from the text before and after. Widely used by code models to support code completion;
* **Multi-token prediction** (MTP): add extra prediction heads beside the backbone, so that each position predicts several following tokens at once. It provides a denser learning signal during
  training, and at inference it can also serve directly as the draft for speculative decoding (see the survey of inference techniques). Adopted by DeepSeek-V3, Qwen3-Next, and others;
* **Diffusion language models**: instead of generating one token at a time from left to right, they predict masked tokens in parallel and denoise iteratively. Their quality currently usually lags
  behind autoregressive models, and they remain an active research direction.

## Optimization

### Optimizers

The AdamW we use is also a commonly used optimizer today. Common configurations for large-model pre-training include:

* **Adam hyperparameters**: $\beta_1 = 0.9$, $\beta_2 = 0.95$; a smaller $\beta_2$ makes the model more sensitive to recent gradients and makes large-scale training more stable;
* **Weight decay**: usually 0.1;
* **Gradient clipping**: clip the global gradient norm to 1.0, which keeps the occasional large gradient from wrecking the stability of training.

**Muon** is the most closely watched alternative to AdamW in recent years. It is applied only to two-dimensional weight matrices: it first orthogonalizes the momentum and then updates the parameters,
while the remaining parameters (such as embeddings and normalization) still use AdamW. Muon keeps only one momentum buffer, so its optimizer state takes less GPU memory than AdamW's, and its sample
efficiency is also an advantage. Kimi K2 (1.04T parameters, 15.5T tokens) used its improved version MuonClip, and GLM-4.5, GLM-5, and DeepSeek-V4 also used Muon.

### Learning rate schedulers

We use the warmup cosine decay learning rate scheduler, which raises the learning rate to a peak through warmup and then decays it through cosine decay.

But it has a drawback: the total number of training steps must be fixed in advance, which is inflexible.

**WSD** (Warmup-Stable-Decay) does not need the amount of training to be fixed in advance:

* **Warmup**: the learning rate rises to the peak;
* **Stable**: the learning rate is held at the peak;
* **Decay**: only when we want to stop training is the learning rate decayed quickly over the last stretch (for example 10%).

This way, we can start from any checkpoint in the stable phase and decide how to finish, and it is also convenient to continue training or adjust the data after the stable phase.

### Scaling hyperparameters

Hyperparameters tuned on a small model are often no longer optimal on a large one. The usual practice is: the peak learning rate of a large model is smaller than that of a small model, and the batch
size is gradually increased during training. There are also methods such as **μP** (Maximal Update Parametrization), which adjust the initialization and the learning rate scaling of each layer so that
hyperparameters tuned on a small model can be transferred directly to a large one.

### Training stability

Large-model pre-training often lasts weeks or even months, and a **loss spike** or even divergence in the middle of training is very costly. Common causes include: a learning rate that is too large,
attention scores growing too fast and saturating the Softmax, numerical overflow, and a few abnormal data points. Commonly used countermeasures:

* **Prevention**: QK-Norm, gradient clipping, a smaller $\beta_2$, learning rate warmup, and suitable initialization;
* **Constraining the output**: **z-loss** adds a term to the loss that penalizes an overly large normalization term of the output-layer Softmax, to keep the output logits from running out of control;
* **Constraining the weights directly**: after every update, MuonClip directly rescales the query and key projection weights that make attention scores too large; according to the Kimi K2 technical
  report, its whole pre-training run had no loss spike;
* **Monitoring**: continuously record the loss, the gradient norm, the maximum of the attention scores, and the outliers of the activations, to catch problems before they grow;
* **Remedy**: after a spike, roll back to an earlier checkpoint and skip the data batches that triggered the spike.

## Training data

The abilities of a model depend largely on data. Data work usually takes most of the energy of a pre-training team.

### Data sources

* **Web pages**: from web crawls such as Common Crawl; the largest source, but of uneven quality;
* **Code**: from open-source code repositories; besides improving programming ability, it is also thought to help reasoning;
* **Books, papers, and encyclopedias**: high quality, but limited in quantity;
* **Mathematics and science text**: STEM data that is specially collected or synthesized;
* **Multilingual data**: the pre-training data of Qwen3 covers 119 languages and dialects.

Public large-scale pre-training datasets, such as FineWeb (about 15 trillion tokens), let researchers reproduce the complete data processing pipeline.

### Data cleaning

Raw data must go through several processing steps before it can be used. A typical pipeline is:

* **Extraction and filtering**: extract the main text from web pages, identify the language, and use heuristic rules to remove garbled, too-short, and overly repetitive text;
* **Quality classification**: train a classifier to score the quality of text and keep only high-scoring data. For example, FineWeb-Edu keeps web pages that are "educationally valuable", and DCLM uses
  a lightweight classifier;
* **Deduplication**: use algorithms such as MinHash to remove duplicate and highly similar documents. Duplicate data wastes compute and also makes the model memorize by rote;
* **Safety and privacy**: remove personal private information and harmful content;
* **Decontamination**: remove content that overlaps with evaluation benchmarks, so that evaluation results are not inflated.

### Data mixing

The proportions in which data from different sources are mixed have a large impact on model ability. Mixtures can be tuned through experiments on small models, and there are also methods such as
DoReMi that search for the mixture automatically. The gain from reusing the same data diminishes: some studies have found that data repeated up to 4 epochs works about as well as fresh data.

The pre-training of modern large models is usually **staged**, with higher data quality in the later stages. Take Qwen3 as an example:

1. Stage 1: about 30 trillion tokens with a sequence length of 4K, to build general language ability and knowledge;
2. Stage 2: about 5 trillion tokens, increasing the share of knowledge-intensive data such as STEM, code, and reasoning, and speeding up the learning rate decay;
3. Stage 3: several hundred billion tokens of long text, to extend the context to 32K.

Adding high-quality data during the learning rate decay phase is also called **annealing**. These later stages are sometimes called **mid-training** separately.

### Synthetic data

High-quality natural data is limited, so **synthetic data** is used more and more:

* **Rewriting**: have a model rewrite existing text into different styles or clearer expressions, which increases the diversity of the data. The pre-training data of Kimi K2 used rewriting;
* **Textbook-style data**: have a strong model generate textbook-style content and exercises. The Phi series of small models is the representative;
* **Domain synthesis**: Qwen3 used Qwen2.5-Math and Qwen2.5-Coder to generate mathematics and code data;
* **Verifiable synthesis**: generate problems with answers or test cases, and keep them only after verifying them with programs.

The risk of synthetic data is that a model repeatedly trained on its own generated data may gradually lose diversity. So it must be filtered strictly and mixed with real data.

## Scaling laws

### The relationship between loss and scale

**Scaling laws** found that the loss of a model has a predictable power-law relationship with the number of model parameters $N$, the amount of training data $D$, and the amount of compute $C$. A
commonly used form is:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

Where:

* $E$: the irreducible lower bound of the loss;
* $A$, $B$, $\alpha$, $\beta$: constants obtained by fitting experiments.

The compute of training is about $C \approx 6ND$. With scaling laws, we can fit curves on small-scale experiments and extrapolate to predict the loss of large models, making decisions before investing
huge amounts of compute.

### Compute-optimal

Given a fixed compute budget, how large a model should we train, and with how much data? Early research (Kaplan et al.) tended to spend more of the budget on enlarging the model. The later Chinchilla
research revised the conclusion: the number of parameters and the amount of data should grow together, and the **compute-optimal** ratio is about 20 tokens per parameter.

### Overtraining

Compute-optimal considers only the cost of training. But a model is trained once and used for inference countless times, and a smaller model is cheaper for inference. So modern large models are often
**overtrained**: a smaller model is trained with far more data than is compute-optimal. For example, the 8B model of Llama 3 was trained on about 15 trillion tokens, nearly a hundred times the
Chinchilla ratio; Qwen3 used 36 trillion tokens.

For MoE models, the parameter count in the scaling law should be the number of activated parameters, and the data requirement changes accordingly.

### Emergent abilities

After a model's scale exceeds some threshold, certain abilities seem to appear suddenly; these are called **emergent abilities**. But there is also research pointing out that this depends largely on
the evaluation metric: a metric that scores only "completely correct" shows abrupt jumps, while with a smooth metric the ability actually improves gradually. For readers, the more important conclusion
is that the trend of abilities improving with scale is predictable, but when a specific ability becomes usable still needs to be measured.

New dimensions of scaling have also appeared in the post-training and inference stages: the compute for reinforcement learning and the compute at inference time, which later chapters introduce.

## Training systems

### Parallelism strategies

A large model cannot fit in one GPU, so training must be split across hundreds or thousands of GPUs. Commonly used parallelism methods are:

* **Data parallelism**: each card holds a complete copy of the model and processes different data, and the gradients are then synchronized. **ZeRO** and **FSDP** go further and shard the optimizer
  state, gradients, and parameters across cards to save GPU memory;
* **Tensor parallelism**: split a single matrix multiplication across several cards. Communication is frequent, so it is usually limited to within one machine;
* **Pipeline parallelism**: put different layers on different cards, and the data passes through them in turn like an assembly line;
* **Sequence parallelism / context parallelism**: split a long sequence across several cards, to handle long contexts;
* **Expert parallelism**: an MoE model puts different experts on different cards.

Real training usually combines these methods. Megatron-LM, DeepSpeed, and torchtitan are commonly used training frameworks.

### Memory and efficiency

* **Mixed precision**: compute the forward and backward passes in BF16 and keep the optimizer state in FP32. FP8 training has been widely adopted, with DeepSeek-V3 as the representative; NVIDIA has
  also validated 4-bit floating-point training with NVFP4 on the Blackwell architecture, and in a training run with 12B parameters and 10 trillion tokens, the loss and downstream results were close to
  FP8;
* **Activation checkpointing**: do not save all intermediate results in the forward pass and recompute them in the backward pass, trading compute for memory;
* **Operator fusion and FlashAttention**: reduce reads and writes of GPU memory;
* **Overlapping communication and computation**: during training, let gradient synchronization, expert communication, and so on run at the same time as computation, hiding the latency of
  communication.

A commonly used metric of training efficiency is **model FLOPs utilization** (MFU): the fraction of the hardware's theoretical peak that is actually achieved. The MFU of large-scale training is
usually between 30% and 50%.

### Fault tolerance and persistence

The larger the scale, the more normal failures become. When Llama 3 was trained on 16K H100 GPUs, there were more than 400 unexpected interruptions in about 54 days, most of them hardware failures. So
large-scale training must have:

* **Regular checkpoints**: in the "Model persistence" chapter we saved the parameters, and large-scale training also needs to save the optimizer state, the position in data reading, and the random
  number state, to be fully recoverable;
* **Asynchronous saving**: overlap the process of writing checkpoints to storage with training, to reduce interruptions to training;
* **Automatic fault detection and recovery**: after a fault is found, automatically replace the node and continue training from the latest checkpoint.

## Evaluation

The goal of evaluation in the pre-training stage is to judge what the model has learned and whether training is going in the right direction:

* **Validation loss**: compute the loss (or perplexity) on held-out data. When comparing models with different tokenizers, it is often converted to loss per byte;
* **Downstream benchmarks**: use few-shot prompting to evaluate knowledge, reasoning, mathematics, and code abilities, for example MMLU, ARC, HellaSwag, GSM8K, and HumanEval. In small-scale
  experiments, tasks that already discriminate between small models should be chosen;
* **Data contamination**: if benchmark questions appear in the training data, the scores are not trustworthy, so decontamination is needed, and one should also watch out for benchmarks gradually
  saturating;
* **Evaluation tools**: open-source frameworks such as lm-evaluation-harness and OpenCompass can unify the evaluation process.

What pre-training yields is a base model, and its evaluation differs from that of a chat model, which is introduced in the fine-tuning survey.

---

In this chapter, we surveyed the mainstream techniques of pre-training:

* **Efficiency**: how to learn more with limited compute (byte-level BPE, RoPE, SwiGLU, MoE, FlashAttention, mixed precision);
* **Stability**: how to make training more stable (RMSNorm, QK-Norm, WSD, gradient clipping, MuonClip);
* **Length**: how to handle longer contexts (RoPE extension, GQA and MLA, sparse attention, linear attention, and hybrid architectures);
* **Data**: how to get better data (cleaning, deduplication, mixing, synthetic data);
* **Scale**: how to scale up in a planned way (scaling laws, parallelism strategies, training systems).

The GPT model we implemented from scratch is the starting point of these techniques. The basic structure of the Transformer is still the skeleton of mainstream models today, and later improvements
mostly patch this skeleton, while the hybrid architectures since 2025 have begun to change the skeleton itself.

## References

* Vaswani et al., *Attention Is All You Need*, 2017.
* Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019.
* Brown et al., *Language Models are Few-Shot Learners*, 2020.
* Kaplan et al., *Scaling Laws for Neural Language Models*, 2020.
* Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022.
* Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
* Press et al., *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*, 2021.
* Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023.
* Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022.
* Zhang and Sennrich, *Root Mean Square Layer Normalization*, 2019.
* Xiong et al., *On Layer Normalization in the Transformer Architecture*, 2020.
* Shazeer, *GLU Variants Improve Transformer*, 2020.
* DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, 2024.
* DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024.
* Meta, *The Llama 3 Herd of Models*, 2024.
* Qwen Team, *Qwen3 Technical Report*, 2025.
* Kimi Team, *Kimi K2: Open Agentic Intelligence*, 2025.
* NVIDIA, *Pretraining Large Language Models with NVFP4*, 2025.
* GLM-5 Team, *GLM-5: from Vibe Coding to Agentic Engineering*, 2026.
* Muennighoff et al., *Scaling Data-Constrained Language Models*, 2023.
* Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, 2024.
* Schaeffer et al., *Are Emergent Abilities of Large Language Models a Mirage?*, 2023.
