# Fine-Tuning Techniques Survey

The base model obtained from pre-training can do only one thing: continue text. In Chapters 15 to 18, the supervised fine-tuning (SFT) we implemented taught it to converse in the "prompt → response"
format, and direct preference optimization (DPO) made it prefer better responses. These two steps are a miniature of today's post-training pipeline for LLMs.

But industrial post-training is far more complex: larger-scale instruction data, more memory-efficient fine-tuning methods, preference alignment based on reinforcement learning, and the most important
breakthrough of the past two years: training reasoning models that "think" with verifiable rewards. In this chapter, we survey the mainstream techniques in fine-tuning and build a complete map of the
field.

> LLM technology evolves quickly. The content of this chapter is current as of November 2026.

## Overview of fine-tuning techniques

A modern LLM, from base model to final product, usually goes through these stages:

| Stage                                                 | Data                                          | Learning signal                       | Purpose                                                     |
|-------------------------------------------------------|-----------------------------------------------|---------------------------------------|-------------------------------------------------------------|
| Continued pre-training / mid-training                 | Large amounts of unlabeled text               | Next-token prediction                 | Add domain knowledge, extend the context size               |
| Supervised fine-tuning (SFT)                          | "Prompt + response" demonstrations            | Imitating demonstrations              | Learn the conversation format and instruction following     |
| Preference optimization (RLHF, DPO)                   | Comparisons of better and worse responses     | Preferences                           | Align with human preferences: helpful, harmless, style      |
| Reinforcement learning with verifiable rewards (RLVR) | Problems with standard answers                | Correct or wrong, judged by a program | Improve reasoning, math, coding, and agent abilities        |
| Distillation and merging                              | Outputs of a strong model, or multiple models | Imitation, parameter averaging        | Transfer abilities to small models, combine multiple models |

Different models combine these stages differently. Take Qwen3 as an example: its post-training has four steps: cold start with long chain-of-thought data → reinforcement learning for reasoning →
fusing the "thinking" and "non-thinking" modes → general reinforcement learning. Small models gain their abilities through "strong-to-weak" distillation. Llama 3, on the other hand, iterates over
multiple rounds of "sampling → rejection sampling → SFT → DPO" and does not use PPO.

## Advanced supervised fine-tuning

### The superficial alignment hypothesis

Our SFT used only a few hundred samples. Surprisingly, this works to some extent. The LIMA study fine-tuned a 65B base model with only 1000 carefully selected samples and obtained strong
conversational ability. This led to the **superficial alignment hypothesis**: almost all of a model's knowledge and abilities come from pre-training, and fine-tuning mainly teaches the model which
style and format to use in its responses.

The hypothesis has some truth, but it is incomplete:

* For behaviors such as format, style, politeness, and refusals, a small amount of high-quality data is indeed enough;
* For abilities that need to be added or strengthened, such as math, coding, and reasoning, large amounts of high-quality data are needed, and the scale of post-training data has kept growing in
  recent years.

So a more accurate statement is: the quality and diversity of the data matter more than its quantity, and different abilities need different amounts of data.

### Instruction data

Where does instruction data come from?

* **Human-written**: the highest quality and the highest cost. For example, OpenAssistant was written by volunteers, while companies hire annotators;
* **Converting existing datasets**: rewrite traditional NLP tasks (classification, translation, question answering) into instruction form, such as FLAN;
* **Model-generated**: use a strong model to generate instructions and responses. Self-Instruct lets the model expand seed tasks by itself, Alpaca used it to generate 52,000 samples, and UltraChat,
  Evol-Instruct, and others further increased complexity and diversity;
* **Real conversations**: real conversations between users and models, which cover a wide range but raise privacy and quality issues;
* **Verifiable synthetic data**: for math and code, have a model generate multiple solutions and keep only the correct ones after program verification (rejection sampling).

The key points of data processing: deduplication, filtering out low-quality and harmful content, ensuring diversity of tasks and domains, and controlling the difficulty distribution. Also note that
training a model on the outputs of other commercial models may be restricted by their terms of service.

### Chat template

A multi-turn conversation must tell the model which segment is the system prompt, which is what the user said, and which is the assistant's response. This is done by a **chat template**: special
tokens mark the role and boundaries of each message. For example, the ChatML format wraps each message in `<|im_start|>` and `<|im_end|>`, while Llama 3 uses dedicated role header tokens.

A few key points:

* **End-of-turn marker**: the end of each assistant response needs an end token, so that the model learns to stop. Our implementation uses a blank line as the end marker, while industry uses dedicated
  special tokens;
* **Loss only on the response**: the user's and system's content serves only as the condition and is not counted in the loss. This is exactly the loss mask we implemented in SFT;
* **The template must be consistent**: the template used in training and inference must be exactly the same, otherwise the model's performance drops noticeably;
* **Extended roles**: tool calls, tool results, and the thinking process (such as the `<think>` tag) are also distinguished by special markers in the template.

### Training tips

* **Learning rate**: 1 to 2 orders of magnitude smaller than in pre-training, and about 1e-5 is common for full-parameter fine-tuning;
* **Epochs**: usually 2 to 3, and too many cause overfitting and forgetting of pre-trained knowledge;
* **Packing**: concatenate multiple short samples into one long sequence to improve GPU utilization. An attention mask is needed to separate the different samples so that they cannot "see" each other;
* **Loss normalization**: averaging per token or per sample changes the weight that long and short responses get in training, so pay attention to it;
* **Data replay**: mix some pre-training data or general data into the fine-tuning data to reduce forgetting;
* **Splitting a validation set**: our SFT did not use a validation set, but in practice you should monitor the validation loss and the performance on real tasks.

### Continued pre-training

When a model needs to master a new domain (medicine, law, code) or a new language, the small amount of SFT data is not enough, and the model must be **continually pre-trained** on large amounts of
unlabeled text. The training objective is exactly the same as in pre-training.

Key points:

* Warm the learning rate up again and then decay it gradually; the learning rate is usually smaller than in pre-training from scratch;
* Mix in some of the original general data (replay) to prevent catastrophic forgetting;
* At the end of pre-training, large models use high-quality data and long texts for "mid-training" (see the pre-training survey), which is essentially also continued pre-training.

## Parameter-efficient fine-tuning

### The cost of full-parameter fine-tuning

Full-parameter fine-tuning has to store many things for every parameter. With BF16 mixed precision and the Adam optimizer, each parameter needs: 2 bytes for the weight + 2 bytes for the gradient + 4
bytes for the FP32 master weight + 4 bytes each for Adam's two moments, 16 bytes in total. A 7B model needs about 112 GB of GPU memory, not including activations, and a 70B model needs more than 1 TB.

In addition, every task needs a complete copy of the model, so serving multiple tasks makes storage and switching expensive.

### LoRA

**LoRA** (Low-Rank Adaptation) is based on an observation: the weight change brought by fine-tuning roughly lies in a low-rank subspace. So we freeze the original weight $W$ and train only a low-rank
increment:

$$
W' = W + \frac{\alpha}{r} B A, \quad A \in \mathbb{R}^{r \times k},\ B \in \mathbb{R}^{d \times r},\ r \ll \min(d, k)
$$

Where:

* $W$: the original weight, of shape $d \times k$, frozen during training;
* $A$, $B$: two small low-rank matrices, the only parameters that need training;
* $r$: the rank, much smaller than $d$ and $k$;
* $\alpha$: a scaling coefficient that, together with $r$, determines the size of the increment.

$B$ is initialized to 0, so at the start of training the model is identical to the original. The number of trainable parameters drops from $dk$ to $r(d+k)$, usually 1% of the original or less.

The benefits of LoRA:

* **Saves memory**: no gradients or optimizer states need to be stored for the frozen weights;
* **Easy to deploy**: after training, $BA$ can be merged back into $W$, with no extra inference cost; alternatively, the adapters can be kept, and multiple adapters can be switched on demand on the
  same base model to serve multiple tasks;
* **Easy to store**: each task needs to store only an adapter of a few dozen MB.

Research from Thinking Machines ("LoRA Without Regret") gave some practical conclusions:

* On SFT datasets of common sizes, as long as the adapter has enough capacity, LoRA can match full-parameter fine-tuning; when the data exceeds the capacity, LoRA falls behind;
* LoRA should be applied to all linear layers, including the MLP layers; applying it only to the attention layers works worse;
* The optimal learning rate for LoRA is about 10 times that of full-parameter fine-tuning, and it is more sensitive to large batch sizes than full-parameter fine-tuning;
* In reinforcement learning, each sample provides very little information, so even rank 1 can match full-parameter fine-tuning.

### QLoRA

QLoRA builds on LoRA by quantizing the frozen base weights to 4 bits (NF4 format) and training the adapters only in BF16, dequantizing the weights on the fly in the forward and backward passes.
Combined with double quantization and paged optimizers, a single 48 GB GPU can fine-tune a 65B model. The cost is that dequantization adds extra computation, so training is slower than LoRA.

## Reinforcement learning from human feedback

SFT can only make the model imitate demonstrations, but often "what is a good response" is hard to write as a demonstration and easy to compare: given two responses, people can easily tell which one
is better. **Reinforcement learning from human feedback** (RLHF) exploits exactly this comparison information. The 1.3B model that InstructGPT trained with it beat the 175B GPT-3 in human evaluations.

### Three steps

1. **Supervised fine-tuning**: train on demonstration data to get the initial policy (which is also the reference model later);
2. **Training a reward model**: collect human preference comparisons among multiple responses to the same prompt, and train a model to score responses;
3. **Reinforcement learning**: let the model generate responses, score them with the reward model, and update the model with a reinforcement learning algorithm so that it gets higher scores.

### Reward model

A reward model is usually a language model with an extra head that outputs a scalar. It is trained with the Bradley-Terry model: assume that the probability that response $y_c$ is better than $y_r$
depends on the difference of their rewards:

$$
P(y_c \succ y_r \mid x) = \sigma\big(r(x, y_c) - r(x, y_r)\big)
$$

Where:

* $x$: the prompt;
* $y_c$, $y_r$: the chosen answer and the rejected answer;
* $r(x, y)$: the score the reward model gives to response $y$;
* $\sigma$: the Sigmoid function.

The training loss is $-\log \sigma\big(r(x, y_c) - r(x, y_r)\big)$, which makes the preferred response score higher.

A reward model is only an imperfect proxy for human preferences. The model exploits its loopholes: it generates responses that are not truly better but score higher with the reward model, such as
longer and more flattering ones. This is called **reward hacking** or over-optimization.

### Reinforcement learning

The optimization objective of the reinforcement learning stage is:

$$
\max_{\pi} \ \mathbb{E}_{x,\, y \sim \pi}\big[r(x, y)\big] - \beta\, \mathrm{KL}\big(\pi \,\|\, \pi_{\text{ref}}\big)
$$

Where:

* $\pi$: the policy being optimized, that is, the language model;
* $\pi_{\text{ref}}$: the reference model;
* $\mathrm{KL}$: the KL divergence, which measures the difference between two probability distributions;
* $\beta$: the coefficient of the KL penalty.

The first term maximizes the reward, and the second term uses the KL divergence to penalize the policy for drifting too far from the reference model. Its role is to keep the responses fluent and to
limit the room for reward hacking. The larger $\beta$ is, the stronger the constraint.

The most commonly used algorithm is **PPO** (Proximal Policy Optimization):

* It needs a **value model** (critic) to estimate the value at each position, and then uses generalized advantage estimation (GAE) to compute the advantage of each token;
* It uses a clipped objective function to limit the size of each update and avoid excessive policy changes;
* The process is a loop: sample responses → score them → compute advantages → run several rounds of mini-batch updates on the same batch of data.

In a language model, each token is an "action", and the reward is usually given only at the end of the whole response.

### The cost of RLHF

* **Many models**: the policy, reference, reward, and value models must be maintained at the same time, which is a heavy burden in memory and engineering;
* **Online sampling**: every round has to generate responses with the current model, and generation is slow;
* **Hard to tune**: it is sensitive to hyperparameters, training is unstable, and reward hacking is hard to guard against;
* **Expensive annotation**: high-quality human preference data is costly.

### The relationship between DPO and RLHF

DPO starts from the discovery that the optimization objective above has a closed-form optimal solution:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\frac{r(x, y)}{\beta}\Big)
$$

Where:

* $\pi^*$: the optimal policy;
* $Z(x)$: a normalization constant that makes $\pi^*$ a probability distribution.

Conversely, the reward can be expressed in terms of the policy:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Substituting it into the Bradley-Terry preference probability, the term $Z(x)$, which does not depend on $y$, cancels exactly in the difference between the two responses. The reward model is "
eliminated", and we get a loss that depends only on the policy and the reference model:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\Big(\beta\Big[\log\frac{\pi_\theta(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \log\frac{\pi_\theta(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)}\Big]\Big)
$$

Where:

* $\pi_\theta$: the policy being trained;
* the other symbols are the same as above.

This is the DPO we implemented. So DPO is not a method completely different from RLHF, but an offline, simplified way of solving the same optimization objective: no reward model, no online sampling,
and no value model, and it trains just like SFT. The quantity in the brackets can be seen as the model's implicit reward, and $\beta$ is the strength of the KL constraint.

## Other preference optimization algorithms

### Problems with DPO

DPO is simple and effective, but it has some known problems:

* **Distribution shift of offline data**: the preference data was generated by other models and does not match the distribution of the current policy, so the training signal does not necessarily
  correspond to the mistakes the current model makes;
* **Likelihood displacement**: DPO cares only about the gap between good and bad responses, so during training the probabilities of both may fall together, and the probability of the good response may
  even drop below its value before training;
* **Length bias**: it tends to prefer longer responses;
* **Overfitting**: preference data almost always has a definite "winner and loser", and when $\beta$ is too small, the policy drifts too far from the reference model;
* **Noise sensitivity**: noise in the annotations directly affects training.

### Online preference optimization

The fundamental limitation of offline methods is that the data does not come from the current policy. Online or iterative methods let the model generate new data while it trains:

* **Online DPO, iterative DPO**: sample multiple responses with the current model, mark good and bad with a reward model or a judge model, build preference pairs, and repeat after training;
* **Rejection sampling fine-tuning**: sample multiple responses for each prompt, keep only the highest-scoring ones, and use them for SFT. It is simple, stable, and widely used;
* **RLOO**: an improved version of REINFORCE that uses the average reward of the other responses to the same prompt as the baseline, with no need for a value model.

### AI feedback

Human annotation is expensive and slow, so AI is increasingly used to provide feedback (RLAIF):

* **LLM as a judge**: have a strong model compare or score responses to generate preference data, such as UltraFeedback;
* **Constitutional AI**: give a set of principles (a "constitution"), have the model critique and rewrite its own responses according to the principles, and then train on the AI's preferences;
* **Self-rewarding**: let the model act as its own judge and score its own responses.

AI judges have biases of their own: they prefer longer responses, responses placed first, and responses in their own style. So human data is still needed for calibration.

## Reinforcement learning with verifiable rewards

### Verifiable rewards

The reward of RLHF comes from a trained reward model that is easy to exploit. For tasks with objective answers, the reward can be computed directly by a program:

* **Math**: whether the final answer matches the standard answer;
* **Code**: whether it passes the unit tests;
* **Instruction following**: whether the response satisfies checkable constraints (word count, format, keywords);
* **Logic puzzles**: whether the answer satisfies the rules.

This is **reinforcement learning with verifiable rewards** (RLVR), a name proposed by Tülu 3. Its benefits are that the reward is objective, no reward model has to be trained, and it is hard to hack.
Its limitation is that it applies only to tasks whose correctness can be verified automatically.

### GRPO

**GRPO** (Group Relative Policy Optimization) was proposed by DeepSeekMath, and it removes the value model from PPO. The approach: for each prompt, sample a group of ($G$) responses and use the
relative reward within the group as the advantage:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

Where:

* $\hat{A}_i$: the advantage of the $i$-th response;
* $r_i$: the reward of the $i$-th response;
* $\text{mean}$, $\text{std}$: the mean and standard deviation of the rewards of a group of responses.

Within a group, responses better than average are encouraged, and those worse than average are suppressed. The optimization objective is still a PPO-style clipped objective: for each token, compute
the probability ratio between the new and old policies and multiply it by this advantage, usually with a KL penalty against the reference model added.

Removing the value model saves a lot of memory and engineering complexity, and there is no longer a value function that is hard to train. GRPO has therefore become one of the mainstream algorithms for
training reasoning models.

### DeepSeek-R1

DeepSeek-R1, released in January 2025, made RLVR the focus of attention:

* **R1-Zero**: run reinforcement learning with GRPO directly on the base model, where the reward is only whether the answer is correct and whether the format is valid, with no human-written thinking
  demonstrations at all. During training, the model spontaneously learned to generate longer thinking, check itself, and go back to correct itself, and its reasoning ability improved significantly;
* But the output of R1-Zero is hard to read and mixes languages. So the official R1 adopted a four-stage pipeline: cold-start SFT with a small amount of long chain-of-thought data → RL for reasoning →
  generating data by rejection sampling from the RL model and doing SFT again → RL for all scenarios;
* Distilling the data generated by R1 into small models works better than running reinforcement learning directly on small models.

This pipeline has become the standard practice for reasoning models. Later models such as Qwen3 also adopted a similar "cold start → reasoning RL → general RL" pipeline.

### Variants and improvements

GRPO exposed some problems in large-scale training with long chains of thought, which gave rise to a batch of improved methods:

* **Dr. GRPO**: points out that normalizing by response length and by standard deviation in GRPO introduces biases (for example, wrong responses get longer and longer), and removes both;
* **DAPO**: a set of practical techniques proposed by ByteDance: Clip-Higher (raise the upper clipping bound so that low-probability tokens have more room to grow, which eases entropy collapse),
  dynamic sampling (discard prompts whose whole group is all correct or all wrong, because their advantage is 0 and there is no learning signal), token-level loss (average per token rather than per
  response), and a penalty for overlong responses, while also removing the KL term;
* **GSPO**: changes the importance ratio from token level to sequence level (the geometric mean over the whole response), which reduces variance and is especially stable for MoE models;
* **CISPO**: clips the importance weights themselves, instead of directly discarding the gradients of tokens that fall outside the range.

Another class of problems comes from the system: the training engine and the inference engine are not numerically identical, which quietly turns supposedly "online" sampling into "offline" sampling. A
common fix is truncated importance sampling.

Beyond math and code, RLVR has been extended to agent scenarios: the model completes tasks in multi-turn environments where it can use tools (for example, fixing a bug in a code repository), and
whether the task is completed serves as the reward. DeepSeek-V4 (released in April 2026) is one example: it targets long-running agent tasks and is trained with reinforcement learning in real tool
environments.

What reinforcement learning actually brings is still debated: some studies found that RLVR mainly raises the probability that the model gets a problem right in one attempt, but with a large number of
samples, the range of problems the base model can solve has not clearly expanded. This remains an active research topic.

### Process rewards

The rewards above look only at the final answer and are called **outcome rewards**. A **process reward model** (PRM) scores every step of the reasoning:

* Advantage: the signal is denser, it can locate the step where an error occurred, and it can also be used to filter branches during search;
* Disadvantage: manually annotating every step is extremely costly, the division into "steps" is hard to define, and it is easier to exploit.

There are also quite a few methods for building process labels automatically (for example, estimating the value of each step with Monte Carlo sampling, or learning it implicitly from outcome rewards).
But the DeepSeek-R1 report mentions that their attempts with process rewards had limited effect. Mainstream reasoning models still rely mainly on outcome rewards.

## Knowledge distillation and model merging

### Knowledge distillation

**Knowledge distillation** lets a small model (the student) learn the behavior of a large model (the teacher):

* **Soft-label distillation**: make the student's output distribution approach the teacher's (minimizing the KL divergence). It carries much more information than the "hard label" of a single token,
  but the student and teacher must use the same vocabulary;
* **Data distillation**: have the teacher generate responses and then do SFT on the student with this data. The small models distilled from Alpaca and R1 both took this approach, which is simple and
  also works with closed-source teachers;
* **On-policy distillation**: the student generates responses, the teacher evaluates each token, and the student moves closer to the teacher. The student trains on the distribution it will actually
  encounter, and every token carries a signal. Qwen3's "strong-to-weak" distillation first does data distillation with the teacher's outputs and then on-policy distillation. The Qwen3 report shows
  that, on the same small model, on-policy distillation used about one tenth of the GPU time of reinforcement learning and reached a higher score.

When using distillation, pay attention to the license of the teacher model.

### Model merging

**Model merging** needs no training and directly combines multiple models into one in parameter space:

* **Linear averaging**: average the parameters of multiple fine-tuned models (model soups), which is usually more robust than any single model. Llama 3 averaged models trained with different data or
  hyperparameters;
* **Task vectors** (task arithmetic): subtract the base parameters from the fine-tuned parameters to get a "task vector". Adding several task vectors gives a model several abilities at once, and
  subtracting one can make the model "forget" an ability;
* **TIES, DARE**: handle conflicts among multiple task vectors: TIES trims small changes and resolves sign conflicts; DARE randomly drops a portion of the changes and rescales the rest;
* **SLERP**: interpolate the parameters of two models on a sphere.

Merging requires the models to come from the same base model, and the result is not always guaranteed, so it needs evaluation. The open-source community often uses tools such as mergekit.

## Training systems

### Memory optimization

The memory of fine-tuning comes mainly from weights, gradients, optimizer states, and activations. Common ways to save memory:

* **ZeRO and FSDP**: shard the optimizer states, gradients, and weights across multiple GPUs;
* **Activation recomputation**: do not save intermediate results in the forward pass, and recompute them in the backward pass;
* **Gradient accumulation**: accumulate the gradients of several small batches to simulate a larger batch size;
* **Low-precision optimizers**: for example, 8-bit optimizers, which shrink the optimizer states;
* **Offloading**: move the optimizer states, or even the weights, to CPU memory or SSD;
* **Parameter-efficient fine-tuning**: LoRA and QLoRA, which fundamentally reduce the state that must be stored;
* **Fused kernels**: reduce memory reads and writes and save activations.

### Training systems for reinforcement learning

Reinforcement learning training is completely different from SFT: every round first lets the model generate a large number of responses and then updates the model with them. Generation often takes
most of the time, and the lengths of the responses vary widely, so a few very long responses slow down the whole batch.

So a reinforcement learning system usually has two parts:

* **Inference engine** (vLLM, SGLang): responsible for generating responses efficiently;
* **Training engine** (FSDP, Megatron): responsible for computing the loss and updating the parameters.

The latest weights must be synchronized between the two. There are two deployment modes: colocated (the same GPUs run generation and training in turns, saving GPUs) and disaggregated (generation and
training use different GPUs and can run asynchronously, but generation uses a slightly older model, so data staleness has to be handled).

In addition, computing the reward is itself a systems problem: for example, running code concurrently in sandboxes, calling verifiers, and managing the environments of multi-turn agents. The numerical
differences between the training and inference engines must be handled, as mentioned in the inference survey, with batch-size-invariant kernels and importance correction.

### Common tools

* **Hugging Face TRL**: provides trainers for SFT, DPO, GRPO, and more; PEFT provides methods such as LoRA;
* **Axolotl, LLaMA-Factory, Unsloth**: ready-made fine-tuning tools that work through configuration files; Unsloth especially emphasizes training efficiency on a single GPU;
* **verl, OpenRLHF**: frameworks for large-scale reinforcement learning that integrate inference engines and distributed training;
* **DeepSpeed, Megatron**: low-level distributed training frameworks.

PyTorch's official torchtune is no longer actively maintained (its documentation says development wound down in 2025), so it is not recommended for new projects. These tools update quickly, so when
choosing, look at how active they are and how well they support new algorithms.

### Evaluation

A fine-tuned model is evaluated along many more dimensions than a base model:

* **Instruction following**: IFEval evaluates with automatically checkable constraints, while MT-Bench, AlpacaEval, and Arena-Hard use a strong model as the judge;
* **Human preference**: platforms such as Arena (formerly LMArena) let users vote on the responses of anonymous models to produce a ranking;
* **Reasoning**: math problems such as AIME and MATH-500, and science question answering such as GPQA;
* **Code**: HumanEval, LiveCodeBench, and evaluations such as SWE-bench Verified that fix bugs in real code repositories;
* **Agents and tool use**: BFCL evaluates function calling, and τ-bench and Terminal-Bench evaluate multi-turn tasks;
* **Safety**: the refusal rate for harmful requests, robustness under jailbreak attacks, and whether requests that should not be refused are over-refused.

Evaluation has its own pitfalls: biases of judge models, contamination and saturation of benchmarks, and reward hacking in training, where the optimized metric improves but the real quality does not.
So evaluation should use multiple complementary metrics and include human inspection wherever possible.

---

In this chapter, we surveyed the mainstream techniques in fine-tuning:

* **Supervised fine-tuning**: high-quality, diverse instruction data, chat templates, loss masks, continued pre-training;
* **Parameter efficiency**: LoRA and QLoRA, which reach results close to full-parameter fine-tuning with very few trainable parameters;
* **Preference alignment**: RLHF, DPO and its variants, online preference optimization, AI feedback;
* **Reasoning ability**: RLVR, GRPO and its improvements, the DeepSeek-R1 pipeline, process rewards;
* **Transfer and fusion**: knowledge distillation, model merging;
* **Systems and evaluation**: memory optimization, training systems for reinforcement learning, multi-dimensional evaluation.

The SFT and DPO we implemented from scratch are the starting point of this chain: SFT teaches the model the format through demonstrations, and DPO teaches the model preferences through comparisons.
Reinforcement learning moves the model from "imitating human responses" to "finding solutions by trial and error", which is one of the fastest-improving directions of LLM capability today.

## References

* Zhou et al., *LIMA: Less Is More for Alignment*, 2023.
* Wang et al., *Self-Instruct: Aligning Language Models with Self-Generated Instructions*, 2022.
* Wei et al., *Finetuned Language Models Are Zero-Shot Learners*, 2021.
* Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*, 2022.
* Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
* Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023.
* Schulman et al., *LoRA Without Regret*, Thinking Machines Lab, 2025.
* Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
* Rafailov et al., *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model*, 2023.
* Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*, 2024.
* Bai et al., *Constitutional AI: Harmlessness from AI Feedback*, 2022.
* Cui et al., *UltraFeedback: Boosting Language Models with High-quality Feedback*, 2023.
* Yuan et al., *Self-Rewarding Language Models*, 2024.
* Lambert et al., *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*, 2024.
* Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024.
* DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025.
* Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*, 2025.
* Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 2025.
* Zheng et al., *Group Sequence Policy Optimization*, 2025.
* MiniMax, *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*, 2025.
* Yao et al., *Your Efficient RL Framework Secretly Brings You Off-Policy RL Training*, 2025.
* Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*, 2025.
* Lightman et al., *Let's Verify Step by Step*, 2023.
* Meta, *The Llama 3 Herd of Models*, 2024.
* Qwen Team, *Qwen3 Technical Report*, 2025.
* Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*, 2023.
* Lu et al., *On-Policy Distillation*, Thinking Machines Lab, 2025.
* Wortsman et al., *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy without Increasing Inference Time*, 2022.
* Ilharco et al., *Editing Models with Task Arithmetic*, 2022.
* Yadav et al., *TIES-Merging: Resolving Interference When Merging Models*, 2023.
* Yu et al., *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch*, 2023.
* Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, 2019.
* Zhou et al., *Instruction-Following Evaluation for Large Language Models*, 2023.
* Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, 2023.
