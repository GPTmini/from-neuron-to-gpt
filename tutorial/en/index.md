# 前言

![banner.png](banner.png)

一套从单个神经元出发，只用 NumPy 一步步手搓出 GPT 的中文教程。

不依赖 PyTorch、TensorFlow 等深度学习框架：自动求导、网络层、优化器、注意力、Transformer、训练、推理、微调，全部从零实现，并配有逐步讲解的 Jupyter Notebook。

* 代码仓库：[https://github.com/gptmini/from-neuron-to-gpt](https://github.com/gptmini/from-neuron-to-gpt)
* 许可证：[MIT License](LICENSE)

## 特点

* **零框架依赖**：只用 NumPy，从张量、计算图、反向传播开始，每一行代码都能读懂；
* **循序渐进**：从 1 个神经元，到 GPT、监督微调（SFT）和直接偏好优化（DPO），每章只引入少量新概念；
* **讲解与代码同在**：每章都是一个可以独立运行的 Notebook，多数章节末尾附有课后练习；
* **纯 CPU 运行**：不需要 GPU，默认的 GPT 模型约 10.8M 参数，一台笔记本即可完成训练。

## 教程目录

教程分为两部分，共 34 章，外加一篇框架总结。**两部分的章节编号各自独立**：第一部分是 01–16，第二部分是 01–18。

### 第一部分：神经网络基础

目录：`tutorial/zh/artificial-neural-network`

| 目录                          | 章节    | 内容                                              |
|-----------------------------|-------|-------------------------------------------------|
| `01-multilayer-perceptron`  | 01–10 | 线性回归、损失函数、梯度下降、自动微分、学习率、迭代、批次、轮次、隐藏层、激活函数 |
| `02-recurrent-neural-network` | 11–16 | 编码、嵌入、交叉熵损失、循环网络、Adam 优化器、学习率调度器                |
| `03-core-framework`         | 总结    | 汇总前两部分，形成完整的核心深度学习框架（`src/core`）                |

### 第二部分：大语言模型基础

目录：`tutorial/zh/large-language-model`

| 目录                | 章节    | 内容                                  |
|-------------------|-------|-------------------------------------|
| `01-tokenizer`    | 01–02 | 词元嵌入、位置嵌入                           |
| `02-attention`    | 03–06 | 自注意力、因果注意力、加权注意力、多头注意力              |
| `03-pre-training` | 07–10 | 残差连接、归一化、随机失活、模型持久化                 |
| `04-inference`    | 11–14 | 采样策略（温度、Top-k、Top-p）、停止条件、KV 缓存、批量推理 |
| `05-fine-tuning`  | 15–18 | SFT 数据准备与训练、DPO 数据准备与训练             |

## 快速开始

### 环境要求

* Python 3.10 及以上
* 依赖：`numpy`、`matplotlib`、`requests`、`jupyter-book`

### 安装

```bash
git clone https://github.com/gptmini/from-neuron-to-gpt.git
cd from-neuron-to-gpt

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install jupyterlab
```

### 数据

教程使用莎士比亚剧本 tinyshakespeare 作为训练数据，保存位置是 `data/tinyshakespeare.txt`。

### 运行教程

按编号顺序打开 `tutorial/zh` 下的 Notebook 即可。

### 用命令行运行完整流程

如果只想训练出模型，不想逐章阅读，可以在项目根目录依次运行三个脚本：

```bash
python -m src.gpt.gpt   # 预训练：下载数据、训练 GPT、生成示例文本
python -m src.sft.sft   # 监督微调：准备 SFT 样本、训练、生成示例文本
python -m src.dpo.dpo   # 直接偏好优化：准备 DPO 样本、训练、生成示例文本
```

三个阶段有依赖关系，必须**按顺序**运行：

| 阶段     | 依赖                          | 产出（均在 `data/` 目录）                           |
|--------|-----------------------------|---------------------------------------------|
| 预训练    | `tinyshakespeare.txt`       | `tinyshakespeare-gpt.npz`                   |
| SFT    | 预训练模型                       | `sft-samples.pkl`、`tinyshakespeare-sft.npz` |
| DPO    | SFT 模型，数据准备阶段还会用预训练模型生成拒绝样本 | `dpo-samples.pkl`、`tinyshakespeare-dpo.npz` |

Notebook 里的第二部分同样遵循这个顺序：`03-pre-training/10` 生成预训练模型，`05-fine-tuning` 依次使用它。

## 模型规模

默认的 GPT 模型配置如下（对应第二部分“模型持久化”章节，以及 `src/gpt/gpt.py`）：

| 参数              |               值 |
|-----------------|----------------:|
| 上下文尺寸           |             256 |
| 嵌入尺寸            |             384 |
| 注意力头数           |               6 |
| Transformer 模块数 |               6 |
| 批大小             |              16 |
| 训练轮次            |              10 |
| 词表（字符级）         |              65 |
| 参数量             | 约 1080 万（10.8M） |

此前的章节为了方便快速运行，使用的是更小的配置；到“模型持久化”章节才换成上表的配置，目的是训练一个在推理和微调阶段可用的模型。模型完全在 CPU 上训练，规模较小，目的是看清每一步的原理，而不是追求生成质量。

### 运行环境与耗时

全部代码基于 Mac mini M4 / 16GB 开发完成，完全运行在 CPU 上。

* 预训练（默认配置）：约 1–2 小时；
* 微调阶段中，DPO 的数据准备需要逐条采样生成拒绝样本，通常是耗时最长的一步。

耗时与 CPU 性能、核数以及 NumPy 所链接的 BLAS 库有关，不同机器上的差别可能很大。

## 项目结构

```text
.
├── data/                       # 训练数据（tinyshakespeare）及训练产生的模型参数、样本文件
├── src/
│   ├── core/                   # 核心框架：张量、网络层、激活函数、损失函数、优化器、
│   │                           #         学习率调度器、数据集、训练器
│   ├── gpt/                    # GPT 模型、数据集、损失函数、训练器；gpt.py 为预训练脚本
│   ├── sft/                    # 监督微调；sft.py 为微调脚本
│   └── dpo/                    # 直接偏好优化；dpo.py 为优化脚本
├── tutorial/zh/                # 中文教程（Jupyter Notebook）
│   ├── artificial-neural-network/
│   └── large-language-model/
├── myst.yml                    # 教程站点（MyST / Jupyter Book）配置
├── requirements.txt
└── LICENSE
```

## 常见问题

**`ModuleNotFoundError: No module named 'src'`**
没有把项目根目录加入 `PYTHONPATH`，或者设置之后没有重启 Jupyter。请按“运行教程”一节设置，并在**同一个终端**里启动 `jupyter lab`。

**`FileNotFoundError: ... tinyshakespeare.txt`**
数据文件还没有准备好。请参考“准备数据”一节。

**运行很慢**
所有代码都是纯 NumPy 实现并运行在 CPU 上，这是预期行为。可以先把轮次、批大小或模型规模调小，确认流程跑通后再使用默认配置。

## 致谢与参考

* 训练数据 tinyshakespeare 来自 Andrej Karpathy 的 [char-rnn](https://github.com/karpathy/char-rnn) 项目；
* Vaswani 等，*Attention Is All You Need*，2017；
* Radford 等，*Language Models are Unsupervised Multitask Learners*（GPT-2），2019；
* Kingma 和 Ba，*Adam: A Method for Stochastic Optimization*，2014；
* Loshchilov 和 Hutter，*Decoupled Weight Decay Regularization*（AdamW），2017；
* Rafailov 等，*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*，2023。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。