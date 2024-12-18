

# 本地微调: 使用MLX在MacOS上进行微调



## MLX Intro

MLX is a NumPy-like array framework designed for efficient and flexible machine learning on Apple silicon, brought to you by Apple machine learning research.

The Python API closely follows NumPy with a few exceptions. MLX also has a fully featured C++ API which closely follows the Python API.

The main differences between MLX and NumPy are:

> - **Composable function transformations**: MLX has composable function transformations for automatic differentiation, automatic vectorization, and computation graph optimization.
> - **Lazy computation**: Computations in MLX are lazy. Arrays are only materialized when needed.
> - **Multi-device**: Operations can run on any of the supported devices (CPU, GPU, …)

The design of MLX is inspired by frameworks like [PyTorch](https://pytorch.org/), [Jax](https://github.com/google/jax), and [ArrayFire](https://arrayfire.org/). **A notable difference from these frameworks and MLX is the unified memory model. Arrays in MLX live in shared memory. Operations on MLX arrays can be performed on any of the supported device types without performing data copies.** Currently supported device types are the CPU and GPU.



**非常适用于个人进行微调小模型、学习**



> ### MLX关键优势：
>
> - 数组位于共享内存
> - 跨设备操作无需数据传输
> - 显著降低数据移动开销
> - MLX内存模型更高效
>
> ### 对比NVIDIA GPU框架：
>
> - PyTorch需要手动.to(device)
> - 数据传输是性能瓶颈



## Finetuning

>  前提：
>
> 1、安装[Miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/macos.html#)
>
> 2、注册[huggingface账号](https://huggingface.co/)并在终端已经登录huggingface



### step 1、set up env 环境准备

- 创建虚拟环境

```shell
# 创建虚拟环境
conda create --name finetuning python=3.11 -y
#	激活环境
conda activate finetuning
# 终止环境
conda deactivate 

# 下载项目mlx-examples，使用
mkdir mlx
cd mlx
git clone https://github.com/ml-explore/mlx-examples.git
cd mlx-examples/lora
pip install -r requirements.txt
pip install mlx-lm
```



> ## LoRA (Low-Rank Adaptation)
>
> LoRA（低秩适应）是一种高效的参数微调技术，主要解决以下问题：
>
> 1. 减少微调时的计算和存储开销
> 2. 保持预训练模型的整体性能
> 3. 只训练少量可学习参数
>
> ### 核心原理
>
> 传统微调会更新模型的所有参数，而LoRA只在原始预训练权重上附加小型可训练矩阵。
>
> 数学表示：
>
> - 原始权重矩阵：W
> - 可学习的低秩矩阵：A和B
> - 更新后的权重：W + BA
>
> ### LoRA vs QLoRA
>
> - LoRA：低秩矩阵微调
> - QLoRA：在LoRA基础上加入量化技术，进一步降低内存消耗



> To install from PyPI you must meet the following requirements:
>
> [Build and Install](https://ml-explore.github.io/mlx/build/html/install.html)
>
> - Using an M series chip (Apple silicon)
> - Using a native Python >= 3.9
> - macOS >= 13.5



> 我的系统环境：
>
> Apple M3 Max、36 GB、14.6.1 



### step 2、download model 模型下载模型

```
pip install hf_transfer
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
# 查看下载的镜像
ls -la /Users/${username}/.cache/huggingface/hub 
```





> 国内镜像加速
>
> export HF_ENDPOINT=https://hf-mirror.com



> [Qwen2.5-Intro](https://github.com/QwenLM/Qwen2.5?tab=readme-ov-file)
>
> In the past three months since Qwen2's release, numerous developers have built new models on the Qwen2 language models, providing us with valuable feedback. During this period, we have focused on creating smarter and more knowledgeable language models. Today, we are excited to introduce the latest addition to the Qwen family: **Qwen2.5**.
>
> - Dense, easy-to-use, decoder-only language models, available in **0.5B**, **1.5B**, **3B**, **7B**, **14B**, **32B**, and **72B** sizes, and base and instruct variants.
> - Pretrained on our latest large-scale dataset, encompassing up to **18T** tokens.
> - Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON.
> - More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.
> - Context length support up to **128K** tokens and can generate up to **8K** tokens.
> - Multilingual support for over **29** languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.



### step 3、data preparation 数据准备



- [mlx数据要求](https://github.com/ml-explore/mlx-examples/blob/69181e00584976f717c27f90c7e009e70cc1b0bf/llms/mlx_lm/LORA.md#data)

```
The LoRA command expects you to provide a dataset with --data. The MLX Examples GitHub repo has an example of the WikiSQL data in the correct format.

For fine-tuning (--train), the data loader expects a train.jsonl and a valid.jsonl to be in the data directory. For evaluation (--test), the data loader expects a test.jsonl in the data directory.

Currently, *.jsonl files support three data formats: chat, completions, and text. Here are three examples of these formats:
```



- mlx支持的三个格式数据

`chat`:

```
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello."
    },
    {
      "role": "assistant",
      "content": "How can I assistant you today."
    }
  ]
}
```

`completions`:

```
{
  "prompt": "What is the capital of France?",
  "completion": "Paris."
}
```

`text`:

```
{
  "text": "This is an example for the model."
}
```



- 下载sql数据集：[b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)

```
# 安装依赖性
pip install datasets

touch load_data.py

vim load_data.py
```



`load_data.py` 使用 `chat`类型的数据

```py
from datasets import load_dataset
folder="my-data-chat/" 
system_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""
def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": system_message.format(schema=sample["context"])},
      {"role": "user", "content": sample["question"]},
      {"role": "assistant", "content": sample["answer"]}
    ]
  }
dataset = load_dataset("b-mc2/sql-create-context", split="train")
dataset = dataset.shuffle().select(range(150))
dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
dataset = dataset.train_test_split(test_size=50/150)
dataset_test_valid = dataset['test'].train_test_split(0.5)
print(dataset["train"][45]["messages"])
dataset["train"].to_json(folder + "train.jsonl", orient="records")
dataset_test_valid["train"].to_json(folder + "test.jsonl", orient="records")
dataset_test_valid["test"].to_json(folder + "valid.jsonl", orient="records")
```



### step 4、training 训练

- 验证未训练的模型生成结果 LM Studio

`prompt`

```
You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
CREATE TABLE table_13505192_3 (series_number INTEGER, season_number VARCHAR)
User:What is the series number for season episode 24?
Assistant:
```



- training

```
mlx_lm.lora --help

mlx_lm.lora --train --model Qwen/Qwen2.5-0.5B-Instruct --batch-size 1 --num-layers 4 --iters 1000 --data my-data-chat
```



### step 5、fuse & evaluation 模型合并和评估



- evaluation 评估微调后的模型

```
# mlx_lm.lora  --model Qwen/Qwen2.5-0.5B-Instruct --data my-data-chat --adapter-path adapters --test
Loading pretrained model
Fetching 7 files: 100%|████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 18712.64it/s]
Loading datasets
Testing
Test loss 0.651, Test ppl 1.917.
```



> Test loss:
>
> 损失值越低越好，表示模型预测与实际结果的偏差程度
>
> 0.651 是一个相对较低的损失值，说明模型在测试集上表现不错
>
>  困惑度（Test ppl/Perplexity）: 
>
> - 困惑度是衡量语言模型性能的重要指标
> - 越接近1越好，1.917 是一个非常好的困惑度
> - 解释：模型对测试数据的预测相当准确



- fuse 融合模型

```
mlx_lm.fuse --help

mlx_lm.fuse \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter-path adapters \
    --save-path Qwen2.5-0.5B-Instruct-v1-new
    
    
mlx_lm.generate  --model Qwen2.5-0.5B-Instruct-v1-new \
     --prompt "You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_13505192_3 (series_number INTEGER, season_number VARCHAR)\nUser:What is the series number for season episode 24?\nAssistant:"


mlx_lm.generate  --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompt "You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_13505192_3 (series_number INTEGER, season_number VARCHAR)\nUser:What is the series number for season episode 24?\nAssistant:"
```



## LM Studio加载模型



[下载地址lmstudio](https://lmstudio.ai/)



```
You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
CREATE TABLE table_13505192_3 (series_number INTEGER, season_number VARCHAR)
User:What is the series number for season episode 24?
Assistant:
```





## Reference

[Fine-Tuning with LoRA or QLoRA](https://github.com/ml-explore/mlx-examples/blob/69181e00584976f717c27f90c7e009e70cc1b0bf/llms/mlx_lm/LORA.md#data)

[fine-tuning-on-a-mac-with-mlx/](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/)

[Fine-tuning LLMs with Apple MLX locally](https://heidloff.net/article/apple-mlx-fine-tuning)

[lmstudio](https://lmstudio.ai/)

[Qwen2.5](https://github.com/QwenLM/Qwen2.5)
