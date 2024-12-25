- [MLX的生态](#mlx的生态)
  - [MLX参数调优](#mlx参数调优)
  - [community](#community)
    - [MLX community](#mlx-community)
    - [lmstudio-community](#lmstudio-community)
  - [性能对比](#性能对比)
    - [视频字幕提取对比](#视频字幕提取对比)
      - [MLX量化版: lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-4bit](#mlx量化版-lmstudio-communityqwen25-coder-32b-instruct-mlx-4bit)
      - [非MLX的量化版模型: qwen2.5-coder:32b](#非mlx的量化版模型-qwen25-coder32b)
    - [token生成性能评估](#token生成性能评估)
      - [MLX量化版: mlx-community/Llama-3.2-3B-Instruct-4bit](#mlx量化版-mlx-communityllama-32-3b-instruct-4bit)
      - [非MLX的GGUF版本Llama-3.2-3B-Instruct-Q4\_K\_M.gguf](#非mlx的gguf版本llama-32-3b-instruct-q4_k_mgguf)
  - [Reference](#reference)


# MLX的生态


## [MLX参数调优](https://github.com/ml-explore/mlx-examples/blob/69181e00584976f717c27f90c7e009e70cc1b0bf/llms/mlx_lm/LORA.md#memory-issues)

Fine-tuning a large model with LoRA requires a machine with a decent amount of memory. Here are some tips to reduce memory use should you need to do so:

1. Try quantization (QLoRA). You can use QLoRA by generating a quantized model with `convert.py` and the `-q` flag. See the [Setup](https://github.com/ml-explore/mlx-examples/blob/69181e00584976f717c27f90c7e009e70cc1b0bf/lora/README.md#setup) section for more details. 

   ```
   # mlx_lm.convert --help
   usage: mlx_lm.convert [-h] [--hf-path HF_PATH] [--mlx-path MLX_PATH] [-q] [--q-group-size Q_GROUP_SIZE] [--q-bits Q_BITS]
                         [--dtype {float16,bfloat16,float32}] [--upload-repo UPLOAD_REPO] [-d]
   
   Convert Hugging Face model to MLX format
   
   options:
     -h, --help            show this help message and exit
     --hf-path HF_PATH     Path to the Hugging Face model.
     --mlx-path MLX_PATH   Path to save the MLX model.
     -q, --quantize        Generate a quantized model.
     --q-group-size Q_GROUP_SIZE
                           Group size for quantization.
     --q-bits Q_BITS       Bits per weight for quantization.
     --dtype {float16,bfloat16,float32}
                           Type to save the non-quantized parameters.
     --upload-repo UPLOAD_REPO
                           The Hugging Face repo to upload the model to.
     -d, --dequantize      Dequantize a quantized model.
   ```

   > mlx_lm.convert 等同于 lora下的`convert.py`

2. Try using a smaller batch size with `--batch-size`. The default is `4` so setting this to `2` or `1` will reduce memory consumption. This may slow things down a little, but will also reduce the memory use.

3. Reduce the number of layers to fine-tune with `--lora-layers`. The default is `16`, so you can try `8` or `4`. This reduces the amount of memory needed for back propagation. It may also reduce the quality of the fine-tuned model if you are fine-tuning with a lot of data.

4. Longer examples require more memory. If it makes sense for your data, one thing you can do is break your examples into smaller sequences when making the `{train, valid, test}.jsonl` files.

5. Gradient checkpointing lets you trade-off memory use (less) for computation (more) by recomputing instead of storing intermediate values needed by the backward pass. You can use gradient checkpointing by passing the `--grad-checkpoint` flag. Gradient checkpointing will be more helpful for larger batch sizes or sequence lengths with smaller or quantized models.

> - **梯度检查点（Gradient Checkpointing）**：这是一种优化策略，允许你通过牺牲更多的计算时间来换取较少的内存使用。在训练深度神经网络时，每个前向传递（forward pass）都会生成许多中间激活值，这些值会被用在反向传播（backward pass）中计算梯度。通常，这些中间值会被存储在内存中，以便在反向传播时使用。
> - **内存和计算的权衡**：使用梯度检查点，你可以选择不存储所有的中间激活值，而是在反向传播时重新计算这些值。这意味着你需要更多的计算资源（因为要重复计算），但可以显著减少所需的内存。特别是在GPU内存有限的情况下，这个策略非常有用。
> - **使用方法**：要启用梯度检查点，你可以在训练命令中加入--grad-checkpoint标志。这个标志会告知训练框架采用这种策略。
> - **适用场景**：
>   - **较大的批量大小（Batch Sizes）**：当你处理较大的批量数据时，内存需求会增加。梯度检查点可以在这里发挥作用，减少内存使用。
>   - **较长的序列长度（Sequence Lengths）**：处理长序列数据（如长文本或时间序列）时，中间激活值的数量会显著增加，梯度检查点可以缓解这种情况下的内存压力。
>   - **较小的或量化的模型**：尽管模型本身较小或经过量化，处理大量数据时仍可能面临内存限制。梯度检查点在这里仍然有用，因为它能帮助管理内存使用。
>
> 总之，梯度检查点是一种有效的技术，特别适用于内存受限的情况或当你想在计算时间允许的条件下，尽可能减少内存使用。它通过重新计算而不是存储中间值来实现这一目标。



For example, for a machine with 32 GB the following should run reasonably fast:

```shell
python lora.py \
    --model mistralai/Mistral-7B-v0.1 \
    --train \
    --batch-size 1 \
    --lora-layers 4 \
    --data wikisql
```

上个视频的MLX微调命令：

```
mlx_lm.lora --train --model Qwen/Qwen2.5-0.5B-Instruct --batch-size 1 --num-layers 4 --iters 1000 --data my-data-chat
```



## community

### [MLX community](https://huggingface.co/mlx-community)

A community org for model weights compatible with [mlx-examples](https://github.com/ml-explore/mlx-examples) powered by [MLX](https://github.com/ml-explore/mlx).

These are pre-converted weights and ready to be used in the example scripts.



### [lmstudio-community](https://huggingface.co/lmstudio-community)

Models quantized and uploaded by the LM Studio community, for the LM Studio community. 



## 性能对比



### 视频字幕提取对比

[VideoLingo](https://github.com/Huanshere/VideoLingo)

#### MLX量化版: lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-4bit

- 第一次：8:24

- 第二次：8:40 

```shell
Compatibility: Apple Silicon Macs
Model creator: Qwen
Original model: Qwen2.5-Coder-32B-Instruct
MLX quantizations: provided by bartowski from mlx-examples
```



#### 非MLX的量化版模型: qwen2.5-coder:32b

- 第一次：9:10

- 第二次：8:56  

```shell
ollama show qwen2.5-coder:32b
  Model
    architecture        qwen2     
    parameters          32.8B     
    context length      32768     
    embedding length    5120      
    quantization        Q4_K_M    

  System
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.    

  License
    Apache License               
    Version 2.0, January 2004 
```



> [Which GGUF is right for me? (Opinionated)](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)
>
> Q4_K_M: 4.83 Bits
>
> 针对不同类型的量化版本，遵循特定的命名约定：“q”+ 用于存储权重的位数（精度）+ 特定变体



###  token生成性能评估



#### MLX量化版: mlx-community/Llama-3.2-3B-Instruct-4bit

```shell
# pip install -r mlx
# sh mlx_evaluation.sh
```



`mlx_evaluation.sh`

```shell
#!/bin/bash
mlx_lm.generate --model mlx-community/Llama-3.2-3B-Instruct-4bit --max-kv-size 33000 --max-tokens 1000 --temp 0.0 --top-p 0.9 --seed 1000 --prompt  -<./portugal.txt;say done
```



- ==0.20.0==

`pip install --upgrade mlx==0.20.0`



Prompt: 32160 tokens, 592.339 tokens-per-sec

Generation: 1000 tokens, 39.023 tokens-per-sec

Peak memory: 9.179 GB



Prompt: 32160 tokens, 588.907 tokens-per-sec

Generation: 1000 tokens, 39.181 tokens-per-sec

Peak memory: 9.179 GB



- ==0.21.1==

`pip install --upgrade mlx==0.21.1`



Prompt: 32160 tokens, 590.838 tokens-per-sec

Generation: 1000 tokens, 36.030 tokens-per-sec

Peak memory: 8.405 GB



Prompt: 32160 tokens, 613.386 tokens-per-sec

Generation: 1000 tokens, 36.393 tokens-per-sec

Peak memory: 8.405 GB



#### 非MLX的GGUF版本Llama-3.2-3B-Instruct-Q4_K_M.gguf

[llama. cpp安装](https://github.com/ggerganov/llama.cpp/blob/master/docs/install.md)



- q4_K_M on Llama.cpp with flash attention

`non_mlx_evaluation_gguf_flash_attention.sh`

```shell
#!/bin/bash
llama-cli -m /Users/{username}/.cache/lm-studio/models/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf -c 33000 -n 1000 --temp 0.0 --top_p 0.9 --seed 1000 -fa -f ./portugal.txt;say done
```

```shell
llama_perf_sampler_print:    sampling time =      79.32 ms / 33125 runs   (    0.00 ms per token, 417633.26 tokens per second)
llama_perf_context_print:        load time =   13690.74 ms
llama_perf_context_print: prompt eval time =   58325.58 ms / 32125 tokens (    1.82 ms per token,   550.79 tokens per second)
llama_perf_context_print:        eval time =   36068.92 ms /   999 runs   (   36.11 ms per token,    27.70 tokens per second)
llama_perf_context_print:       total time =   94615.82 ms / 33124 tokens
ggml_metal_free: deallocating
```

- q4_K_M on Llama.cpp without flash attention

`non_mlx_evaluation_gguf.sh`

```shell
#!/bin/bash
llama-cli -m /Users/{username}/.cache/lm-studio/models/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf -c 33000 -n 1000 --temp 0.0 --top_p 0.9 --seed 1000 -f ./portugal.txt;say done
```



```shell
llama_perf_sampler_print:    sampling time =      80.14 ms / 33125 runs   (    0.00 ms per token, 413318.53 tokens per second)
llama_perf_context_print:        load time =    2369.94 ms
llama_perf_context_print: prompt eval time =   82601.02 ms / 32125 tokens (    2.57 ms per token,   388.92 tokens per second)
llama_perf_context_print:        eval time =   72978.77 ms /   999 runs   (   73.05 ms per token,    13.69 tokens per second)
llama_perf_context_print:       total time =  155812.62 ms / 33124 tokens
ggml_metal_free: deallocating
```

> **Flash Attention** 是一种优化技术，专门用于提高注意力机制的计算效率。它通过减少内存访问次数和加速矩阵乘法来提升速度，特别是在大批量和长序列的情况下。





## Reference



[mlx_lm_0201_finally_has_the_comparable_speed_as](https://www.reddit.com/r/LocalLLaMA/comments/1h01719/mlx_lm_0201_finally_has_the_comparable_speed_as/)

