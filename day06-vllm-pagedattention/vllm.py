import time
import torch
from vllm import LLM, SamplingParams

PROMPTS_AS_MESSAGES = [
    [{"role": "user", "content": "你好，请介绍一下你自己。"}],
    [{"role": "user", "content": "写一个关于一只勇敢的小猫去外太空探险的童话故事。"}],
    [{"role": "user", "content": "请解释一下什么是人工智能，并列举三个实际应用案例。"}],
    [{"role": "user", "content": "Translate the following English sentence to French: 'The quick brown fox jumps over the lazy dog.'"}],
    [{"role": "user", "content": "编写一个 Python 函数，用于计算斐波那契数列的第 n 项。"}],
    [{"role": "user", "content": "宇宙的起源是什么？"}],
    [{"role": "user", "content": "给我讲个笑话吧。"}],
    [{"role": "user", "content": "介绍一下 vLLM 的 PagedAttention 机制是如何工作的，它解决了什么问题？"}],
]

llm = LLM(
    model="Qwen/Qwen2-72B-Instruct",
    dtype=torch.bfloat16,
    tensor_parallel_size=2,  # 2张 H800 并行加载 72B
    trust_remote_code=True,
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
outputs_vllm = llm.generate(formatted_prompts, sampling_params)
