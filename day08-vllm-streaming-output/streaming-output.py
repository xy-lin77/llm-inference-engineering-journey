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
    tensor_parallel_size=2,
    trust_remote_code=True,
)

formatted_prompts = [
    llm.llm_engine.tokenizer.apply_chat_template(
        msg, 
        tokenize=False, 
        add_generation_prompt=True
    ) 
    for msg in PROMPTS_AS_MESSAGES
]

sampling_params = SamplingParams(
    temperature=0.7, 
    top_p=0.95,
    max_tokens=256
)

# 一次性传入所有 prompt，并发生成，流式返回
results = {}  # 存储每个请求的上一轮文本，用于计算增量
for output in llm.generate(formatted_prompts, sampling_params, stream=True):
    # output = RequestOutput（包含当前这个请求的最新状态）
    rid = output.request_id
    current_text = output.outputs[0].text

    if rid not in results:
        # 第一次输出：打印标题
        idx = int(rid.split("_")[-1])
        print(f"\n========== 问题 {idx+1} ==========")
        print(f"用户：{PROMPTS_AS_MESSAGES[idx][0]['content']}")
        print("助手：", end="", flush=True)
        results[rid] = ""

    # 增量流式输出
    delta = current_text[len(results[rid]):]
    print(delta, end="", flush=True)
    results[rid] = current_text
