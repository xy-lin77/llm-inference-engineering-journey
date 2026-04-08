import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# --- 1. 配置 ---
MODEL_ID = "Qwen/Qwen2-7B-Instruct"
DTYPE = torch.bfloat16

# 用于测试的对话式 Prompts
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
MAX_TOKENS = 256

# --- 2. Transformers 推理测试 ---
print("--- (1/3) 开始 Transformers 推理测试 ---")

# 加载 Tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
)

# 将对话格式的 prompts 转换为模型可以理解的单个字符串
# 这是进行批处理的标准做法
formatted_prompts = [
    tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
    for p in PROMPTS_AS_MESSAGES
]

print("Transformers 开始生成...")
start_time_hf = time.time()

# 对所有格式化后的 prompts 进行批处理
inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs_hf_tokens = model.generate(**inputs, max_new_tokens=MAX_TOKENS)

end_time_hf = time.time()
print("Transformers 生成完成。")

# 计算并打印 Transformers 性能
duration_hf = end_time_hf - start_time_hf
total_tokens_hf = sum(len(output) - len(prompt_input) for output, prompt_input in zip(outputs_hf_tokens, inputs.input_ids))
throughput_hf = total_tokens_hf / duration_hf

print(f"\n[Transformers 结果]")
print(f"总耗时: {duration_hf:.2f} 秒")
print(f"总生成 Token 数: {total_tokens_hf}")
print(f"吞吐量 (Tokens/秒): {throughput_hf:.2f}")

# 释放 Transformers 模型占用的显存
del model
del inputs
del outputs_hf_tokens
torch.cuda.empty_cache()
print("\n--- Transformers 测试结束，已释放显存 ---\n")


# --- 3. vLLM 推理测试 ---
print("--- (2/3) 开始 vLLM 推理测试 ---")

# 加载 vLLM 模型
llm = LLM(
    model=MODEL_ID,
    dtype=DTYPE,
    # 如果你有多个 GPU，可以设置 tensor_parallel_size=N
    # tensor_parallel_size=1,
)

# 定义采样参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=MAX_TOKENS)

print("vLLM 开始生成...")
start_time_vllm = time.time()
# vLLM 直接接收格式化后的字符串列表进行高效批处理
outputs_vllm = llm.generate(formatted_prompts, sampling_params)
end_time_vllm = time.time()
print("vLLM 生成完成。")

# 计算并打印 vLLM 性能
duration_vllm = end_time_vllm - start_time_vllm
total_tokens_vllm = sum(len(output.outputs[0].token_ids) for output in outputs_vllm)
throughput_vllm = total_tokens_vllm / duration_vllm

print(f"\n[vLLM 结果]")
print(f"总耗时: {duration_vllm:.2f} 秒")
print(f"总生成 Token 数: {total_tokens_vllm}")
print(f"吞吐量 (Tokens/秒): {throughput_vllm:.2f}")

del llm
torch.cuda.empty_cache()
print("\n--- vLLM 测试结束，已释放显存 ---\n")


# --- 4. 最终对比 ---
print("--- (3/3) 最终性能对比 ---")
print(f"Hugging Face Transformers 吞吐量: {throughput_hf:.2f} Tokens/秒")
print(f"vLLM 吞吐量: {throughput_vllm:.2f} Tokens/秒")
if throughput_hf > 0:
    speedup = throughput_vllm / throughput_hf
    print(f"\n结论: 在此批处理测试中，vLLM 的速度是 Transformers 的 {speedup:.2f} 倍。")
