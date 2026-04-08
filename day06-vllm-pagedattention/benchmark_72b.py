import time
import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. 配置 ---
# 注意：Qwen-72B 模型非常大（约 144GB），首次下载会非常耗时！
MODEL_ID = "Qwen/Qwen-72B-Chat"
DTYPE = "bfloat16"  # H800/A100 等现代 GPU 的最佳选择

PROMPTS = [
    "你好，请介绍一下你自己。",
    "写一个关于一只勇敢的小猫去外太空探险的童话故事。",
    "请解释一下什么是人工智能，并列举三个实际应用案例。",
    "Translate the following English sentence to French: 'The quick brown fox jumps over the lazy dog.'",
    "编写一个 Python 函数，用于计算斐波那契数列的第 n 项。",
    "宇宙的起源是什么？",
    "给我讲个笑话吧。",
    "介绍一下 vLLM 的 PagedAttention 机制是如何工作的，它解决了什么问题？",
]
MAX_TOKENS = 256

# --- 2. vLLM 推理测试 (使用 2 张 GPU) ---
print("--- 开始 vLLM 推理测试 (2x H800) ---")

# 加载 vLLM 模型，使用张量并行
# tensor_parallel_size=2 表示将模型切分到 2 张 GPU 上
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=2,
    trust_remote_code=True,
    dtype=DTYPE
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAX_TOKENS)

print("vLLM 开始生成...")
start_time_vllm = time.time()
outputs_vllm = llm.generate(PROMPTS, sampling_params)
end_time_vllm = time.time()
print("vLLM 生成完成。")

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


# --- 3. Transformers 推理测试 (使用 2 张 GPU) ---
print("--- 开始 Transformers 推理测试 (2x H800) ---")

# 加载 Transformers 模型和 Tokenizer
# device_map="auto" 会让 accelerate 自动将模型切分到所有可用 GPU
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Transformers 开始生成...")
start_time_hf = time.time()

inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True).to("cuda")
# 注意：这里可能会因为 padding 后的总 token 数过多而导致 OOM
try:
    outputs_hf_tokens = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
    end_time_hf = time.time()
    print("Transformers 生成完成。")

    duration_hf = end_time_hf - start_time_hf
    total_tokens_hf = sum(len(output) - len(prompt_input) for output, prompt_input in zip(outputs_hf_tokens, inputs.input_ids))
    throughput_hf = total_tokens_hf / duration_hf

    print(f"\n[Transformers 结果]")
    print(f"总耗时: {duration_hf:.2f} 秒")
    print(f"总生成 Token 数: {total_tokens_hf}")
    print(f"吞吐量 (Tokens/秒): {throughput_hf:.2f}")

except Exception as e:
    print(f"\n[Transformers 结果]")
    print(f"Transformers 推理失败! 错误: {e}")
    print("这很可能是因为预分配和 padding 导致了显存不足 (OOM)。")
    throughput_hf = 0

# --- 4. 最终对比 ---
print("\n--- 最终性能对比 ---")
print(f"vLLM 吞吐量: {throughput_vllm:.2f} Tokens/秒")
if throughput_hf > 0:
    print(f"Transformers 吞吐量: {throughput_hf:.2f} Tokens/秒")
    speedup = throughput_vllm / throughput_hf
    print(f"\n结论: vLLM 速度是 Transformers 的 {speedup:.2f} 倍。")
else:
    print("Transformers 未能成功运行。")
    print("\n结论: 在此场景下，vLLM 能够稳定运行，而标准 Transformers 因内存问题失败，展示了其在处理大模型时的巨大优势。")
