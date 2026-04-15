import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

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

model_name = "Qwen/Qwen2-72B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# bitsandbytes INT8 量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True,
)

formatted_prompts = [
    tokenizer.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True
    )
    for msg in PROMPTS_AS_MESSAGES
]

inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to("cuda")

start_time = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )

end_time = time.time()

responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, response in enumerate(responses):
    print(f"\n========== 问题 {i+1} ==========")
    print(f"用户：{PROMPTS_AS_MESSAGES[i][0]['content']}")
    print(f"助手：response.split('assistant\n')[-1].strip()")

print(f"\n生成耗时：{end_time - start_time:.2f}s")
print(f"显存占用：{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
