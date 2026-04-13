from fastapi import FastAPI, StreamingResponse
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
import asyncio

# FastAPI 应用初始化
app = FastAPI(title="Qwen2-72B-Instruct 推理服务")

# --------------------------
# Qwen-72B 生产级模型配置（核心）
# --------------------------
llm = LLM(
    model="Qwen/Qwen2-72B-Instruct",
    trust_remote_code=True,
    tensor_parallel_size=2,
    # 显存利用率：建议 0.8~0.9（平衡性能和稳定性）
    gpu_memory_utilization=0.85,
    # 上下文长度：Qwen 72B 原生支持 8192
    max_model_len=8192,
    # 量化配置（可选，4bit/8bit，需 GPU 支持）
    quantization="4bit",  # 注释掉则为 FP16/BF16 全精度
    # 其他优化（生产必开）
    enforce_eager=False,  # 启用 CUDA graph，提升吞吐量
    disable_log_stats=False,  # 开启性能监控
    swap_space=4,  # CPU 交换空间（单位：GB，应对峰值显存）
)

# 请求参数校验
class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="输入提示词")
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="最大生成长度")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="随机性")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="采样阈值")

# --------------------------
# 1. 非流式接口 /generate
# --------------------------
@app.post("/generate")
async def generate(req: InferenceRequest):
    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop_token_ids=[151643],  # Qwen 专属 EOS token
        skip_special_tokens=True,
    )
    # vLLM 同步生成，FastAPI 自动线程池调度
    outputs = llm.generate(req.prompt, sampling_params)
    response = outputs[0].outputs[0].text
    return {
        "prompt": req.prompt,
        "response": response,
        "finish_reason": outputs[0].outputs[0].finish_reason
    }

# --------------------------
# 2. 流式接口 /stream（SSE 标准）
# --------------------------
async def stream_generator(req: InferenceRequest):
    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop_token_ids=[151643],
        skip_special_tokens=True,
        stream=True,
    )
    generator = llm.generate_stream(req.prompt, sampling_params)
    for output in generator:
        if output.outputs[0].text:
            yield f"data: {output.outputs[0].text}\n\n"
        await asyncio.sleep(0)  # 让出事件循环，避免阻塞

@app.post("/stream")
async def stream(req: InferenceRequest):
    return StreamingResponse(
        stream_generator(req),
        media_type="text/event-stream"
    )
