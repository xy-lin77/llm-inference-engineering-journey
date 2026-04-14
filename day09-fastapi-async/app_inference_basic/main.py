from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import asyncio

# 初始化 FastAPI
app = FastAPI(title="vLLM + FastAPI 封装服务")

# 初始化 vLLM 模型（启动时加载一次）
# 本地模型路径 / HuggingFace 模型名
model_name = "Qwen/Qwen2-0.5B-Instruct"
llm = LLM(model=model_name, trust_remote_code=True)

# 请求参数校验（统一接口格式）
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

# ------------------------------
# 1. 非流式接口 /generate
# ------------------------------
@app.post("/generate")
async def generate(req: GenerateRequest):
    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature
    )
    
    # vLLM 生成（同步API，FastAPI会自动线程池运行）
    outputs = llm.generate(req.prompt, sampling_params)
    response_text = outputs[0].outputs[0].text
    
    return {
        "prompt": req.prompt,
        "response": response_text
    }

# ------------------------------
# 2. 流式接口 /stream
# ------------------------------
async def stream_generator(prompt: str, sampling_params: SamplingParams):
    # vLLM 流式生成器
    generator = llm.generate_stream(prompt, sampling_params)
    
    for output in generator:
        token = output.outputs[0].text
        yield f"data: {token}\n\n"  # SSE 标准格式
        await asyncio.sleep(0)  # 让出事件循环

@app.post("/stream")
async def stream(req: GenerateRequest):
    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature
    )
    
    return StreamingResponse(
        stream_generator(req.prompt, sampling_params),
        media_type="text/event-stream"
    )
