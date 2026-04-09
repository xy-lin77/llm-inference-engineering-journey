# LLM推理性能核心指标

## 1. 延迟（Latency）
- TTFT（首字延迟）：预填充阶段 Prefill Phase，处理输入 Prompt 到输出第一个 token 耗时
- TPOT（逐字延迟）：解码阶段 Decoding Phase，平均每生成1个 token 耗时

---

## 2. 吞吐量（Throughput）
- 公式：Throughput = 总生成token数 / 总耗时
- 单位：TPS (Tokens Per Second)
- 关键：Continuous Batching 提升效率，可略增单请求延迟

---

## 3. 显存占用
- 模型参数：静态占用，FP16=参数量×2字节，INT4可大幅缩减
- KV Cache：动态占用，受对话长度、Batch Size影响
- CUDA上下文：会占用部分显存，属于推理时的基础显存开销
- 激活值：推理时占用较小
