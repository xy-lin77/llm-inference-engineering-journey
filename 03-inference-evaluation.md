# LLM推理性能核心指标

## 1. 延迟（Latency）
- TTFT（首字延迟）：输入到输出第一个token耗时，影响流利感，受Prefill影响
- TPOT（逐字延迟）：平均每生成1个token耗时，影响连贯性

## 2. 吞吐量（Throughput）
- 公式：Throughput = 总生成token数 / 总耗时
- 关键：Continuous Batching提升效率，可略增单请求延迟

## 3. 显存占用
- 模型参数：FP16=参数量×2字节，INT4可大幅缩减
- KV Cache：动态占用，受对话长度、Batch Size影响
- CUDA上下文：会占用部分显存，属于推理时的基础显存开销
- 激活值：推理时占用较小
