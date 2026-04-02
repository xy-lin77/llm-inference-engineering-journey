# 大模型推理性能瓶颈分析

对比两种典型场景下大模型推理的性能瓶颈，帮助理解瓶颈如何随硬件、模型和负载类型动态变化。

## 核心结论

- **低并发、关注延迟**的场景，瓶颈通常是**显存带宽 (Memory-Bound)**。
- **高并发、关注吞吐量**的场景，瓶颈通常是**计算能力 (Compute-Bound)** 和 **调度效率**。

---

## 场景对比分析表

| 对比维度 | 场景一：消费级部署 | 场景二：生产级服务 |
| :--- | :--- | :--- |
| **场景描述** | 单个用户与模型进行交互式对话 | 不同用户频繁请求，系统需同时处理 |
| **硬件配置** | **NVIDIA RTX 4070S** (12GB GDDR6X VRAM) | **NVIDIA H800** (80GB HBM3 VRAM) |
| **模型配置** | **Qwen-7B** (4-bit 量化后约 4GB) | **Llama-2-70B** (FP16 约 140GB, 需 2x H800) |
| **负载类型** | 单用户、低并发 (Batch Size = 1) | 多用户、高并发 (Batch Size > 16) |
| **核心优化目标** | **低延迟 (Latency)**：尽快返回第一个 token | **高吞吐量 (Throughput)**：每秒处理更多请求 |
| **主要时间瓶颈** | **显存带宽 (Memory Bandwidth)** | **计算能力 (Compute Power)，请求调度与批处理效率** |
| **瓶颈表现** | - 生成每个 token 的速度较慢 (tokens/sec 不高)<br>- GPU 计算单元利用率可能不高，处于“等待数据”状态 | - 系统总吞吐量达到上限<br>- GPU 利用率持续处于高位 (90%+) <br>- 用户平均延迟随并发数增加而上升 |
| **`torch.profiler`** | - `gemm` (矩阵乘法) 耗时占比可观，但不是绝对主导 (如 40-50%)<br>- `dequantize`, `aten::copy_`, `rms_norm` 等大量小 kernel 耗时总和很高 | - `gemm` 耗时占比**极高** (如 >70%)，表明 GPU 在全力计算<br>- 若调度不佳，GPU 时间线上会出现“空洞”，等待 CPU 派发任务 |
| **关键优化策略** | - **Flash Attention**: 减少 Attention 计算中的显存读写<br>- **Kernel Fusion (`torch.compile`)**: 合并小操作，减少访存次数 | - **Continuous Batching**: 动态批处理，避免请求间互相等待<br>- **PagedAttention (vLLM)**: 高效管理 KV Cache，解决显存碎片<br>- **Tensor Parallelism**: 使用多卡并行计算，分担计算压力 |
