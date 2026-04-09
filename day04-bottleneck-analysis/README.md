# 大模型推理性能瓶颈分析

对比两种典型场景下大模型推理的性能瓶颈，帮助理解瓶颈如何随硬件、模型和负载类型动态变化。

## 1. 核心结论

- **低并发、关注延迟**的场景，瓶颈通常是**显存带宽 (Memory-Bound)**。
- **高并发、关注吞吐量**的场景，瓶颈通常是**计算能力 (Compute-Bound)** 和 **调度效率**。

## 2. 场景对比分析表

| 对比维度 | 场景一：消费级部署 | 场景二：生产级服务 |
| :--- | :--- | :--- |
| **场景描述** | 单个用户与模型进行交互式对话 | 不同用户频繁请求，系统需同时处理 |
| **硬件配置** | **NVIDIA RTX 4070S** (12GB GDDR6X VRAM) | **NVIDIA H800** (80GB HBM3 VRAM) |
| **模型配置** | **Qwen-7B** (4-bit 量化后约 4GB) | **Qwen-72B** (**BF16** 约 140GB, 需 2x H800) |
| **负载类型** | 单用户、低并发 (Batch Size = 1) | 多用户、高并发 (Batch Size > 16) |
| **核心优化目标** | **低延迟 (Latency)**：尽快返回第一个 token | **高吞吐量 (Throughput)**：每秒处理更多请求 |
| **主要时间瓶颈** | **显存带宽 (Memory Bandwidth)** | **计算能力 (Compute Power)，请求调度与批处理效率** |
| **瓶颈表现** | - 生成每个 token 的速度较慢 (tokens/sec 不高)<br>- GPU 计算单元利用率可能不高，处于“等待数据”状态 | - 系统总吞吐量达到上限<br>- GPU 利用率持续处于高位 (90%+) <br>- 用户平均延迟随并发数增加而上升 |
| **`torch.profiler`** | **总耗时: ~30ms/token**<br/><br/>1. **Python 层**: `QwenMLP.forward` (50%, 15ms), `QwenAttention.forward` (40%, 12ms)<br/>2. **算子层 (ATen)**: `aten::addmm` (40%, 12ms), `aten::silu` (10%, 3ms), `aten::copy_` (10%, 3ms), 其他小算子 (40%, 12ms)<br/>3. **GPU 核函数层**: `cutlass_..._gemm` (40%, 12ms), `dequantize_row_nf4` (15%, 4.5ms), `fast_silu_...` (10%, 3ms), `rms_norm_kernel` (10%, 3ms), 其他小 kernel 众多<br/>4. **CPU 活动**: CPU 基本空闲，仅快速发起 `cudaLaunchKernel` | **总耗时: ~80ms/batch_step**<br/><br/>1. **Python 层**: `QwenMLP.forward` (60%, 48ms), `QwenAttention.forward` (30%, 24ms)<br/>2. **算子层 (ATen)**: `aten::bmm` 或 `aten::addmm` **(75%, 60ms)**, `aten::_scaled_dot_product...` (15%, 12ms), 其他算子 (10%, 8ms)<br/>3. **GPU 核函数层**: `cutlass_..._gemm` **(75%, 60ms)**, `paged_attention_v1_kernel` (15%, 12ms), 其他 kernel 占比很小<br/>4. **CPU 活动**: CPU 持续繁忙，进行请求调度和批处理组合。若调度不佳，GPU 时间线会出现“空洞” |
| **关键优化策略** | - **Flash Attention**: 减少 Attention 计算中的显存读写<br>- **Kernel Fusion (`torch.compile`)**: 合并小操作，减少访存次数 | - **Continuous Batching**: 动态批处理，避免请求间互相等待<br>- **PagedAttention (vLLM)**: 高效管理 KV Cache，解决显存碎片<br>- **Tensor Parallelism**: 使用多卡并行计算，分担计算压力 |
