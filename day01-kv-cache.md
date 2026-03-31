# Attention 时间复杂度与 KV Cache 总结

## 1. Attention 时间复杂度对比

| 阶段 | 模式 | 单步复杂度 (Per-token) | 总体复杂度 (Total) | 核心原因 |
| :--- | :--- | :--- | :--- | :--- |
| **训练/预填充** | 标准 Self-Attention | $O(n^2 d)$ | $O(n^3 d)$ | 需要计算全量 $Q \times K^T$ 矩阵 ($n \times n$) |
| **推理生成** | 开启 KV Cache | **$O(n d)$** | **$O(n^2 d)$** | 仅需计算当前词与历史 K/V 的注意力，避免重复计算 |

---

## 2. KV Cache 显存占用（推理阶段）

### 估算公式
$$显存 \approx 2 \times L \times d \times n \times B \times \text{bytes}$$

### 参数说明
* **2**：代表 Key 和 Value 两份独立的缓存。
* **L**：Transformer 层数 (`num_layers`)。
* **d**：隐藏层维度 (`hidden_size`)。
* **n**：序列长度 (`sequence_length`)。
* **B**：批次大小 (`batch_size`)。
* **bytes**：数据类型字节数（如 FP16 = 2 bytes, INT8 = 1 byte）。

> **说明**：该公式仅用于估算 **KV Cache** 动态占用的显存，不包含模型参数（Weights）以及推理过程中的中间激活值（Activations）。
