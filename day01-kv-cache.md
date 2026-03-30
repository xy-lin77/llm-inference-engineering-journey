# Attention 时间复杂度总结

## 1. 标准 Self-Attention

- 单步：**O(n² d)**
- 总体：**O(n³ d)**
- 原因：QKᵀ 计算 → n × n

## 2. 推理（KV Cache）

- 单步：**O(n d)**
- 总体：**O(n² d)**
- 原因：避免重复计算 K/V

# KV Cache 显存占用（推理阶段）

显存 ≈ 2 × L × d × n × B × bytes

其中：

- L：Transformer 层数（num_layers）
- d：隐藏层维度（hidden_size）
- n：序列长度（sequence length）
- B：批次大小（batch size）
- bytes：数据类型字节数（如 FP16 = 2 bytes）

说明：

- “2” 表示 Key 和 Value 两份缓存
- 该公式仅估算 KV cache，不包含模型参数和中间激活
