# Attention 时间复杂度与 KV Cache

## 1. Attention 时间复杂度对比

| 阶段 | 模式 | 单步复杂度 (Per-token) | 总体复杂度 (Total) | 核心原因 |
| :--- | :--- | :--- | :--- | :--- |
| **训练/预填充** | 标准 Self-Attention | $O(n^2 d)$ | $O(n^3 d)$ | 需要计算全量 $Q \times K^T$ 矩阵 ($n \times n$) |
| **推理生成** | 开启 KV Cache | **$O(n d)$** | **$O(n^2 d)$** | 仅需计算当前词与历史 K/V 的注意力，避免重复计算，空间换时间 |

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

---

## 3. Transformers 库源码实现关键

### 核心方法：`prepare_inputs_for_generation`
```python
def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
    # 如果存在缓存，说明非首词生成阶段
    if past_key_values is not None:
        # 【关键：裁剪】仅传入最后一个 token，形状变为 [batch, 1]
        input_ids = input_ids[:, -1:]
    
    return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}
```

### 核心逻辑：Attention.forward 拼接
```python
def forward(self, hidden_states, past_key_value=None, use_cache=False):
    # 1. 计算当前 token 的 K, V
    key_states, value_states = self.k_proj(hidden_states), self.v_proj(hidden_states)

    # 2. 【关键：拼接】在序列维度 (seq_len) 拼接历史缓存
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # 3. 更新并传回接力棒
    present_key_value = (key_states, value_states) if use_cache else None
    return attn_output, present_key_value
```
