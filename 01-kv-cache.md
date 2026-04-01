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
# 摘自 transformers 源码简化逻辑
def forward(self, hidden_states, past_key_value=None, use_cache=False, ...):
    # 1. 计算当前时刻的 K, V
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # 2. 获取历史缓存（如果存在）
    if past_key_value is not None:
        # 这里的 key_states 是当前 token 的，past_key_value 是历史所有 token 的
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

    # 3. 计算 Attention 时，使用了包含历史信息的完整 key_states/value_states
    attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
    
    # 4. 返回当前层更新后的 KV Cache 供下一轮推理使用
    return attn_output, past_key_value if use_cache else None
```

为了支持更复杂的解码（如 PagedAttention），新版本 HuggingFace 引入了 DynamicCache 对象，它封装了拼接逻辑，不是直接cat
