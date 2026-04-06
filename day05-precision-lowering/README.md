# Quick Start

## 说明

- `infer_base.py`：基于 **Qwen-7B 全精度（FP32）** 的最小推理实现脚本，保留用于理解最基础的大模型推理流程（加载模型 → 构造输入 → generate）。
- `infer_compare.py`：在 `infer_base.py` 基础上加入 **参数解析（argparse）和控制流**，用于对比不同模型规模（7B / 72B）和不同精度（FP32 / BF16 / AMP）的实验脚本。

---

## Qwen 7B

### FP32
```bash
python infer_compare.py --model qwen7b --precision fp32
```

### BF16

```bash
python infer_compare.py --model qwen7b --precision bf16
```

### 混合精度（AMP）

```bash
python infer_compare.py --model qwen7b --precision amp
```

---

## Qwen 72B

### FP32（2张H800显存不够）

```bash
python infer_compare.py --model qwen72b --precision fp32
```

### BF16

```bash
python infer_compare.py --model qwen72b --precision bf16
```

### 混合精度（AMP）

```bash
python infer_compare.py --model qwen72b --precision amp
```

# 降低浮点精度

## 关键参数与方法

- **`device_map="auto"`（accelerate）**：自动根据 GPU/CPU 内存情况将模型按层切分并分配到不同设备上，实现大模型的多卡与异构设备部署。

- **`model.to(torch.bfloat16)`（BF16）**：将模型权重转换为 bfloat16，显著降低显存占用并提升吞吐，同时相比 FP16 具有更好的数值稳定性（更大指数范围）。

- **`torch.amp.autocast()`（AMP）**：在运行时按算子自动选择 **BF16** 或 **FP32**，实现性能与数值稳定性的平衡；QKV、FFN 矩阵乘法使用 BF16，LayerNorm、Softmax、RoPE 等数值敏感算子保留 FP32。

- **注**：AMP 只优化计算精度而不降低模型权重显存占用，因此大模型推理必须先用 **BF16** 加载权重，再配合 AMP 才能同时避免 OOM 并获得性能与稳定性的平衡。

# 总结

| 方法 | 显存 | 速度 | 精度 |
|------|------|------|------|
| FP32 | 1x | 1x | 基准 |
| BF16 | ~0.5x | 1.2–1.8x | 理论无损 |
| BF16 + AMP | ~0.5x | 在 BF16 的基础上再 +10–30% | 理论无损 |