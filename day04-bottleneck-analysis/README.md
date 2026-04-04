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
