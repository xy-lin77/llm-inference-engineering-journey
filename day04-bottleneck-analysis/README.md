````md
# Quick Start

## Qwen 7B

### FP32
```bash
python infer_compare.py --model qwen7b --precision fp32
````

### 半精度（FP16）

```bash
python infer_compare.py --model qwen7b --precision half
```

### 混合精度（AMP）

```bash
python infer_compare.py --model qwen7b --precision amp
```

---

## Qwen 72B

### FP32

```bash
python infer_compare.py --model qwen72b --precision fp32
```

### 半精度（FP16）

```bash
python infer_compare.py --model qwen72b --precision half
```

### 混合精度（AMP）

```bash
python infer_compare.py --model qwen72b --precision amp
```
