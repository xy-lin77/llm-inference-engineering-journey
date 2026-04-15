# 量化（FP32/BF16 → INT8/INT4）
## 1. 目标
通过线性映射将高精度浮点数（FP32/BF16）压缩到低比特整数域（INT8/INT4），以可控精度损失，换取显存降低、推理加速、部署成本下降。

| 类型 | 位宽 | 单参字节 | 核心收益 |
|------|------|----------|--------------------------|
| FP32 | 32bit | 4B       | 精度损失                 |
| BF16 | 16bit | 2B       | 平衡精度与显存           |
| INT8 | 8bit  | 1B       | 显存减半，GPU专项优化加速 |
| INT4 | 4bit  | 0.5B     | 显存骤降，需精度优化      |

---

## 2. 量化原理

### 1. 核心原理

| 类别 | 具体内容 |
|------|----------|
| 基本思想 | 通过尺度因子 scale 和零点 zero_point，建立浮点数与定点整数之间的双向映射 |
| 量化（Q）公式 | x_quant = round(x_float / scale) + zero_point |
| 反量化（D）公式 | x_float = (x_quant - zero_point) * scale |
| 互逆操作特性 | 量化 Q 和反量化 D 是一对互逆操作，满足 D(Q(x_float)) ≈ x_float |
| 对称量化（常用于权重） | zero_point = 0，适用于权重等分布相对对称的场景 |
| 非对称量化（常用于激活） | 需要 zero_point，适用于激活等分布不以0为中心的场景 |

设线性变换为 $y = W \cdot x$，量化：

$$W \approx s_w \cdot W_q, \quad x \approx s_x \cdot x_q$$

先做整数矩阵乘法，再乘回 scale，即反量化：

$$
W \cdot x \approx 
\underbrace{(s_w \cdot W_q) \cdot (s_x \cdot x_q) = s_w \cdot s_x \cdot (W_q \cdot x_q)}_{\text{标量等价对角矩阵} \ sI,  \text{标量-矩阵交换律}, \text{矩阵乘法结合律}}
$$

误差来自哪里？

以对称量化（zero_point = 0）为例：                        

$$x_q = \text{round}\left(\frac{x}{s_x}\right), \quad x \approx s_x \cdot x_q$$

其中 $s_x$ 是 scale（标量），$x_q$ 是整数张量，误差来自 round。

精确展开后：

$$W = s_w \cdot W_q + \varepsilon_w, \quad x = s_x \cdot x_q + \varepsilon_x$$

$$W \cdot x = s_w s_x (W_q x_q) + \underbrace{s_w W_q \varepsilon_x + \varepsilon_w s_x x_q + \varepsilon_w
\varepsilon_x}_{\text{量化误差项}}$$

量化精度越高（bit 越多），$\varepsilon$ 越小，近似越准。                                                           
  
双线性性 $(cA)(dB) = cd(AB)$

### 2. 权重量化

| 模型环节 | 量化方式及细节 |
|----------|----------------|
| 词嵌入层 | 仅量化存储，使用时反量化回浮点；使用方式为查表（index lookup）而非GEMM，反量化仅需对取出的行乘以scale，代价极低，属于纯存储压缩 |
| Q, K, V, O 投影层、<br>FFN 线性层 | 占模型参数的绝大部分，是量化的核心目标；常用量化<br>- per-tensor：整个矩阵共享一个scale，精度最差<br>- per-channel（per-row）：每行一个scale，精度较好，是主流<br>- group-wise/per-block：每g个元素（如g=128）共享一个scale，精度接近FP16，GPTQ/AWQ默认采用此方式 |

---

## 3. 量化误差来源
1. 舍入误差：浮点连续值→整数离散值的固有偏差。
2. 截断/饱和误差：浮点极值超出整数范围，离群值是精度暴跌主因。
3. 累积误差：多层误差叠加，LLM更明显。
4. 校准偏差：校准数据与真实场景分布不一致。

---

## 4. 精度恢复方法（核心方案）
| 方案 | 核心逻辑 | 优势 | 适用场景 |
|------|----------|------|----------|
| PTQ（训练后量化） | 预训练模型直接量化，少量校准数据 | 无训练、速度快 | 快速部署 |
| QAT（量化感知训练） | 训练中模拟量化噪声，微调适应低精度 | 精度几乎无损 | 高精度任务 |
| QLoRA | 4-bit量化 + LoRA微调，冻结基础模型 | 显存↓95%，精度接近FP16 | 大模型单卡微调 |
| QAD（量化感知蒸馏） | FP16教师模型指导INT4学生模型 | 比PTQ准、比QAT快 | 多阶段微调模型 |
| 离群值处理 | 单独量化/保留敏感离群值通道 | 针对性解决精度暴跌 | 大模型量化 |

---

## 5. 实战流程
1. 模型分析（权重分布、离群值、敏感层）→ 2. 选择方案 → 3. 校准（100-500条场景数据）→ 4. 量化 → 5. 评测 → 6. 精度修复。

---

## 6. 关键误区
1. 量化≠压缩：不仅减显存，更改变计算格式、触发硬件加速。
2. INT4不一定比INT8差：GPTQ/AWQ优化后，INT4精度接近INT8，显存再省50%。
3. 量化可加速：依赖内存带宽、硬件算力、缓存命中率三重提升。
