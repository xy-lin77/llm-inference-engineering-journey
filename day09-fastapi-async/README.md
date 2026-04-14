# FastAPI 异步接口开发

## 1. Quick Start

### 1.1 依赖安装
```bash
pip install fastapi uvicorn pydantic
```

### 1.2 目录结构
```
/day09-fastapi-async/
├── app_example/
│   └── main.py # 学习FastAPI
└── app_inference_basic/
│   └── main.py # （极简版）vLLM封装
└── app_inference_advanced/
    └── main.py # （生产版）校验+异常+结构化返回
```

### 1.3 启动前提
1. 终端进入 **/day09-fastapi-async/** 根目录
   ```bash
   cd /day09-fastapi-async
   ```
2. 安装依赖
   ```bash
   pip install fastapi uvicorn vllm pydantic
   ```

### 1.4 分别启动三个独立服务
1. 启动示例服务（app_example）
端口：8000
```bash
uvicorn app_example.main:app --host 0.0.0.0 --port 8000
```

2. 启动基础推理服务（app_inference_basic）
**新开一个终端**，端口：8001
```bash
uvicorn app_inference_basic.main:app --host 0.0.0.0 --port 8001
```

3. 启动基础推理服务（app_inference_advanced）
**新开一个终端**，端口：8002
```bash
uvicorn app_inference_advanced.main:app --host 0.0.0.0 --port 8002
```

4. 访问服务
- 示例服务文档：http://localhost:8000/docs
- 推理服务文档：http://localhost:8001/docs ，http://localhost:8002/docs

---

## 2. 异步接口（async def）
 
| 特性 | `async def` 异步函数 | 普通 `def` 同步函数 |
|------|----------------------|--------------------|
| 核心场景 | IO 密集型任务（网络请求、文件读写、模型推理） | CPU 密集型任务（纯计算、数据处理） |
| 执行方式 | 非阻塞，需 `await` 挂起/恢复 | 阻塞执行 |
| 并发能力 | 高，单线程处理大量请求 | 低，依赖线程池调度 |
| FastAPI 处理 | 直接运行在主事件循环 | 自动放入线程池，不阻塞主循环 |
| 适用接口 | /generate、/stream 推理接口 | 数据计算、格式转换接口 |

---

## 3. 请求参数校验（自动完成）
### 3.1 路径参数
```python
@app.get("/user/{user_id}")
async def get_user(user_id: int):  # 自动校验int类型
    return {"user_id": user_id}
```

### 3.2 查询参数
```python
@app.get("/search")
async def search(keyword: str, page: int = 1, size: Optional[int] = None):
    return {"keyword": keyword, "page": page}
```

### 3.3 请求体（Pydantic模型）
```python
class Item(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    price: float = Field(gt=0)
    description: Optional[str] = None

@app.post("/item")
async def create_item(item: Item):  # 自动校验所有字段
    return {"data": item.dict()}
```
