# FastAPI 异步接口开发
## 1. 依赖安装
```bash
pip install fastapi uvicorn pydantic
```

---

## 2. 异步接口（async def）
- 用 `async def` 定义，处理IO操作需加 `await`。高并发、高性能，禁止在里面跑耗时的纯计算。
- 纯计算/同步操作可用普通 `def`，FastAPI 会自动把它放到线程池执行，不阻塞主事件循环。

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

---

## 4. 启动与文档
- 启动：`uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- 交互式文档：http://127.0.0.1:8000/docs
