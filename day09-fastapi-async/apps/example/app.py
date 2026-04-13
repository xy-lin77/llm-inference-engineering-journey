from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

# 初始化 FastAPI 应用
app = FastAPI(title="异步接口学习")

# ----------------------------------------------------
# 1. 异步接口 + 路径参数（自动校验类型）
# ----------------------------------------------------
@app.get("/user/{user_id}")
async def get_user(user_id: int):  # 必须是 int，传字符串会自动报错
    return {"user_id": user_id, "msg": "异步接口，路径参数校验成功"}

# ----------------------------------------------------
# 2. 异步接口 + 查询参数（必填/可选/默认值）
# ----------------------------------------------------
@app.get("/search")
async def search(
    keyword: str, 
    page: int = 1,                # 默认值
    size: Optional[int] = None    # 可选参数
):
    return {
        "keyword": keyword,
        "page": page,
        "size": size
    }

# ----------------------------------------------------
# 3. 请求体 + 强校验（Pydantic 模型）
# ----------------------------------------------------
class Item(BaseModel):
    # 商品名：最短2字符，最长50字符
    name: str = Field(min_length=2, max_length=50)
    # 价格：必须 > 0
    price: float = Field(gt=0)
    # 可选描述
    description: Optional[str] = None

# POST 异步接口
@app.post("/item")
async def create_item(item: Item):
    # item 已经自动校验完成
    return {
        "msg": "创建成功",
        "data": item.dict()
    }
