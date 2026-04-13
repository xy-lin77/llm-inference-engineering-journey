from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

# 初始化 FastAPI 应用
app = FastAPI(title="异步接口学习")

# ----------------------------------------------------
# 1. 异步接口 + 路径参数（自动校验类型）
# ----------------------------------------------------
# 可访问 URL 示例：
# GET http://127.0.0.1:8000/user/1001
# 注意：user_id 必须是数字，否则自动报错
@app.get("/user/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id, "msg": "异步接口，路径参数校验成功"}

# ----------------------------------------------------
# 2. 异步接口 + 查询参数（必填/可选/默认值）
# ----------------------------------------------------
# 可访问 URL 示例：
# GET http://127.0.0.1:8000/search?keyword=电脑
# GET http://127.0.0.1:8000/search?keyword=手机&page=2
# GET http://127.0.0.1:8000/search?keyword=耳机&page=3&size=10
# keyword 必填，page 有默认值1，size 可选
@app.get("/search")
async def search(
    keyword: str, 
    page: int = 1,
    size: Optional[int] = None
):
    return {
        "keyword": keyword,
        "page": page,
        "size": size
    }

# ----------------------------------------------------
# 3. 请求体 + 强校验（Pydantic 模型）
# ----------------------------------------------------
# 访问方式：POST 请求，发送 JSON 格式数据
# URL：http://127.0.0.1:8000/item
# 请求体 JSON 示例：
# {
#   "name": "测试商品",
#   "price": 99.9,
#   "description": "这是一个示例"
# }
class Item(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    price: float = Field(gt=0)
    description: Optional[str] = None

@app.post("/item")
async def create_item(item: Item):
    return {
        "msg": "创建成功",
        "data": item.dict()
    }
