# main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 定义 POST 请求的数据模型
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float

# GET 接口
@app.get("/")
async def read_root():
    return {"message": "Hello World"}

# POST 接口
@app.post("/items/")
async def create_item(item: Item):
    return {
        "item_name": item.name,
        "item_price": item.price,
        "description": item.description
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)