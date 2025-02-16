from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from dotenv import load_dotenv
import os

import test

load_dotenv(".env")
app = FastAPI()
# 用于存储连接的 WebSocket 客户端
connected_clients = set()
@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.post("/chat")
def chat(query: str):
    master = Master()
    return master.run(query)
@app.post("/add_urls")
def add_urls():
    return {"response": "URLs added!"}
@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added!"}
@app.post("/add_texts")
def add_texts():
    return {"response": "Texts added!"}
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # 回复接收到的消息给客户端
            await websocket.send_text(f"Received: {data}")
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        await websocket.close()
class Master:
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=0,
            streaming=True,
            api_key="sk-QH6zwcxIVlTb5N6gZvLWYL8JcsX2653J54qFYPeDmPMSiPXY",  # 替换为实际的API密钥
            base_url="https://api.openai-proxy.org/v1"  # 替换为实际的API地址
        )
        self.MEMORY_KEY = "chat_history"
        self.SYSTEM_PROMPT = ""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个助理"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        self.memory = []
        tools = [test]  # 确保 'test' 是一个有效的工具对象或函数
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
        )

    def run(self, query):
        result = self.agent_executor.invoke(query)
        return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
