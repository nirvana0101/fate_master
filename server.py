from fastapi import FastAPI
# 加载 .env 文件
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import os
load_dotenv(".env ")
# 读取环境变量
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")

# 打印配置信息
# print(f"OpenAI API Key: {openai_api_key}")
# print(f"OpenAI Base URL: {openai_base_url}")
# 示例工具定义（根据实际需求替换）

app = FastAPI()
@tool
def search(query: str) -> str:
    """用于执行网络搜索的工具"""
    return f"搜索结果: {query} (示例)"


class AIAgent:
    def __init__(self):
        # 初始化 OpenAI 配置
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

        # 定义工具集
        self.tools = [search]  # 可添加更多工具

        # 构建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是专业的人工智能助手"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 创建代理
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # 初始化执行器
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )


# 初始化主代理
master = AIAgent()


@app.get("/")
def health_check():
    return {"status": "OK"}

@app.post("/chat")
def chat_endpoint(query: str):
    try:
        result = master.agent_executor.invoke({"input": query})
        return {"response": result["output"]}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)