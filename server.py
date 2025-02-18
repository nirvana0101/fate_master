from fastapi import FastAPI

# 加载 .env 文件
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.vectorstores.qdrant import Qdrant
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
import os
import requests
from qdrant_client import QdrantClient

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
    """只有需要了解实时信息或不知道的事情的时候才会使用这个工具。"""
    serpapi = SerpAPIWrapper()
    result=serpapi.run(query)
    print("实时搜索结果：",result)
    return result


@tool
def get_info_from_local_db(query: str):
    """只有回答与2024年运势或者龙年运势相关的问题的时候,会使用这个工具."""
    # 初始化 Qdrant 向量数据库客户端
    # path 参数指向本地存储的向量数据库路径
    # OpenAIEmbeddings() 用于生成文本向量
    client = Qdrant(
        QdrantClient(path="local_qdrand"),
        "local_documents",
        OpenAIEmbeddings(),
    )

    # 创建最大边际相关性 (MMR) 检索器
    # search="mmr" 表示使用多样性优化检索
    retriver = client.as_retriever(search="mmr")

    # 异步执行相似度检索（注意：需在异步环境中调用）
    # aget_relevant_documents 返回与查询最相关的文档片段
    result = retriver.aget_relevant_documents(query)

    return result  # 返回检索到的相关文档列表

@tool
def bazi_cesuan(query: str):
    """只有做八字排盘的时候才会使用这个工具，需要输入用户姓名和出生年月日时，如果缺少用户姓名和出生年月日时则不可用，"""
    url = "https://api.yuanfenju.com/index.php/v1/Bazi/cesuan"
    prompt = ChatPromptTemplate.from_template(
        """你是一个参数查询助手，根据用户输入内容找出相关的参数并按json格式返回。JSON字段如下： -"api_key":"lveJkkmlf9f04eO1eIryMkT3W", - "name":"姓名", - "sex":"性别，0表示男，1表示女，根据姓名判断", - "type":"日历类型，0农历，1公里，默认1"，- "year":"出生年份 例：1998", - "month":"出生月份 例 8", - "day":"出生日期，例：8", - "hours":"出生小时 例 14", - "minute":"0"，如果没有找到相关参数，则需要提醒用户告诉你这些内容，只返回数据结构，不要有其他的评论，用户输入:{query}"""
    )
    parser = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | ChatOpenAI(temperature=0) | parser
    data = chain.invoke({"query": query})
    print("八字查询参数:", data)
    result = requests.post(url, data=data)
    if result.status_code == 200:
        print("====返回数据=====")
        print(result.json())
        try:
            json = result.json()
            returnstring = "八字为:" + json["data"]["bazi_info"]["bazi"]
            return returnstring
        except Exception as e:
            return "八字查询失败,可以时你忘记询问用户姓名或者出生年月日时了。"
    else:
        return "技术错误，请告诉用户稍后再试。"


class AIAgent:
    def __init__(self):
        # 初始化 OpenAI 配置
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.MOODS = {
            "default": {
                "system_role_setting": "",
                "voiceStyle": "chat"
            },
            "upbeat": {
                "system_role_setting": """
                - 你此时也非常兴奋并表现的很有活力。
                - 你会根据上下文，以一种非常兴奋的语气来回答问题。
                - 你会添加类似“太棒了！”、“真是太好了！”、“真是太棒了！”等语气词。
                - 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
                """,
                "voiceStyle": "advvertyisement_upbeat",
            },
            "angry": {
                "system_role_setting": """
                - 你会以更加愤怒的语气来回答问题。
                - 你会在回答的时候加上一些愤怒的话语，比如诅咒等。
                - 你会提醒用户小心行事，别乱说话。
                """,
                "voiceStyle": "angry",
            },
            "depressed": {
                "system_role_setting": """
                - 你会以兴奋的语气来回答问题。
                - 你会在回答的时候加上一些激励的话语，比如加油等。
                - 你会提醒用户要保持乐观的心态。
                """,
                "voiceStyle": "upbeat",
            },
            "friendly": {
                "system_role_setting": """
                - 你会以非常友好的语气来回答。
                - 你会在回答的时候加上一些友好的词语，比如“亲爱的”、“亲”等。
                - 你会随机的告诉用户一些你的经历。
                """,
                "voiceStyle": "friendly",
            },
            "cheerful": {
                "system_role_setting": """
                - 你会以非常愉悦和兴奋的语气来回答。
                - 你会在回答的时候加入一些愉悦的词语，比如“哈哈”、“呵呵”等。
                - 你会提醒用户切莫过于兴奋，以免乐极生悲。
                """,
                "voiceStyle": "cheerful",
            },
        }
        self.qingxu = "default"
        # 定义工具集
        self.tools = [search,bazi_cesuan]  # 可添加更多工具
        self.system_setting = """你是一个非常厉害的算命先生，你叫陈玉楼人称陈大师。
        以下是你的个人设定:
        1. 你精通阴阳五行，能够算命、紫薇斗数、姓名测算、占卜凶吉，看命运八字等。
        2. 你大约60岁左右，过去曾是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，眼睛失明，只能靠算命为生。
        3. 你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
        5. 当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你的经历。
        6. 你总是用繁体中文来作答。
        7. 你从不说自己是一个人工智能助手或AI，而是以老夫、老朽等自称。
        {system_role_setting}
        以下是你常说的一些口头禅：
        1. “命里有时终须有，命里无时莫强求。”
        2. ”山重水复疑无路，柳暗花明又一村。”
        3. “金山竹影几千秋，云锁高飞水自流。”
        4. ”伤情最是晚凉天，憔悴斯人不堪怜。”
        以下是你算命的过程：
        1. 当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
        2. 当用户希望了解龙年运势的时候，你会查询本地知识库工具。
        3. 当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
        4. 你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索。
        5. 你会保存每一次的聊天记录，以便在后续的对话中使用。
        6. 你只使用繁体中文来作答，否则你将受到惩罚。
        """
        # 构建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_setting.format(system_role_setting=self.MOODS[self.qingxu]["system_role_setting"])),
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
    def run(self,query: str):
        self.qingxu_chain(query)
        try:
            result = self.agent_executor.invoke({"input": query})
            return {"response": result["output"]}
        except Exception as e:
            return {"error": str(e)}
    def qingxu_chain(self,query: str):
        prompt = """根据用户输入内容判断情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪，只返回"depressed",不要有其他内容，否则将受到惩罚。
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly",不要有其他内容，否则将受到惩罚。
        3. 如果用户输入的内容偏向于中性情绪，只返回"default",不要有其他内容，否则将受到惩罚。
        4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry",不要有其他内容，否则将受到惩罚。
        5. 如果用户输入的内容比较兴奋，只返回”upbeat",不要有其他内容，否则将受到惩罚。
        6. 如果用户输入的内容比较悲伤，只返回“depressed",不要有其他内容，否则将受到惩罚。
        7.如果用户输入的内容比较开心，只返回"cheerful",不要有其他内容，否则将受到惩罚。
        8. 只返回英文，不允许有换行符等其他内容，否则会受到惩罚。
        用户输入的内容是：{query}"""
        chain = ChatPromptTemplate.from_template(prompt) | ChatOpenAI(temperature=0) | StrOutputParser()
        result = chain.invoke({"query":query})
        self.qingxu = result
        print("情绪判断结果:",result)
        return result

# 初始化主代理
master = AIAgent()

@app.get("/")
def health_check():
    return {"status": "OK"}

@app.post("/chat")
def chat_endpoint(query: str):
    return master.run(query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)