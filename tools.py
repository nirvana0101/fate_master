from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser

import os
import requests
load_dotenv(".env ")
YUANFENJU_API_KEY = os.getenv("YUANFENJU_API_KEY")


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
