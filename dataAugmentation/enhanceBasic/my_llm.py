import os
from langchain_openai import ChatOpenAI

# 模型全局参数配置  根据自己的实际情况进行调整
# openai模型相关配置 根据自己的实际情况进行调整
OPENAI_API_BASE = "https://api.wlai.vip/v1"
OPENAI_CHAT_API_KEY = "sk-dsLvfvO3OLzr6jKpjgFBaEra3sZVM5UWVCpYlZ34QIjmIbVs"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
# 非gpt大模型相关配置(oneapi方案 通义千问为例) 根据自己的实际情况进行调整
ONEAPI_API_BASE = "http://139.224.72.218:3000/v1"
ONEAPI_CHAT_API_KEY = "sk-iEokjc8sQa9yvf8c655eCa8b0b834aB2Ab67213e639259Df"
ONEAPI_CHAT_MODEL = "qwen-max"
# 本地大模型相关配置(Ollama方案 qwen2.5:7b为例) 根据自己的实际情况进行调整
OLLAMA_API_BASE = "http://localhost:11434/v1"
OLLAMA_CHAT_API_KEY = "ollama"
OLLAMA_CHAT_MODEL = "qwen2.5:7b"

# 选择使用哪一个大模型
# openai:调用gpt大模型
# oneapi:调用非gpt大模型(国产大模型等)
# ollama:调用本地大模型
LLM_TYPE = "openai"

# 定函数 模型初始化
def myLLM():
    llmType = LLM_TYPE

    if llmType == "oneapi":
        # 实例化一个oneapi客户端对象
        llm = ChatOpenAI(
            base_url=ONEAPI_API_BASE,
            api_key=ONEAPI_CHAT_API_KEY,
            model=ONEAPI_CHAT_MODEL,  # 本次使用的模型
            temperature=0.8,  # 发散的程度
            # timeout=None,# 服务请求超时
            # max_retries=2,# 失败重试最大次数
        )

    elif llmType == "ollama":
        # 实例化一个ChatOpenAI客户端对象
        os.environ["OPENAI_API_KEY"] = "NA"
        llm = ChatOpenAI(
            base_url=OLLAMA_API_BASE,  # 请求的API服务地址
            api_key=OLLAMA_CHAT_API_KEY,  # API Key
            model=OLLAMA_CHAT_MODEL,  # 本次使用的模型
            temperature=0.8,  # 发散的程度
            # timeout=None,# 服务请求超时
            # max_retries=2,# 失败重试最大次数
        )

    else:
        # 实例化一个ChatOpenAI客户端对象
        llm = ChatOpenAI(
            base_url=OPENAI_API_BASE,  # 请求的API服务地址
            api_key=OPENAI_CHAT_API_KEY,  # API Key
            model=OPENAI_CHAT_MODEL,  # 本次使用的模型
            # temperature=0.7,# 发散的程度，一般为0
            # timeout=None,# 服务请求超时
            # max_retries=2,# 失败重试最大次数
        )

    return llm