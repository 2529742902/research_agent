import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv("DeepSeek_API_KEY")


# 1. 初始化执行 Agent 使用的模型 (DeepSeek-V3)
# 专注于高频工具调用、文本抓取与报告撰写，追求极致性价比
research_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=api_key,
    openai_api_base="https://api.deepseek.com",
    temperature=0.2,  # 较低温度保证调研报告的严谨性与格式稳定
    max_tokens=4096
)

# 2. 初始化评测 Agent 使用的模型 (DeepSeek-R1)
# 专注于复杂的深度推理、打分、诊断与 Prompt 优化
critic_llm = ChatOpenAI(
    model="deepseek-v4-pro", # 对应 DeepSeek-R1 推理模型
    api_key=api_key,
    openai_api_base="https://api.deepseek.com",
    temperature=0.0,  # 裁判模型温度设为 0，确保评测标准的一致性
)

print("LLM 初始化成功：Research LLM (deepseek-chat) & Critic LLM (deepseek-reasoner)")