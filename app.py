import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"D:\python训练\python_practice\GitHub_practice1.env")


API_KEY=os.getenv("DEEPSEEK_API_KEY")# 本地运行时用默认值，云端配置环境变量

# 读取手册内容（固定知识库）
with open("product_manual.txt", "r", encoding="utf-8") as f:
    knowledge_base = f.read()

# 页面标题
st.title("📖 产品知识库问答助手")
st.write("请提问关于 '智净X3' 电动牙刷的问题")

# 用户输入框
user_question = st.text_input("输入你的问题：")

# 提交按钮
if st.button("提交问题"):
    if user_question:
        # 构造提示词（和rag_qa.py一样）
        prompt = f"""
请根据下面提供的【产品手册】内容，回答用户的问题。
要求：
- 只能基于手册内容回答，严禁添加手册中没有的信息。
- 如果手册里没有提到，请直接回答“手册中未提及”。

【产品手册】
{knowledge_base}

【用户问题】
{user_question}
"""

        # 调用API
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            st.success("回答：")
            st.write(answer)
        else:
            st.error(f"API调用失败，错误码：{response.status_code}")
            st.text(response.text)
    else:
        st.warning("请先输入一个问题！")

