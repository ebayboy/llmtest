#!/bin/bash

# 安装LangChain LLM Embedding测试所需的依赖包

echo "正在安装依赖包..."

# 基础依赖
pip install langchain langchain-openai langchain-community

# 向量存储和embedding
pip install faiss-cpu

# 科学计算
pip install numpy scikit-learn

# 其他依赖
pip install openai
pip install python-dotenv

echo "依赖包安装完成！"