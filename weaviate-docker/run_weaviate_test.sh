#!/bin/bash

# Weaviate测试运行脚本
# 启动Weaviate服务并运行测试

echo "🚀 启动Weaviate服务..."
cd weaviate-docker

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请先启动Docker"
    exit 1
fi

# 停止已有的Weaviate容器
echo "🛑 停止已有的Weaviate容器..."
docker compose down 2>/dev/null || true

# 启动Weaviate
echo "📦 启动Weaviate容器..."
docker compose up -d

# 等待Weaviate启动
echo "⏳ 等待Weaviate启动..."
sleep 10


# 检查Weaviate是否就绪
echo "🔍 检查Weaviate状态..."
for i in {1..30}; do
    if curl -s http://localhost:8081/v1/.well-known/ready > /dev/null 2>&1; then
        echo "✅ Weaviate已就绪!"
        break
    fi
    echo "等待中... ($i/30)"
    sleep 5
done

# 如果Weaviate仍未就绪
if ! curl -s http://localhost:8081/v1/.well-known/ready > /dev/null 2>&1; then
    echo "⚠️ Weaviate可能还未完全就绪，但继续运行测试..."
fi

cd ..

# 安装Weaviate依赖
echo "📦 安装Weaviate依赖..."
pip install weaviate-client langchain-weaviate 

# 运行Weaviate测试
echo "🧪 运行Weaviate集成测试..."
python test_weaviate_simple.py

echo "✅ 测试完成！"