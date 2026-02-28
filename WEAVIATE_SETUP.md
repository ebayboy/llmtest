# Weaviate向量数据库集成指南

## 概述

本项目提供了使用Weaviate作为向量数据库的完整Embedding解决方案，包括：
- 基于Weaviate的向量存储和检索
- 支持OpenAI Embedding模型
- 自动回退到TF-IDF模式（当Weaviate不可用时）
- Docker容器化部署

## 文件结构

```
weaviate-docker/
├── docker-compose.yml          # Weaviate服务配置
├── run_weaviate_test.sh        # 一键启动和测试脚本
└── run.sh                      # 基础启动脚本

test_langchain_weaviate_embedding.py  # 主测试程序（Weaviate版本）
test_weaviate_simple.py               # 简化版演示程序
requirements-weaviate.txt             # Weaviate依赖包
.env.weaviate.example                 # 环境变量配置示例
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements-weaviate.txt
```

### 2. 启动Weaviate服务

```bash
# 使用一键脚本（推荐）
./weaviate-docker/run_weaviate_test.sh

# 或者手动启动
cd weaviate-docker
docker compose up -d
```

### 3. 运行测试

```bash
# 运行完整测试
python test_langchain_weaviate_embedding.py

# 或者运行简化版演示
python test_weaviate_simple.py
```

## 配置说明

### 环境变量

复制 `.env.weaviate.example` 为 `.env` 并根据需要修改：

```bash
# Weaviate配置
WEAVIATE_URL=http://localhost:8080
WEAVIATE_COLLECTION=KnowledgeBase

# LLM配置
LLM_API_URL=http://116.198.229.83:9998/v1
LLM_MODEL=baidu/ERNIE-4.5-21B-A3B-PT
LLM_API_KEY=openai
LLM_TEMPERATURE=0.3

# Embedding配置
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_API_URL=http://116.198.229.83:9998/v1
EMBEDDING_API_KEY=openai
USE_EMBEDDING=true
```

### Docker配置

`weaviate-docker/docker-compose.yml` 包含以下关键配置：

- **端口**: 8080 (REST API), 50051 (gRPC)
- **数据持久化**: 使用Docker卷存储数据
- **匿名访问**: 开发环境启用（生产环境建议关闭）
- **模块支持**: 启用OpenAI、Cohere等第三方API模块

## 使用示例

### 基本用法

```python
from test_langchain_weaviate_embedding import WeaviateLLMTester
import asyncio

async def main():
    # 创建测试器
    tester = WeaviateLLMTester()
    
    # 测试LLM响应（使用知识库）
    result = await tester.test_llm_response(
        "请介绍一下京东云安全运营中心的主要功能",
        use_knowledge=True
    )
    
    print(f"响应内容: {result['content']}")
    print(f"使用了知识库: {result['knowledge_used']}")
    
    # 关闭资源
    tester.close()

# 运行
asyncio.run(main())
```

### 知识库搜索

```python
# 搜索相似文档
results = tester.knowledge_base.search_similar_documents(
    "安全运营中心", top_k=3
)

for result in results:
    print(f"相似度: {result['similarity']:.3f}")
    print(f"内容: {result['content'][:100]}...")
    print(f"来源: {result['source']}")
```

## 工作原理

### 架构设计

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   用户查询       │───▶│   Embedding模型   │───▶│   Weaviate向量   │
│                 │    │                  │    │   数据库        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                        ┌──────────────────┐    ┌─────────────────┐
                        │   向量生成        │    │   相似度搜索     │
                        │                  │    │                 │
                        └──────────────────┘    └─────────────────┘
```

### 处理流程

1. **文档加载**: 从`knowledge/`目录读取文本文件
2. **文本分块**: 按段落分割，保留长度>50的文本块
3. **向量生成**: 使用OpenAI Embedding模型生成向量
4. **存储索引**: 将向量存储到Weaviate集合中
5. **相似度搜索**: 用户查询时生成查询向量，在Weaviate中搜索最相似的文档
6. **结果增强**: 将检索到的知识作为上下文提供给LLM

## 故障排除

### Weaviate连接失败

如果Weaviate服务不可用，系统会自动回退到TF-IDF模式：

```
⚠️ Weaviate初始化失败，回退到TF-IDF搜索
📚 知识库加载完成，共 42 个文档段落
✅ TF-IDF向量索引构建完成
```

### Docker容器启动问题

```bash
# 检查容器状态
docker compose ps

# 查看日志
docker compose logs

# 重新启动
docker compose down
docker compose up -d
```

### 端口冲突

如果8080端口被占用，修改`docker-compose.yml`：

```yaml
ports:
  - "8081:8080"  # 改为其他端口
```

然后更新环境变量：
```bash
WEAVIATE_URL=http://localhost:8081
```

## 性能优化

### 向量搜索优化

- **相似度阈值**: 默认0.1，可根据需要调整
- **返回结果数**: 通过`top_k`参数控制
- **向量维度**: 使用合适的Embedding模型

### 资源管理

- **连接池**: Weaviate客户端自动管理连接
- **内存使用**: 定期清理不需要的集合
- **数据持久化**: 使用Docker卷保证数据安全

## 扩展功能

### 支持多种Embedding模型

```python
# OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# HuggingFace Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### 混合搜索

结合向量搜索和关键词搜索：

```python
# 向量搜索 + BM25关键词搜索
results = vectorstore.similarity_search(
    query, 
    k=10,
    search_type="hybrid"  # 混合搜索
)
```

## 安全建议

1. **认证配置**: 生产环境启用Weaviate认证
2. **网络安全**: 限制Weaviate访问端口
3. **数据加密**: 敏感数据考虑加密存储
4. **访问控制**: 实施适当的权限管理

## 相关资源

- [Weaviate官方文档](https://weaviate.io/developers/weaviate)
- [LangChain Weaviate集成](https://python.langchain.com/docs/integrations/vectorstores/weaviate)
- [Docker部署指南](https://weaviate.io/developers/weaviate/installation/docker-compose)

## 技术支持

如遇到问题，请检查：
1. Docker服务是否正常运行
2. Weaviate容器是否成功启动
3. 网络连接是否正常
4. 环境变量配置是否正确