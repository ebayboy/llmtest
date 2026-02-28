# LangChain LLM Embedding 测试工具

这是一个基于LangChain的LLM API测试工具，支持知识库增强的问答功能，并实现了从TF-IDF到Embedding模型的升级。

## 功能特性

### 核心功能
- **LLM API测试**: 测试OpenAI兼容的LLM API连接和响应
- **知识库集成**: 基于本地知识文档的智能问答
- **流式响应**: 支持流式输出的测试
- **双检索模式**: 支持TF-IDF和Embedding两种知识检索方式

### 新增Embedding功能
- **OpenAI Embeddings**: 使用text-embedding-ada-002模型
- **FAISS向量存储**: 高效的向量相似度搜索
- **智能回退**: Embedding失败时自动回退到TF-IDF
- **检索方法显示**: 清晰展示使用的检索技术

## 快速开始

### 1. 安装依赖
```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，设置你的API配置
```

### 3. 运行测试
```bash
python test_langchain_llm_embedding.py
```

## 配置说明

### 基础配置
- `LLM_API_URL`: LLM API地址
- `LLM_MODEL`: 使用的模型名称
- `LLM_API_KEY`: API密钥
- `LLM_TEMPERATURE`: 温度参数(0-1)

### Embedding配置
- `USE_EMBEDDING`: 是否启用embedding功能 (true/false)
- `EMBEDDING_MODEL`: embedding模型名称
- `EMBEDDING_API_URL`: embedding API地址（可选，默认使用LLM API地址）
- `EMBEDDING_API_KEY`: embedding API密钥（可选，默认使用LLM API密钥）

#### 可用的Embedding模型配置
```bash
# Qwen3-Embedding-8B模型（推荐）
EMBEDDING_API_URL=http://116.198.229.83:8009/v1
EMBEDDING_API_KEY=1F981D0DAF0135FE228C505557A4412F
EMBEDDING_MODEL=Qwen3-Embedding-8B
```

## 知识库结构

知识文档存储在`knowledge/`目录下，支持`.txt`格式：
```
knowledge/
├── soc1.txt    # 产品概述和技术架构
├── soc2.txt    # 功能模块详情
├── soc3.txt    # 配置和管理
└── soc4.txt    # 其他功能说明
```

## 使用示例

### 基础LLM测试
```python
# 测试API连接
result = await tester.test_connection()

# 测试LLM响应
result = await tester.test_llm_response("你的问题")
```

### 知识增强问答
```python
# 使用知识库进行问答
result = await tester.test_llm_response(
    "请介绍一下京东云安全运营中心的主要功能",
    use_knowledge=True
)
```

### 知识库搜索
```python
# 搜索相关知识
results = tester.knowledge_base.search_similar_documents("安全运营", top_k=3)
```

## 技术实现

### 双检索架构
1. **Embedding模式** (优先)
   - OpenAI Embeddings生成向量
   - FAISS向量存储和相似度搜索
   - 语义理解能力更强

2. **TF-IDF模式** (回退)
   - 传统的词频-逆文档频率
   - 余弦相似度计算
   - 稳定可靠，无需额外API

### 智能切换机制
- Embedding API可用时优先使用语义搜索
- Embedding失败时自动回退到TF-IDF
- 用户无感知切换，保证功能连续性

## 测试结果展示

运行测试后会显示：
- ✅ API连接状态
- 🤖 基础LLM响应测试
- 📚 知识增强问答测试（显示检索方法）
- 🌊 流式响应测试
- 🔍 知识库搜索功能测试

## 注意事项

1. **Embedding API支持**: 确保你的API支持embedding功能
2. **知识文档格式**: 建议每个段落不少于50个字符
3. **搜索阈值**: 相似度低于0.1的结果会被过滤
4. **性能优化**: 大量文档时建议使用embedding提升检索质量

## 故障排除

### Embedding初始化失败
- 检查API是否支持embedding端点
- 确认embedding模型名称正确
- 设置`USE_EMBEDDING=false`使用TF-IDF模式

### 知识检索效果差
- 调整`top_k`参数增加返回结果数
- 降低相似度阈值
- 优化知识文档内容质量

## 扩展开发

欢迎提交Issue和Pull Request来改进功能，包括但不限于：
- 支持更多embedding模型
- 优化检索算法
- 增加可视化界面
- 扩展知识库格式支持


# 向量数据库 && Embedding向量 || TF-IDF的FAISS向量
现在我看到了当前的Embedding实现。代码中已经支持了基于OpenAI的Embedding模型和FAISS向量存储，但你需要使用Weaviate作为向量数据库。让我为你创建一个使用Weaviate的集成方案。
推荐方案： 用专用向量库（Milvus/Weaviate）存 ， 用BERT Embedding 做语义召回