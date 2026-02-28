#!/usr/bin/env python3
"""
LangChain LLM API测试工具 - Weaviate版本
使用LangChain接口实现LLM调用，集成Weaviate向量数据库
"""

import os
import sys
import time
from typing import Dict, Any, Optional, List
import logging
import asyncio
import uuid

# 加载环境变量
from dotenv import load_dotenv

load_dotenv()

# 导入LangChain相关模块
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore

# 导入知识库相关模块
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob

# 配置日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(log_dir, "weaviate_embedding.log"), mode="w", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)


class WeaviateKnowledgeBase:
    """基于Weaviate的知识库管理类"""

    def __init__(
        self,
        knowledge_dir: str = "knowledge",
        weaviate_url: str = None,
        embedding_model=None,
        collection_name: str = None,
    ):
        """初始化Weaviate知识库"""
        self.knowledge_dir = knowledge_dir

        # 从环境变量获取Weaviate配置，如果没有提供参数
        self.weaviate_url = (
            weaviate_url
            if weaviate_url is not None
            else os.getenv("WEAVIATE_URL", "http://localhost:8081")
        )
        self.embedding_model = embedding_model
        self.collection_name = (
            collection_name
            if collection_name is not None
            else os.getenv("WEAVIATE_COLLECTION", "KnowledgeBase")
        )
        self.documents = []
        self.vectorizer = None
        self.doc_vectors = None
        self.weaviate_client = None
        self.vectorstore = None
        self.collection = None

        # 初始化Weaviate客户端
        self.init_weaviate_client()

        # 加载文档
        self.load_documents()

    def init_weaviate_client(self):
        """初始化Weaviate客户端"""
        try:
            logger.info(f"正在连接Weaviate服务器: {self.weaviate_url}")

            # 创建Weaviate客户端
            # 从 URL 中提取端口，默认为 8081
            weaviate_port = 8081
            if self.weaviate_url and ":" in self.weaviate_url:
                try:
                    # 从 URL 如 http://localhost:8081 中提取端口
                    weaviate_port = int(self.weaviate_url.split(":")[-1])
                except (ValueError, IndexError):
                    weaviate_port = 8081

            self.weaviate_client = weaviate.connect_to_local(
                host="localhost",
                port=weaviate_port,
                grpc_port=50051,
            )

            # 测试连接
            if self.weaviate_client.is_ready():
                logger.info("✅ Weaviate连接成功!")

                # 创建或获取collection
                self.setup_collection()
            else:
                logger.error("❌ Weaviate连接失败")
                self.weaviate_client = None

        except Exception as e:
            logger.error(f"❌ Weaviate初始化失败: {e}")
            self.weaviate_client = None

    def setup_collection(self) -> bool:
        """设置Weaviate集合"""
        try:
            # 检查客户端是否已初始化
            if not self.weaviate_client:
                logger.error("Weaviate客户端未初始化")
                return False

            # 检查集合是否存在
            if self.weaviate_client.collections.exists(self.collection_name):
                logger.info(f"集合 '{self.collection_name}' 已存在")
                return True

            # 创建新集合
            logger.info(f"创建集合 '{self.collection_name}'...")
            self.weaviate_client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none(),  # 使用外部向量
                properties=[
                    Property(
                        name="content", data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    Property(
                        name="source", data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    Property(
                        name="timestamp",
                        data_type=weaviate.classes.config.DataType.DATE,
                    ),
                ],
            )
            logger.info(f"✅ 集合 '{self.collection_name}' 创建成功")
            return True

        except Exception as e:
            logger.error(f"❌ 集合创建失败: {e}")
            return False

    def load_documents(self):
        """加载知识文档"""
        logger.info(f"开始加载知识库文档，目录: {self.knowledge_dir}")

        try:
            # 获取所有txt文件
            txt_files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
            logger.info(f"找到 {len(txt_files)} 个知识文档")

            documents_for_embedding = []
            for file_path in txt_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # 按段落分割文档，避免单个文档过长
                        paragraphs = content.split("\n\n")
                        for para in paragraphs:
                            para = para.strip()
                            if len(para) > 50:  # 只保留长度大于50的段落
                                doc_data = {
                                    "content": para,
                                    "source": os.path.basename(file_path),
                                }
                                self.documents.append(doc_data)

                                # 创建Document对象用于embedding
                                if self.embedding_model:
                                    documents_for_embedding.append(
                                        Document(
                                            page_content=para,
                                            metadata={
                                                "source": os.path.basename(file_path),
                                                "content": para,
                                            },
                                        )
                                    )
                    logger.info(f"✅ 加载文档成功: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"❌ 加载文档失败 {file_path}: {e}")

            logger.info(f"📚 知识库加载完成，共 {len(self.documents)} 个文档段落")

            # 构建向量索引
            if self.documents:
                if self.weaviate_client and self.collection and self.embedding_model:
                    self.build_weaviate_vectorstore(documents_for_embedding)
                else:
                    # 回退到TF-IDF
                    logger.info("Weaviate不可用，回退到TF-IDF向量索引...")
                    self.build_vector_index()

        except Exception as e:
            logger.error(f"❌ 知识库加载异常: {e}")

    def build_vector_index(self):
        """构建TF-IDF向量索引（回退方案）"""
        try:
            logger.info("开始构建TF-IDF向量索引...")

            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words=None,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
            )

            doc_contents = [doc["content"] for doc in self.documents]
            self.doc_vectors = self.vectorizer.fit_transform(doc_contents)

            logger.info(f"✅ TF-IDF向量索引构建完成，维度: {self.doc_vectors.shape}")

        except Exception as e:
            logger.error(f"❌ TF-IDF向量索引构建失败: {e}")
            self.vectorizer = None
            self.doc_vectors = None

    def build_weaviate_vectorstore(self, documents: List[Document]):
        """构建Weaviate向量存储"""
        try:
            logger.info("开始构建Weaviate向量存储...")

            # 检查Weaviate客户端和collection是否可用
            if not self.weaviate_client:
                logger.error("Weaviate客户端未初始化")
                raise Exception("Weaviate客户端未初始化")

            if not self.collection:
                logger.error("Weaviate collection未初始化")
                raise Exception("Weaviate collection未初始化")

            # 创建Weaviate向量存储
            self.vectorstore = WeaviateVectorStore(
                client=self.weaviate_client,
                index_name=self.collection_name,
                text_key="content",
                embedding=self.embedding_model,
            )

            # 检查是否已有数据
            existing_count = self.collection.aggregate.over_all(
                total_count=True
            ).total_count
            if existing_count > 0:
                logger.info(f"Collection中已有 {existing_count} 条数据，跳过导入")
            else:
                # 添加文档到Weaviate
                logger.info(f"正在添加 {len(documents)} 个文档到Weaviate...")
                uuids = self.vectorstore.add_documents(documents)
                logger.info(f"✅ 成功添加 {len(uuids)} 个文档到Weaviate")

            logger.info("✅ Weaviate向量存储构建完成")

        except Exception as e:
            logger.error(f"❌ Weaviate向量存储构建失败: {e}")
            self.vectorstore = None
            # 回退到TF-IDF
            logger.info("回退到TF-IDF向量索引...")
            self.build_vector_index()

    def search_similar_documents(self, query: str, top_k: int = 3) -> list:
        """搜索相似文档"""
        # 优先使用Weaviate向量存储
        if self.vectorstore:
            try:
                logger.info("使用Weaviate进行相似度搜索...")

                # 执行相似度搜索
                relevant_docs = self.vectorstore.similarity_search_with_score(
                    query, k=top_k
                )

                results = []
                for doc, score in relevant_docs:
                    # Weaviate返回的是距离，转换为相似度
                    similarity = max(0, 1 - score)
                    if similarity > 0.1:  # 相似度阈值
                        results.append(
                            {
                                "content": doc.page_content,
                                "source": doc.metadata.get("source", "unknown"),
                                "similarity": float(similarity),
                            }
                        )

                logger.info(f"🔍 Weaviate搜索到 {len(results)} 个相似文档")
                return results

            except Exception as e:
                logger.error(f"❌ Weaviate搜索失败: {e}，回退到TF-IDF搜索")

        # 回退到TF-IDF搜索
        return self.search_with_tfidf(query, top_k)

    def search_with_tfidf(self, query: str, top_k: int = 3) -> list:
        """使用TF-IDF搜索相似文档"""
        if not self.vectorizer or self.doc_vectors is None:
            logger.warning("知识库向量化未完成，返回空结果")
            return []

        try:
            # 将查询向量化
            query_vector = self.vectorizer.transform([query])

            # 计算相似度
            similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

            # 获取最相似的文档索引
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # 构建结果
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # 相似度阈值
                    results.append(
                        {
                            "content": self.documents[idx]["content"],
                            "source": self.documents[idx]["source"],
                            "similarity": float(similarities[idx]),
                        }
                    )

            logger.info(
                f"🔍 TF-IDF搜索到 {len(results)} 个相似文档，最高相似度: {similarities[top_indices[0]]:.3f}"
            )
            return results

        except Exception as e:
            logger.error(f"❌ TF-IDF文档搜索失败: {e}")
            return []

    def close(self):
        """关闭Weaviate连接"""
        if self.weaviate_client:
            try:
                self.weaviate_client.close()
                logger.info("Weaviate连接已关闭")
            except Exception as e:
                logger.error(f"关闭Weaviate连接失败: {e}")


class WeaviateLLMTester:
    """使用Weaviate的LangChain LLM API测试器"""

    def __init__(self):
        """初始化测试器"""
        # 从环境变量获取配置
        self.api_url = os.getenv("LLM_API_URL", "http://116.198.229.83:9998/v1")
        self.model_name = os.getenv("LLM_MODEL", "baidu/ERNIE-4.5-21B-A3B-PT")
        self.api_key = os.getenv("LLM_API_KEY", "openai")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

        # Weaviate配置
        self.weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.collection_name = os.getenv("WEAVIATE_COLLECTION", "KnowledgeBase")

        # Embedding配置
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-ada-002"
        )
        self.use_embedding = os.getenv("USE_EMBEDDING", "true").lower() == "true"

        self.embedding_api_url = os.getenv("EMBEDDING_API_URL", self.api_url)
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", self.api_key)

        # 设置环境变量
        os.environ["OPENAI_API_KEY"] = self.api_key

        # 初始化LangChain LLM客户端
        self.llm = ChatOpenAI(
            model=self.model_name, base_url=self.api_url, temperature=self.temperature
        )

        # 初始化Embedding模型
        self.embeddings = None
        if self.use_embedding:
            try:
                self.embeddings = OpenAIEmbeddings(
                    model=self.embedding_model_name,
                    api_key=self.embedding_api_key,
                    base_url=self.embedding_api_url,
                )
                logger.info(f"✅ Embedding模型初始化成功: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Embedding模型初始化失败: {e}，将使用TF-IDF")
                self.use_embedding = False
                self.embeddings = None

        # 初始化Weaviate知识库
        self.knowledge_base = WeaviateKnowledgeBase(
            weaviate_url=self.weaviate_url,
            embedding_model=self.embeddings,
            collection_name=self.collection_name,
        )

        logger.info(
            f"初始化Weaviate LLM测试器: URL={self.api_url}, Model={self.model_name}, "
            f"Weaviate: {self.weaviate_url}, Collection: {self.collection_name}, "
            f"Embedding: {self.use_embedding} ({self.embedding_model_name if self.use_embedding else 'N/A'})"
        )

    async def test_connection(self, timeout: int = 10) -> Dict[str, Any]:
        """测试API连接"""
        logger.info("开始测试API连接...")

        try:
            start_time = time.time()
            response = await self.llm.ainvoke([HumanMessage(content="ping")])
            response_time = time.time() - start_time

            # 提取回复内容
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            logger.info(f"✅ API连接成功! 响应时间: {response_time:.2f}s")
            return {
                "success": True,
                "response_time": response_time,
                "message": "API连接正常",
                "response_content": content[:100] if content else "",
            }

        except Exception as e:
            logger.error(f"❌ API连接异常: {e}")
            return {
                "success": False,
                "message": f"API连接异常: {str(e)}",
                "error_type": "connection_error",
            }

    async def test_llm_response(
        self,
        prompt: str = "你好，请简单介绍一下自己。",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_knowledge: bool = True,
    ) -> Dict[str, Any]:
        """测试LLM响应功能，带重试机制"""
        logger.info(f"开始测试LLM响应，提示词: {prompt[:50]}...")

        # 知识检索
        knowledge_context = ""
        knowledge_sources = []
        if use_knowledge and self.knowledge_base:
            logger.info("🔍 正在检索相关知识...")
            similar_docs = self.knowledge_base.search_similar_documents(prompt, top_k=2)

            if similar_docs:
                knowledge_context = "\n\n".join(
                    [doc["content"] for doc in similar_docs]
                )
                knowledge_sources = [doc["source"] for doc in similar_docs]
                logger.info(
                    f"📚 检索到 {len(similar_docs)} 条相关知识，总长度: {len(knowledge_context)} 字符"
                )
            else:
                logger.info("📚 未检索到相关知识")

        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"第{attempt + 1}次尝试...")

                # 构建消息列表
                messages = []

                # 如果有相关知识，添加系统提示词
                if knowledge_context:
                    system_prompt = f"""你是一个专业的安全运营助手。以下是相关的安全知识，请结合这些知识来回答用户的问题：

{ knowledge_context }

请基于以上知识，准确、专业地回答用户的问题。如果问题与知识内容无关，请正常回答。"""
                    messages.append(SystemMessage(content=system_prompt))

                # 添加用户消息
                messages.append(HumanMessage(content=prompt))

                start_time = time.time()
                response = await self.llm.ainvoke(messages)
                response_time = time.time() - start_time

                # 提取回复内容
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )

                logger.info(f"✅ LLM响应成功! 响应时间: {response_time:.2f}s")
                logger.info(f"回复内容: {content[:100]}...")

                return {
                    "success": True,
                    "response_time": response_time,
                    "content": content,
                    "attempt": attempt + 1,
                    "knowledge_used": bool(knowledge_context),
                    "knowledge_sources": knowledge_sources,
                }

            except Exception as e:
                last_error = str(e)
                logger.warning(f"第{attempt + 1}次尝试异常: {last_error}")

            # 如果不是最后一次尝试，等待后重试
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # 指数退避

        logger.error(f"❌ 所有{max_retries}次尝试均失败: {last_error}")
        return {
            "success": False,
            "message": f"所有{max_retries}次尝试均失败: {last_error}",
            "error_type": "max_retries_exceeded",
            "knowledge_used": bool(knowledge_context),
        }

    def close(self):
        """关闭资源"""
        if self.knowledge_base:
            self.knowledge_base.close()


async def main():
    """主函数"""
    print("🚀 LangChain LLM API测试工具 - Weaviate版本")
    print("=" * 60)

    # 创建测试器
    tester = WeaviateLLMTester()

    # 测试连接
    print("\n📡 测试API连接...")
    connection_result = await tester.test_connection()

    if not connection_result["success"]:
        print(f"❌ 连接测试失败: {connection_result['message']}")
        if "response_text" in connection_result:
            print(f"响应内容: {connection_result['response_text']}")
        return

    print(f"✅ 连接测试成功! 响应时间: {connection_result['response_time']:.2f}s")

    # 测试LLM响应（不使用知识库）
    print("\n🤖 测试LLM响应（不使用知识库）...")
    test_prompt = "你好！请简单介绍一下自己，并告诉我你能如何帮助用户进行数据安全分析。"
    llm_result = await tester.test_llm_response(
        test_prompt, max_retries=3, use_knowledge=False
    )

    if llm_result["success"]:
        print(f"✅ LLM响应测试成功!")
        print(f"响应时间: {llm_result['response_time']:.2f}s")
        print(f"回复内容: {llm_result['content'][:200]}...")
    else:
        print(f"❌ LLM响应测试失败: {llm_result['message']}")

    # 测试LLM响应（使用知识库）
    print("\n🤖 测试LLM响应（使用知识库）...")
    knowledge_prompt = "请介绍一下京东云安全运营中心的主要功能和特点。"
    knowledge_result = await tester.test_llm_response(
        knowledge_prompt, max_retries=3, use_knowledge=True
    )

    if knowledge_result["success"]:
        print(f"✅ 知识增强LLM响应测试成功!")
        print(f"响应时间: {knowledge_result['response_time']:.2f}s")
        print(f"使用了知识库: {'是' if knowledge_result['knowledge_used'] else '否'}")
        if knowledge_result["knowledge_used"]:
            print(f"知识来源: {', '.join(knowledge_result['knowledge_sources'])}")
        print(f"回复内容: {knowledge_result['content'][:300]}...")
    else:
        print(f"❌ 知识增强LLM响应测试失败: {knowledge_result['message']}")

    # 知识库搜索测试
    print("\n📚 测试知识库搜索功能...")
    search_queries = ["安全运营中心", "漏洞管理", "告警类型", "资产中心"]

    for query in search_queries:
        print(f"\n🔍 搜索: '{query}'")
        results = tester.knowledge_base.search_similar_documents(query, top_k=1)
        if results:
            result = results[0]
            print(f"  📄 最相似文档 (相似度: {result['similarity']:.3f}):")
            print(f"  💾 来源: {result['source']}")
            print(f"  📝 内容预览: {result['content'][:100]}...")
        else:
            print("  ❌ 未找到相关文档")

    # 关闭资源
    tester.close()

    print("\n" + "=" * 60)
    print("🎯 Weaviate版本测试完成!")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
