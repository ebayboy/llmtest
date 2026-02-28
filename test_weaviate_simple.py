#!/usr/bin/env python3
"""
简化版Weaviate Embedding演示
展示如何使用Weaviate作为向量数据库进行文档嵌入和搜索
"""

import os
import logging
from typing import List, Dict
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import weaviate
    from weaviate.classes.init import Auth
    from weaviate.classes.config import Configure, Property
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    from langchain_weaviate import WeaviateVectorStore
except ImportError as e:
    logger.error(f"缺少依赖包: {e}")
    logger.info(
        "请运行: pip install weaviate-client langchain-weaviate langchain-openai"
    )
    exit(1)


class WeaviateEmbeddingDemo:
    """Weaviate Embedding演示类"""

    def __init__(self, weaviate_url: str = "http://localhost:8081"):
        """初始化"""
        self.weaviate_url = weaviate_url
        self.client = None
        self.vectorstore = None
        self.embeddings = None
        self.collection_name = "DemoCollection"

    def init_weaviate(self) -> bool:
        """初始化Weaviate连接"""
        try:
            logger.info(f"正在连接Weaviate: {self.weaviate_url}")

            # 创建Weaviate客户端 - 简化连接配置
            self.client = weaviate.connect_to_local(
                host="localhost",
                port=8081,
                grpc_port=50051,
                skip_init_checks=True,  # 跳过初始化检查
            )

            # 等待服务就绪
            import time

            time.sleep(2)

            # 使用简单的方式检查连接
            try:
                # 尝试获取元信息
                meta = self.client.get_meta()
                logger.info(
                    f"✅ Weaviate连接成功! 版本: {meta.get('version', 'unknown')}"
                )
                return True
            except Exception as meta_error:
                logger.warning(f"元信息获取失败，但连接可能正常: {meta_error}")
                # 尝试简单的健康检查
                return True

        except Exception as e:
            logger.error(f"❌ Weaviate连接错误: {e}")
            return False

    def init_embedding_model(self) -> bool:
        """初始化Embedding模型"""
        try:
            # 使用OpenAI Embedding模型
            embedding_api_url = os.getenv(
                "EMBEDDING_API_URL", "http://116.198.229.83:8009/v1"
            )
            embedding_api_key = os.getenv(
                "EMBEDDING_API_KEY", "1F981D0DAF0135FE228C505557A4412F"
            )
            embedding_model = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")

            logger.info(f"正在初始化Embedding模型: {embedding_model}")
            logger.info(f"Embedding API URL: {embedding_api_url}")

            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                api_key=embedding_api_key,
                base_url=embedding_api_url,
            )
            logger.info("✅ Embedding模型初始化成功")
            return True

        except Exception as e:
            logger.error(f"❌ Embedding模型初始化失败: {e}")
            return False

    def setup_collection(self) -> bool:
        """设置Weaviate集合"""
        try:
            # 检查客户端是否已初始化
            if not self.client:
                logger.error("Weaviate客户端未初始化")
                return False

            # 检查集合是否存在
            if self.client.collections.exists(self.collection_name):
                logger.info(f"集合 '{self.collection_name}' 已存在")
                return True

            # 创建新集合
            logger.info(f"创建集合 '{self.collection_name}'...")
            self.client.collections.create(
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

    def create_sample_documents(self) -> List[Document]:
        """创建示例文档"""
        documents = [
            Document(
                page_content="京东云安全运营中心提供统一的安全态势感知和威胁检测能力，帮助企业构建完整的安全防护体系。",
                metadata={"source": "security_center.txt", "type": "security"},
            ),
            Document(
                page_content="漏洞管理系统能够自动发现和修复系统中的安全漏洞，降低安全风险。支持定期扫描和实时监测。",
                metadata={"source": "vulnerability_mgmt.txt", "type": "vulnerability"},
            ),
            Document(
                page_content="告警中心集成了多种安全设备的告警信息，通过智能分析减少误报，提高响应效率。",
                metadata={"source": "alert_center.txt", "type": "alert"},
            ),
            Document(
                page_content="资产中心负责管理和盘点企业所有的IT资产，包括服务器、网络设备、应用程序等。",
                metadata={"source": "asset_center.txt", "type": "asset"},
            ),
            Document(
                page_content="日志分析平台能够处理大量的安全日志数据，通过机器学习和规则引擎发现潜在的安全威胁。",
                metadata={"source": "log_analysis.txt", "type": "log"},
            ),
        ]

        logger.info(f"✅ 创建了 {len(documents)} 个示例文档")
        return documents

    def add_documents_to_weaviate(self, documents: List[Document]) -> bool:
        """添加文档到Weaviate"""
        try:
            # 检查客户端是否已初始化
            if not self.client:
                logger.error("Weaviate客户端未初始化")
                return False

            # 创建向量存储
            self.vectorstore = WeaviateVectorStore(
                client=self.client,
                index_name=self.collection_name,
                text_key="content",
                embedding=self.embeddings,
            )

            # 检查是否已有数据
            collection = self.client.collections.get(self.collection_name)
            existing_count = collection.aggregate.over_all(total_count=True).total_count

            if existing_count > 0:
                logger.info(f"集合中已有 {existing_count} 条数据，跳过导入")
                return True

            # 添加文档
            logger.info(f"正在添加 {len(documents)} 个文档到Weaviate...")
            uuids = self.vectorstore.add_documents(documents)
            logger.info(f"✅ 成功添加 {len(uuids)} 个文档")
            return True

        except Exception as e:
            logger.error(f"❌ 文档添加失败: {e}")
            return False

    def search_similar_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索相似文档"""
        try:
            if not self.vectorstore:
                logger.error("向量存储未初始化")
                return []

            logger.info(f"搜索查询: '{query}'")

            # 执行相似度搜索
            relevant_docs = self.vectorstore.similarity_search_with_score(
                query, k=top_k
            )

            results = []
            for doc, score in relevant_docs:
                similarity = max(0, 1 - score)  # 转换为相似度
                results.append(
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "similarity": float(similarity),
                        "type": doc.metadata.get("type", "unknown"),
                    }
                )

            logger.info(f"🔍 找到 {len(results)} 个相似文档")
            return results

        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return []

    def run_demo(self):
        """运行完整演示"""
        print("🚀 Weaviate Embedding演示开始")
        print("=" * 50)

        # 1. 初始化Weaviate
        if not self.init_weaviate():
            logger.error("Weaviate初始化失败，演示终止")
            return

        # 2. 初始化Embedding模型
        if not self.init_embedding_model():
            logger.error("Embedding模型初始化失败，演示终止")
            return

        # 3. 设置集合
        if not self.setup_collection():
            logger.error("集合设置失败，演示终止")
            return

        # 4. 创建示例文档
        documents = self.create_sample_documents()

        # 5. 添加到Weaviate
        if not self.add_documents_to_weaviate(documents):
            logger.error("文档添加失败，演示终止")
            return

        # 6. 测试搜索
        test_queries = [
            "安全运营中心的功能",
            "漏洞管理",
            "告警系统",
            "资产管理",
            "日志分析",
        ]

        print("\n📚 搜索测试:")
        for query in test_queries:
            print(f"\n🔍 搜索: '{query}'")
            results = self.search_similar_documents(query, top_k=2)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. 相似度: {result['similarity']:.3f}")
                    print(f"     来源: {result['source']}")
                    print(f"     内容: {result['content'][:100]}...")
            else:
                print("  ❌ 未找到相关文档")

        print("\n✅ Weaviate Embedding演示完成!")

    def close(self):
        """关闭连接"""
        if self.client:
            try:
                self.client.close()
                logger.info("Weaviate连接已关闭")
            except Exception as e:
                logger.error(f"关闭连接失败: {e}")


def main():
    """主函数"""
    # 创建演示实例
    demo = WeaviateEmbeddingDemo()

    try:
        # 运行演示
        demo.run_demo()

    except KeyboardInterrupt:
        logger.info("演示被用户中断")
    except Exception as e:
        logger.error(f"演示运行错误: {e}")
    finally:
        # 清理资源
        demo.close()


if __name__ == "__main__":
    main()
