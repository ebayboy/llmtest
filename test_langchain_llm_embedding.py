#!/usr/bin/env python3
"""
LangChain LLM API测试工具
使用LangChain接口实现LLM调用
"""

import os
import sys
import time
from typing import Dict, Any, Optional
import logging
import asyncio

# 导入LangChain相关模块
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 导入知识库相关模块
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KnowledgeBase:
    """知识库管理类"""

    def __init__(self, knowledge_dir: str = "knowledge"):
        """初始化知识库"""
        self.knowledge_dir = knowledge_dir
        self.documents = []
        self.vectorizer = None
        self.doc_vectors = None
        self.load_documents()

    def load_documents(self):
        """加载知识文档"""
        logger.info(f"开始加载知识库文档，目录: {self.knowledge_dir}")

        try:
            # 获取所有txt文件
            txt_files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
            logger.info(f"找到 {len(txt_files)} 个知识文档")

            for file_path in txt_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # 按段落分割文档，避免单个文档过长
                        paragraphs = content.split("\n\n")
                        for para in paragraphs:
                            para = para.strip()
                            if len(para) > 50:  # 只保留长度大于50的段落
                                self.documents.append(
                                    {
                                        "content": para,
                                        "source": os.path.basename(file_path),
                                    }
                                )
                    logger.info(f"✅ 加载文档成功: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"❌ 加载文档失败 {file_path}: {e}")

            logger.info(f"📚 知识库加载完成，共 {len(self.documents)} 个文档段落")

            # 构建向量索引
            if self.documents:
                self.build_vector_index()

        except Exception as e:
            logger.error(f"❌ 知识库加载异常: {e}")

    def build_vector_index(self):
        """构建向量索引"""
        try:
            logger.info("开始构建向量索引...")

            # 使用TF-IDF向量化文档
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words=None,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
            )

            # 提取文档内容
            doc_contents = [doc["content"] for doc in self.documents]

            # 构建向量
            self.doc_vectors = self.vectorizer.fit_transform(doc_contents)

            logger.info(f"✅ 向量索引构建完成，维度: {self.doc_vectors.shape}")

        except Exception as e:
            logger.error(f"❌ 向量索引构建失败: {e}")
            self.vectorizer = None
            self.doc_vectors = None

    def search_similar_documents(self, query: str, top_k: int = 3) -> list:
        """搜索相似文档"""
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
                f"🔍 搜索到 {len(results)} 个相似文档，最高相似度: {similarities[top_indices[0]]:.3f}"
            )
            return results

        except Exception as e:
            logger.error(f"❌ 文档搜索失败: {e}")
            return []


# TODO: 启动时读取knowledg目录下的txt文件，构建向量知识库， 根据用户输入内容从向量知识库中召回相关知识，并在测试LLM响应时将相关知识作为系统提示词传入，以测试LLM的知识调用能力。


class LangChainLLMTester:
    """LangChain LLM API测试器"""

    def __init__(self):
        """初始化测试器"""
        # 从环境变量获取配置，如果没有则使用默认值
        self.api_url = os.getenv("LLM_API_URL", "http://116.198.229.83:9998/v1")
        self.model_name = os.getenv("LLM_MODEL", "baidu/ERNIE-4.5-21B-A3B-PT")
        self.api_key = os.getenv("LLM_API_KEY", "openai")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

        # 设置环境变量
        os.environ["OPENAI_API_KEY"] = self.api_key

        # 初始化LangChain LLM客户端
        self.llm = ChatOpenAI(
            model=self.model_name, base_url=self.api_url, temperature=self.temperature
        )

        # 初始化知识库
        self.knowledge_base = KnowledgeBase()

        logger.info(
            f"初始化LangChain LLM测试器: URL={self.api_url}, Model={self.model_name}"
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
        if use_knowledge and self.knowledge_base:
            logger.info("🔍 正在检索相关知识...")
            similar_docs = self.knowledge_base.search_similar_documents(prompt, top_k=2)

            if similar_docs:
                knowledge_context = "\n\n".join(
                    [doc["content"] for doc in similar_docs]
                )
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
                    "knowledge_sources": (
                        [doc["source"] for doc in similar_docs]
                        if knowledge_context
                        else []
                    ),
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

    async def test_streaming_response(
        self, prompt: str = "请详细介绍一下人工智能的发展历史。"
    ) -> Dict[str, Any]:
        """测试流式响应功能"""
        logger.info(f"开始测试流式响应，提示词: {prompt[:50]}...")

        try:
            start_time = time.time()

            # 使用标准OpenAI接口的流式响应
            messages = [HumanMessage(content=prompt)]
            response_chunks = []

            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    response_chunks.append(chunk.content)

            response_time = time.time() - start_time
            full_content = "".join(response_chunks)

            logger.info(f"✅ 流式响应成功! 响应时间: {response_time:.2f}s")
            logger.info(f"回复内容: {full_content[:100]}...")

            return {
                "success": True,
                "response_time": response_time,
                "content": full_content,
                "streaming": True,
            }

        except Exception as e:
            logger.error(f"❌ 流式响应异常: {e}")
            return {
                "success": False,
                "message": f"流式响应异常: {str(e)}",
                "error_type": "streaming_error",
            }


async def main():
    """主函数"""
    print("🚀 LangChain LLM API测试工具")
    print("=" * 50)

    # 创建测试器
    tester = LangChainLLMTester()

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

    # 测试流式响应
    print("\n🌊 测试流式响应...")
    streaming_result = await tester.test_streaming_response(
        "请详细介绍一下数据安全分析的重要性。"
    )

    if streaming_result["success"]:
        print(f"✅ 流式响应测试成功!")
        print(f"响应时间: {streaming_result['response_time']:.2f}s")
        print(f"回复内容: {streaming_result['content'][:200]}...")
    else:
        print(f"❌ 流式响应测试失败: {streaming_result['message']}")

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

    print("\n" + "=" * 50)
    print("🎯 测试完成!")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
