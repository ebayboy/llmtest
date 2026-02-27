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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        model=self.model_name,
        base_url=self.api_url,
        temperature=self.temperature
    )
        
    logger.info(f"初始化LangChain LLM测试器: URL={self.api_url}, Model={self.model_name}")
    
    async def test_connection(self, timeout: int = 10) -> Dict[str, Any]:
        """测试API连接"""
        logger.info("开始测试API连接...")
        
        try:
            start_time = time.time()
            response = await self.llm.ainvoke([HumanMessage(content="ping")])
            response_time = time.time() - start_time
            
            # 提取回复内容
            content = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"✅ API连接成功! 响应时间: {response_time:.2f}s")
            return {
                "success": True,
                "response_time": response_time,
                "message": "API连接正常",
                "response_content": content[:100] if content else ""
            }
                
        except Exception as e:
            logger.error(f"❌ API连接异常: {e}")
            return {
                "success": False,
                "message": f"API连接异常: {str(e)}",
                "error_type": "connection_error"
            }
    
    async def test_llm_response(self, prompt: str = "你好，请简单介绍一下自己。",
                              max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
        """测试LLM响应功能，带重试机制"""
        logger.info(f"开始测试LLM响应，提示词: {prompt[:50]}...")
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"第{attempt + 1}次尝试...")
                
                start_time = time.time()
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                response_time = time.time() - start_time
                
                # 提取回复内容
                content = response.content if hasattr(response, 'content') else str(response)
                
                logger.info(f"✅ LLM响应成功! 响应时间: {response_time:.2f}s")
                logger.info(f"回复内容: {content[:100]}...")
                
                return {
                    "success": True,
                    "response_time": response_time,
                    "content": content,
                    "attempt": attempt + 1
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
            "error_type": "max_retries_exceeded"
        }
    
    async def test_streaming_response(self, prompt: str = "请详细介绍一下人工智能的发展历史。") -> Dict[str, Any]:
        """测试流式响应功能"""
        logger.info(f"开始测试流式响应，提示词: {prompt[:50]}...")
        
        try:
            start_time = time.time()
            
            # 使用标准OpenAI接口的流式响应
            messages = [HumanMessage(content=prompt)]
            response_chunks = []
            
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    response_chunks.append(chunk.content)
            
            response_time = time.time() - start_time
            full_content = "".join(response_chunks)
            
            logger.info(f"✅ 流式响应成功! 响应时间: {response_time:.2f}s")
            logger.info(f"回复内容: {full_content[:100]}...")
            
            return {
                "success": True,
                "response_time": response_time,
                "content": full_content,
                "streaming": True
            }
                
        except Exception as e:
            logger.error(f"❌ 流式响应异常: {e}")
            return {
                "success": False,
                "message": f"流式响应异常: {str(e)}",
                "error_type": "streaming_error"
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
    
    # 测试LLM响应
    print("\n🤖 测试LLM响应...")
    
    # 使用更详细的测试提示词
    test_prompt = "你好！请简单介绍一下自己，并告诉我你能如何帮助用户进行数据安全分析。"
    
    llm_result = await tester.test_llm_response(test_prompt, max_retries=3)
    
    if llm_result["success"]:
        print(f"✅ LLM响应测试成功!")
        print(f"响应时间: {llm_result['response_time']:.2f}s")
        print(f"回复内容: {llm_result['content']}")
    else:
        print(f"❌ LLM响应测试失败: {llm_result['message']}")
    
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
    
    print("\n" + "=" * 50)
    print("🎯 测试完成!")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())