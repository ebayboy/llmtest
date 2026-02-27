#!/usr/bin/env python3
"""
OpenAI LLM API测试工具
使用OpenAI Python SDK实现LLM调用
"""

import os
import sys
import time
from typing import Dict, Any, Optional
import logging
from openai import OpenAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenAILLMTester:
    """OpenAI LLM API测试器"""
    
    def __init__(self):
        """初始化测试器"""
        # 从环境变量获取配置，如果没有则使用默认值
        self.api_url = os.getenv("LLM_API_URL", "http://116.198.229.83:9998/v1")
        self.model_name = os.getenv("LLM_MODEL", "baidu/ERNIE-4.5-21B-A3B-PT")
        self.api_key = os.getenv("LLM_API_KEY", "openai")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )
            
        logger.info(f"初始化LLM测试器: URL={self.api_url}, Model={self.model_name}")
    
    def test_connection(self, timeout: int = 10) -> Dict[str, Any]:
        """测试API连接"""
        logger.info("开始测试API连接...")
        
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}],
                temperature=0.1,
                max_tokens=10,
                timeout=timeout
            )
            response_time = time.time() - start_time
            
            logger.info(f"✅ API连接成功! 响应时间: {response_time:.2f}s")
            return {
                "success": True,
                "response_time": response_time,
                "message": "API连接正常"
            }
                
        except Exception as e:
            logger.error(f"❌ API连接异常: {e}")
            return {
                "success": False,
                "message": f"API连接异常: {str(e)}",
                "error_type": "connection_error"
            }
    
    def test_llm_response(self, prompt: str = "你好，请简单介绍一下自己。",
                         max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
        """测试LLM响应功能，带重试机制"""
        logger.info(f"开始测试LLM响应，提示词: {prompt[:50]}...")
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"第{attempt + 1}次尝试...")
                
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    timeout=30
                )
                response_time = time.time() - start_time
                
                # 提取回复内容
                if response.choices and len(response.choices) > 0 and response.choices[0].message:
                    content = response.choices[0].message.content or ""
                else:
                    content = ""
                logger.info(f"✅ LLM响应成功! 响应时间: {response_time:.2f}s")
                logger.info(f"回复内容: {content[:100]}...")
                
                return {
                    "success": True,
                    "response_time": response_time,
                    "content": content,
                    "full_response": response.model_dump()
                }
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"第{attempt + 1}次尝试异常: {last_error}")
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
        
        logger.error(f"❌ 所有{max_retries}次尝试均失败: {last_error}")
        return {
            "success": False,
            "message": f"所有{max_retries}次尝试均失败: {last_error}",
            "error_type": "max_retries_exceeded"
        }

def main():
    """主函数"""
    print("🚀 OpenAI LLM API测试工具")
    print("=" * 50)
    
    # 创建测试器
    tester = OpenAILLMTester()
    
    # 测试连接
    print("\n📡 测试API连接...")
    connection_result = tester.test_connection()
    
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
    
    llm_result = tester.test_llm_response(test_prompt, max_retries=3)
    
    if llm_result["success"]:
        print(f"✅ LLM响应测试成功!")
        print(f"响应时间: {llm_result['response_time']:.2f}s")
        print(f"回复内容: {llm_result['content']}")
    else:
        print(f"❌ LLM响应测试失败: {llm_result['message']}")
    
    print("\n" + "=" * 50)
    print("🎯 测试完成!")

if __name__ == "__main__":
    main()