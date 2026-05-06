"""
llm 模块 —— 大语言模型调用抽象层

独立于任何版本，支持多种 LLM 后端（Ollama / OpenAI 兼容 API）。
可被 v3、v4 等任意版本直接导入使用。

用法:
    from llm import LLMClient

    # Ollama 后端（默认）
    llm = LLMClient(model='qwen2.5:7b')

    # OpenAI 兼容 API
    llm = LLMClient(
        backend='openai',
        base_url='https://api.openai.com/v1',
        model='gpt-4o-mini',
        api_key='sk-...',
    )

    answer = llm.call('你好，请介绍一下自己')
"""

from .client import LLMClient, OllamaBackend, OpenAICompatibleBackend

__all__ = ['LLMClient', 'OllamaBackend', 'OpenAICompatibleBackend']
