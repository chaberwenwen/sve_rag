"""
LLM 客户端 —— v4 适配器（委托给 llm/ 顶层模块）

通过 llm/ 模块调用 LLM，支持 Ollama 本地推理和 DeepSeek 等外接 API。
v4 特有配置（模型名、参数）在此层根据 LLM_BACKEND 注入。

设计说明：
  - 不重复实现 LLM 调用逻辑，全部委托给 llm.client.LLMClient
  - 仅在此层绑定 config.py 的配置项，实现版本隔离
  - 类名和接口与之前完全一致，对 pipeline.py 透明
"""

from llm import LLMClient as _LLMClient
from config import (
    LLM_BACKEND,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DEEPSEEK_API_KEY,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
)


class LLMClient:
    """
    v4 版 LLM 客户端适配器。

    封装 llm.LLMClient，根据 LLM_BACKEND 自动选择后端：
      - 'ollama': 本地 Ollama 推理
      - 'openai': 外接 DeepSeek / 硅基流动等 OpenAI 兼容 API

    对外暴露与原来一致的 call() 接口。
    """

    def __init__(self, base_url: str = None, model: str = None, api_key: str = None):
        if LLM_BACKEND == 'ollama':
            self._client = _LLMClient(
                backend='ollama',
                base_url=base_url or OLLAMA_BASE_URL,
                model=model or OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        elif LLM_BACKEND == 'openai':
            key = api_key or DEEPSEEK_API_KEY
            if not key:
                raise ValueError(
                    'DeepSeek API Key 未设置。\n'
                    '请在项目根目录的 .env 文件中设置：\n'
                    '  DEEPSEEK_API_KEY=你的API密钥\n'
                    '或设置系统环境变量 DEEPSEEK_API_KEY'
                )
            self._client = _LLMClient(
                backend='openai',
                base_url=base_url or DEEPSEEK_BASE_URL,
                model=model or DEEPSEEK_MODEL,
                api_key=key,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        else:
            raise ValueError(
                f'不支持的 LLM_BACKEND: {LLM_BACKEND!r}，'
                f'请使用 "ollama" 或 "openai"'
            )

    def call(self, prompt: str, system_prompt: str = None) -> str:
        """
        调用 LLM API 生成回答。

        Args:
            prompt: 用户提示词（含上下文）
            system_prompt: 系统提示词（角色设定）

        Returns:
            生成的回答文本
        """
        return self._client.call(prompt, system_prompt)
