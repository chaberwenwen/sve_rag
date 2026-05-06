"""
LLM 客户端 —— 多后端抽象的大模型调用层

设计原则：
  - 与 v3/v4 等版本无关，独立可复用
  - 后端可插拔：Ollama（本地）、OpenAI 兼容 API（云端）
  - 所有参数提供合理默认值，零配置即可运行
  - 统一的错误处理，返回友好提示而非抛异常

用法:
    # 默认 Ollama 后端
    llm = LLMClient()
    answer = llm.call('BP01-001 的效果是什么？')

    # 指定模型
    llm = LLMClient(model='qwen2.5:7b')

    # OpenAI 兼容后端
    llm = LLMClient(
        backend='openai',
        base_url='https://api.openai.com/v1',
        model='gpt-4o-mini',
        api_key='sk-...',
    )
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional


# ============================================================
# 默认配置
# ============================================================
DEFAULT_OLLAMA_URL = 'http://localhost:11434'
DEFAULT_OLLAMA_MODEL = 'qwen2.5:7b'
DEFAULT_OPENAI_MODEL = 'gpt-4o-mini'
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 60


# ============================================================
# 后端抽象基类
# ============================================================
class BaseBackend(ABC):
    """LLM 后端抽象基类"""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """调用 LLM 生成回答"""


# ============================================================
# Ollama 后端
# ============================================================
class OllamaBackend(BaseBackend):
    """Ollama 本地推理后端"""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import requests

        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': self.temperature,
                'top_p': 0.9,
                'max_tokens': self.max_tokens,
            },
        }
        if system_prompt:
            payload['system'] = system_prompt

        try:
            resp = requests.post(
                f'{self.base_url}/api/generate',
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get('response', '')
        except requests.exceptions.ConnectionError:
            return '[错误: 无法连接 Ollama，请确保 Ollama 服务已启动]'
        except requests.exceptions.Timeout:
            return '[错误: Ollama 响应超时]'
        except Exception as e:
            return f'[Ollama 调用失败: {e}]'


# ============================================================
# OpenAI 兼容后端（支持 OpenAI / 硅基流动 / DeepSeek API 等）
# ============================================================
class OpenAICompatibleBackend(BaseBackend):
    """OpenAI 兼容 API 后端（OpenAI / 硅基流动 / DeepSeek / 通义千问 等）"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = '',
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        super().__init__(base_url, model, temperature, max_tokens, timeout)
        self.api_key = api_key

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import requests

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        headers = {
            'Content-Type': 'application/json',
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }

        try:
            resp = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get('choices', [])
            if choices:
                return choices[0].get('message', {}).get('content', '')
            return ''
        except requests.exceptions.ConnectionError:
            return f'[错误: 无法连接 API {self.base_url}]'
        except requests.exceptions.Timeout:
            return '[错误: API 响应超时]'
        except Exception as e:
            return f'[API 调用失败: {e}]'


# ============================================================
# LLM 客户端（统一入口）
# ============================================================
class LLMClient:
    """
    大模型调用客户端 —— 后端无关的统一接口。

    Args:
        backend: 后端类型，'ollama'（默认）或 'openai'
        base_url: API 地址
        model: 模型名称
        api_key: OpenAI 兼容后端的 API Key
        temperature: 生成温度 (0-1)，越低越确定
        max_tokens: 最大生成 token 数
        timeout: 请求超时秒数

    用法:
        # Ollama
        llm = LLMClient()
        llm = LLMClient(model='qwen2.5:7b')

        # OpenAI 兼容
        llm = LLMClient(
            backend='openai',
            base_url='https://api.openai.com/v1',
            model='gpt-4o-mini',
            api_key='sk-xxx',
        )
    """

    def __init__(
        self,
        backend: str = 'ollama',
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.backend_type = backend

        if backend == 'ollama':
            self._backend: BaseBackend = OllamaBackend(
                base_url=base_url or DEFAULT_OLLAMA_URL,
                model=model or DEFAULT_OLLAMA_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif backend == 'openai':
            self._backend = OpenAICompatibleBackend(
                base_url=base_url or 'https://api.openai.com/v1',
                model=model or DEFAULT_OPENAI_MODEL,
                api_key=api_key or '',
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            raise ValueError(f'不支持的 backend 类型: {backend}，请使用 "ollama" 或 "openai"')

    @property
    def model(self) -> str:
        return self._backend.model

    @property
    def base_url(self) -> str:
        return self._backend.base_url

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        调用 LLM 生成回答。

        Args:
            prompt: 用户提示词（含上下文）
            system_prompt: 系统提示词（角色设定）

        Returns:
            生成的回答文本
        """
        return self._backend.generate(prompt, system_prompt)
