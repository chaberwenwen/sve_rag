"""
异步 LLM 客户端 —— 将同步 requests 替换为 httpx.AsyncClient。

支持：
  - DeepSeek API（OpenAI 兼容）
  - Ollama 本地推理
  - 超时 + 自动重试（exponential backoff）
  - 连接池管理
"""

import asyncio
import logging
from typing import Optional

import httpx

from server.config import settings

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    """异步 LLM 调用客户端"""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)

    async def _get_client(self) -> httpx.AsyncClient:
        """懒加载 httpx 客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.llm_timeout),
                limits=httpx.Limits(
                    max_connections=settings.max_concurrent_llm_calls + 2,
                    max_keepalive_connections=2,
                ),
            )
        return self._client

    async def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 2,
    ) -> str:
        """
        调用 LLM 生成回答（带并发控制 + 重试）。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            max_retries: 最大重试次数

        Returns:
            LLM 生成的回答文本，失败时返回错误提示
        """
        async with self._semaphore:
            for attempt in range(max_retries + 1):
                try:
                    return await self._call_inner(prompt, system_prompt)
                except Exception as e:
                    logger.warning(
                        "LLM call attempt %d/%d failed: %s",
                        attempt + 1,
                        max_retries + 1,
                        str(e),
                    )
                    if attempt < max_retries:
                        # Exponential backoff: 1s, 2s, 4s...
                        await asyncio.sleep(2 ** attempt)

            return f"[错误: LLM 调用失败，已重试 {max_retries} 次]"

    async def _call_inner(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """实际执行一次 LLM 调用"""
        client = await self._get_client()

        if settings.llm_provider == "ollama":
            return await self._call_ollama(client, prompt, system_prompt)
        else:
            return await self._call_openai_compatible(
                client, prompt, system_prompt
            )

    async def _call_ollama(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Ollama 本地 API"""
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    async def _call_openai_compatible(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """OpenAI 兼容 API（DeepSeek 等）"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if settings.deepseek_api_key:
            headers["Authorization"] = f"Bearer {settings.deepseek_api_key}"

        payload = {
            "model": settings.deepseek_model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }

        resp = await client.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    async def health_check(self) -> bool:
        """检查 LLM API 连通性（轻量测试）"""
        try:
            client = await self._get_client()
            if settings.llm_provider == "ollama":
                resp = await client.get(settings.ollama_base_url)
                return resp.status_code == 200
            else:
                headers = {}
                if settings.deepseek_api_key:
                    headers["Authorization"] = f"Bearer {settings.deepseek_api_key}"
                resp = await client.get(
                    f"{settings.deepseek_base_url}/models",
                    headers=headers,
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        """关闭 HTTP 客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


# 全局单例
_llm_client: Optional[AsyncLLMClient] = None


def get_llm_client() -> AsyncLLMClient:
    """获取全局异步 LLM 客户端单例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncLLMClient()
    return _llm_client
