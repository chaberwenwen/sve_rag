"""
服务端统一配置 —— 基于 pydantic-settings，支持 .env 文件和环境变量覆盖。

所有敏感信息（API Key 等）通过环境变量注入，不写死在代码中。
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """服务端全局配置"""

    # ============================================================
    # LLM 全局开关
    # ============================================================
    llm_enabled: bool = True               # false 时仅返回向量检索结果，不调用任何 LLM API

    # ============================================================
    # LLM 后端配置
    # ============================================================
    llm_provider: str = "deepseek"          # deepseek / ollama
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # LLM 生成参数
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    llm_timeout: int = 60

    # ============================================================
    # 速率限制
    # ============================================================
    global_rate_limit_per_minute: int = 60      # 全局 API 请求限制
    llm_rate_limit_per_minute: int = 20         # LLM 调用限制（LLM 关闭时自动跳过）
    max_concurrent_llm_calls: int = 3           # 并发 LLM 调用上限

    # ============================================================
    # 安全
    # ============================================================
    max_input_length: int = 500

    # ============================================================
    # 模型与索引路径（相对于项目根目录）
    # ============================================================
    model_path: str = "v4/models/bge-small-zh-v1.5"
    lora_adapter_path: str = "v4/finetune/model"
    index_dir: str = "v4/index"

    # ============================================================
    # LoRA
    # ============================================================
    lora_enabled: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# 全局单例
settings = Settings()
