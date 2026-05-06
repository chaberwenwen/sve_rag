"""
统一配置文件 —— 所有路径、模型名、检索参数集中管理

面试要点：
  - 单文件配置，避免魔法值散布在各模块
  - 路径使用 os.path 动态计算，保证可移植性
  - 后续版本可升级为 pydantic-settings / .env 文件管理
"""

import os

# ============================================================
# 项目根目录 (v4/)
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 项目路径
# ============================================================
BASE_DIR = os.path.dirname(PROJECT_DIR)            # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')           # 数据目录

# 注：卡牌/QA/规则的数据加载路径由 data/loader.py 独立管理，
#     此处仅保留 BASE_DIR 和 DATA_DIR 供 .env 加载等用途。

# ============================================================
# 索引存储路径（版本隔离，不同版本有各自独立的 index/ 目录）
# ============================================================
INDEX_DIR = os.path.join(PROJECT_DIR, 'index')

# ============================================================
# Embedding 模型配置
# ============================================================
# bge-small-zh-v1.5: BAAI 开源中文 Embedding 模型，33MB，兼顾中英文
# 选择理由：轻量（适合本地开发）、支持中文语义、FAISS内积兼容
EMBED_MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
# 本地模型保存路径（离线优先，避免重复下载）
EMBED_MODEL_LOCAL_DIR = os.path.join(PROJECT_DIR, 'models', 'bge-small-zh-v1.5')

# ============================================================
# LLM 后端选择
# 可选值: 'ollama' (本地) 或 'openai' (外接 API, 如 DeepSeek / 硅基流动等)
# ============================================================
LLM_BACKEND = 'openai'               # ← 切换为外接 DeepSeek API

# ============================================================
# Ollama 本地推理配置
# 端口 11434 是 Ollama 的默认端口，如果冲突可修改
# ============================================================
OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_MODEL = 'qwen2.5:7b'          # 7B 参数量，本地 RTX 3060+ 可跑

# ============================================================
# DeepSeek API 配置（OpenAI 兼容后端）
# 当 LLM_BACKEND='openai' 时生效
# API Key 通过环境变量 DEEPSEEK_API_KEY 或 .env 文件设置
# ============================================================
DEEPSEEK_BASE_URL = 'https://api.deepseek.com/v1'
DEEPSEEK_MODEL = 'deepseek-v4-flash'  # deepseek-v4-flash: 最新 Flash 级模型

# 尝试从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    _ENV_FILE = os.path.join(BASE_DIR, '.env')
    if os.path.exists(_ENV_FILE):
        load_dotenv(_ENV_FILE)
except ImportError:
    pass  # python-dotenv 未安装时退化为纯系统环境变量

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')

# ============================================================
# LoRA 微调配置（可插拔，方便 A/B 对比效果）
# ============================================================
LORA_ENABLED = False                 # False=原始BGE  True=加载LoRA adapter
LORA_MODEL_DIR = os.path.join(      # LoRA adapter 保存/加载路径
    PROJECT_DIR, 'finetune', 'model',
)

# ============================================================
# 检索参数
# ============================================================
TOP_K = 8                             # 向量检索返回的 Top-K 文档数

# 内容索引分数过滤（双层过滤，防止 LoRA 训练偏差产生的中等分噪声）
# 绝对阈值：低于此分直接丢弃；0 表示禁用
CONTENT_SCORE_MIN_THRESHOLD = 0.55
# 相对比值阈值：仅保留 score ≥ 最高分 × RATIO 的结果（参考 NAME 阈值逻辑）
# 目的：当 LoRA 把所有分数挤到狭窄区间（如 0.60~0.66），通过比值拉大区分度
CONTENT_SCORE_RATIO_THRESHOLD = 0.85

# 名称索引命中分数阈值：仅保留 score ≥ 最高分 × RATIO 的命中
# 低于阈值的低分噪音卡会污染 effect_query，把内容检索方向带偏
# 典型场景：用户问2张卡，search_names top-5 混入无关卡（如"减价交涉"0.43）
# 参考值 0.80 = 打8折以内都保留（0.55×0.80=0.44，0.43被筛掉）
NAME_SCORE_RATIO_THRESHOLD = 0.80

# ============================================================
# LLM 生成参数
# ============================================================
LLM_TEMPERATURE = 0.3                 # 低温度保证回答准确性
LLM_MAX_TOKENS = 2048

# ============================================================
# Prompt 配置
# ============================================================
SYSTEM_PROMPT = (
    '你是一个影之诗 EVOLVE 卡牌游戏的中文规则助手。'
    '根据提供的卡牌信息和规则 Q&A 回答问题。'
    '用中文回答，保持准确清晰。信息不足时诚实说明。'
    '回答时，优先引用向量检索结果中与问题最相关的 1~3 条 Q&A，'
    '明确指出每条 Q&A 涉及的卡牌名称，并说明为什么这条规则适用于当前问题。'
)
