"""
Embedding 模型管理 —— BGE 模型单例封装（不依赖 sentence_transformers）

面试要点：
  - 单例模式：全局只加载一次模型，避免重复加载
  - 本地优先：优先从 v4/models/ 加载，其次 HF 缓存，最后 hf-mirror
  - 延迟加载：不 import 时就加载，只在首次调用 encode() 时初始化
  - mean pooling：与 BGE 官方一致的 attention-weighted mean pooling
  - v4 改进点：可插拔 LoRA adapter（通过 PEFT 注入）
"""

import os

import numpy as np

# EMBED_MODEL_NAME 不变，LORA_ENABLED/LORA_MODEL_DIR 在 get_embedder() 内动态读取
# 以支持 eval_embedder 运行时切换
from config import EMBED_MODEL_NAME

# 全局单例
_embed_model = None          # PeftModel 或原始 AutoModel
_tokenizer = None            # AutoTokenizer
_device = None               # torch device
_hidden_size = None          # embedding dimension
_lora_loaded = False


def get_embedder():
    """
    延迟加载模型（transformers.AutoModel + AutoTokenizer）。

    加载优先级：
      1. 项目本地 v4/models/bge-small-zh-v1.5（离线可用）
      2. HuggingFace 本地缓存
      3. hf-mirror.com 镜像源

    若 LORA_ENABLED=True，且 LORA_MODEL_DIR 存在，则通过 PeftModel.from_pretrained
    注入 LoRA adapter，后续编码自动使用微调后的权重。
    """
    global _embed_model, _tokenizer, _device, _hidden_size, _lora_loaded
    if _embed_model is not None:
        return _embed_model, _tokenizer, _device

    import torch
    from transformers import AutoModel, AutoTokenizer
    from config import EMBED_MODEL_LOCAL_DIR as LOCAL_DIR

    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

    # 确定加载路径：本地目录优先
    if os.path.isdir(LOCAL_DIR):
        model_path = LOCAL_DIR
        print(f'加载 Embedding 模型 (本地): {model_path}')
    else:
        model_path = EMBED_MODEL_NAME
        print(f'加载 Embedding 模型 (HF): {model_path}')

    # ---- Tokenizer ----
    _tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ---- AutoModel ----
    try:
        base_model = AutoModel.from_pretrained(
            model_path,
            local_files_only=(model_path == LOCAL_DIR),
        )
    except Exception as e:
        print(f'  本地加载失败 ({e})，切换至镜像源...')
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        base_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

    _hidden_size = base_model.config.hidden_size
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = base_model.to(_device)
    base_model.eval()

    print(f'  模型维度: {_hidden_size}')
    print(f'  设备: {_device}')

    # ---- 注入 LoRA adapter（动态读取 config，支持运行时切换）----
    import config
    if config.LORA_ENABLED:
        lora_dir = config.LORA_MODEL_DIR
        if not os.path.exists(lora_dir):
            print(f'  [WARNING] LoRA adapter 不存在: {lora_dir}')
        else:
            try:
                from peft import PeftModel
                base_model = PeftModel.from_pretrained(base_model, lora_dir)
                base_model = base_model.to(_device)
                base_model.eval()
                _lora_loaded = True
                print(f'  已注入 LoRA adapter ← {lora_dir}')
            except ImportError:
                print('  [WARNING] peft 未安装，无法加载 LoRA adapter')

    _embed_model = base_model
    return _embed_model, _tokenizer, _device


def _mean_pooling(token_embeddings, attention_mask):
    """
    attention-weighted mean pooling（与 BGE 官方一致）。

    Args:
        token_embeddings: (B, L, D)
        attention_mask:  (B, L)
    Returns:
        (B, D)
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask_expanded).sum(dim=1)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(texts: list[str], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
    """
    批量编码文本为嵌入向量。

    使用 attention-weighted mean pooling → L2 normalize，
    与 BGE 官方 encode() 行为一致。
    """
    import torch

    model, tokenizer, device = get_embedder()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            pooled = _mean_pooling(out.last_hidden_state, enc['attention_mask'])

        if normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        all_embeddings.append(pooled.cpu().numpy())

        if len(texts) > batch_size:
            print(f'  编码进度: {min(i + batch_size, len(texts))}/{len(texts)}')

    return np.concatenate(all_embeddings, axis=0)


def encode_query(query: str, normalize: bool = True) -> np.ndarray:
    """编码单条查询文本。"""
    return encode_texts([query], normalize=normalize)


def reload_embedder():
    """
    强制重新加载模型（运行时切换 LoRA 开关时调用）。

    使用方式:
        from config import LORA_ENABLED
        LORA_ENABLED = True
        reload_embedder()  # 此时起 encode 使用 LoRA 版本
    """
    global _embed_model, _tokenizer, _device, _hidden_size, _lora_loaded
    _embed_model = None
    _tokenizer = None
    _device = None
    _hidden_size = None
    _lora_loaded = False
    return get_embedder()
