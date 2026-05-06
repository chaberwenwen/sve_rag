"""
Embedding 模型管理 —— BGE 双模型架构（原始 + LoRA）

面试要点：
  - 双模型分离：名称索引用原始 BGE，内容索引用 LoRA BGE
  - 各自独立加载，向量空间互不干扰
  - 若 LORA_ENABLED=False，LoRA 端自动回退到原始 BGE
  - 延迟加载：首次调用才初始化，避免不必要的显存占用
  - mean pooling：与 BGE 官方一致的 attention-weighted mean pooling
"""

import os

import numpy as np

from config import EMBED_MODEL_NAME


# ============================================================
# 全局单例（双模型）
# ============================================================
_base_model = None       # 原始 AutoModel（名称索引用）
_lora_model = None       # PeftModel 或原始 AutoModel（内容索引用）
_tokenizer = None        # Tokenizer（两者共用）
_device = None           # torch device
_hidden_size = None      # embedding dimension
_lora_loaded = False


def _load_base_model():
    """加载原始 BGE 模型（不带 LoRA）。"""
    global _base_model, _tokenizer, _device, _hidden_size
    if _base_model is not None:
        return

    import torch
    from transformers import AutoModel, AutoTokenizer
    from config import EMBED_MODEL_LOCAL_DIR as LOCAL_DIR

    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

    # 确定加载路径：本地目录优先
    if os.path.isdir(LOCAL_DIR):
        model_path = LOCAL_DIR
        print(f'加载 Embedding 模型 [原始BGE] (本地): {model_path}')
    else:
        model_path = EMBED_MODEL_NAME
        print(f'加载 Embedding 模型 [原始BGE] (HF): {model_path}')

    # ---- Tokenizer（两者共用）----
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

    _base_model = base_model
    print(f'  模型维度: {_hidden_size}')
    print(f'  设备: {_device}')


def _load_lora_model():
    """加载 LoRA 模型（基于原始 BGE + PEFT adapter）。若 LORA_ENABLED=False 则回退到原始 BGE。"""
    global _lora_model, _lora_loaded
    if _lora_model is not None:
        return

    import config

    # 确保原始模型已加载
    _load_base_model()

    if not config.LORA_ENABLED:
        # LoRA 未启用：直接复用原始模型
        _lora_model = _base_model
        _lora_loaded = False
        print(f'  [内容编码] LoRA 未启用，使用原始 BGE')
        return

    lora_dir = config.LORA_MODEL_DIR
    if not os.path.exists(lora_dir):
        print(f'  [WARNING] LoRA adapter 不存在: {lora_dir}，回退到原始 BGE')
        _lora_model = _base_model
        _lora_loaded = False
        return

    try:
        from peft import PeftModel
        lora_model = PeftModel.from_pretrained(_base_model, lora_dir)
        lora_model = lora_model.to(_device)
        lora_model.eval()
        _lora_model = lora_model
        _lora_loaded = True
        print(f'  已注入 LoRA adapter ← {lora_dir}')
    except ImportError:
        print('  [WARNING] peft 未安装，回退到原始 BGE')
        _lora_model = _base_model
        _lora_loaded = False


# ============================================================
# 向后兼容（供 eval_embedder / run.py 切换 / 旧代码）
# ============================================================
def get_embedder():
    """向后兼容：返回当前活动模型（由 LORA_ENABLED 决定）。新代码请使用 encode_with_model()."""
    import config
    if config.LORA_ENABLED:
        return get_lora_embedder()
    else:
        return get_base_embedder()


def get_base_embedder():
    """获取原始 BGE 模型（名称索引用）."""
    _load_base_model()
    return _base_model, _tokenizer, _device


def get_lora_embedder():
    """获取 LoRA BGE 模型（内容索引用，若 LORA_ENABLED=False 回退到原始 BGE）."""
    _load_lora_model()
    return _lora_model, _tokenizer, _device


# ============================================================
# Mean Pooling
# ============================================================
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


# ============================================================
# 编码接口
# ============================================================
def encode_texts(texts: list[str], normalize: bool = True,
                 batch_size: int = 32, use_lora: bool = False) -> np.ndarray:
    """
    批量编码文本为嵌入向量。

    Args:
        texts: 文本列表
        normalize: 是否 L2 归一化（默认 True，用于 FAISS 内积搜索）
        batch_size: 批次大小
        use_lora: True=使用 LoRA 模型（内容索引），False=使用原始 BGE（名称索引）

    Returns:
        (N, D) numpy 数组
    """
    import torch

    model, _, device = (get_lora_embedder() if use_lora else get_base_embedder())
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = _tokenizer(
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


def encode_query(query: str, normalize: bool = True,
                 use_lora: bool = False) -> np.ndarray:
    """编码单条查询文本。"""
    return encode_texts([query], normalize=normalize, use_lora=use_lora)


# ============================================================
# 运行时切换（向后兼容，供 eval_embedder 等使用）
# ============================================================
def reload_embedder():
    """
    强制重新加载模型（运行时切换 LoRA 开关时调用）。

    注意：现在同时管理双模型，此函数重置两者。
    """
    global _base_model, _lora_model, _tokenizer, _device, _hidden_size, _lora_loaded
    _base_model = None
    _lora_model = None
    _tokenizer = None
    _device = None
    _hidden_size = None
    _lora_loaded = False
    return get_embedder()
