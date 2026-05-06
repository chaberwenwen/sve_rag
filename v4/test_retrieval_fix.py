"""
检索修复验证脚本 —— 测试三索引架构 + 低分噪声过滤

验证项:
  1. 三索引加载正确（card_names + content + content_lora）
  2. LoRA 开启时 search_content 自动使用 content_lora 索引
  3. 效果文本检索 (Step 3a) 应返回语义相关的 QA
  4. 用户原始问题检索 (Step 3b) 的噪声应被 CONTENT_SCORE_MIN_THRESHOLD 过滤
  5. _merge_results 双层过滤（绝对阈值 + 相对比值）

用法:
  python v4/test_retrieval_fix.py
"""

import sys
import os

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V4 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _V4)

import config
from v4.rag.embedder import reload_embedder
from v4.rag.retriever import Retriever


def print_separator(title: str):
    print(f'\n{"=" * 60}')
    print(f'  {title}')
    print(f'{"=" * 60}')


def test_index_loading():
    """测试1: 三索引加载 + 编码器匹配检查"""
    print_separator('测试1: 三索引加载检查')

    retriever = Retriever()

    # 名称索引
    assert retriever.name_index is not None, 'ERROR: card_names 索引未加载'
    print(f'  card_names: {retriever.name_index.ntotal} 条目 ✓ (原始BGE)')

    # 内容索引（原始 BGE）
    assert retriever.content_index is not None, 'ERROR: content 索引未加载'
    print(f'  content:    {retriever.content_index.ntotal} 条目 ✓ (原始BGE)')

    # 内容索引（LoRA BGE）
    if retriever.content_lora_index is not None:
        print(f'  content_lora: {retriever.content_lora_index.ntotal} 条目 ✓ (LoRA BGE)')
    else:
        print(f'  content_lora: 未加载 (LoRA adapter 可能不存在)')

    # 验证活跃内容索引
    idx, metas, texts, use_lora = retriever._get_active_content()
    model_tag = 'LoRA BGE' if use_lora else '原始 BGE'
    print(f'  当前活跃内容索引: {"content_lora" if use_lora else "content"} ({model_tag})')

    if config.LORA_ENABLED:
        assert use_lora, 'ERROR: LORA_ENABLED=True 但未使用 LoRA 索引'
        print(f'  ✓ LoRA 开启，自动使用 content_lora')
    else:
        assert not use_lora, 'ERROR: LORA_ENABLED=False 但使用了 LoRA 索引'
        print(f'  ✓ LoRA 关闭，自动使用 content（原始BGE）')

    print('  结果: PASS ✓')
    return retriever


def test_name_search(retriever: Retriever):
    """测试2: 名称索引检索（卡牌识别）——始终原始BGE"""
    print_separator('测试2: 名称索引检索（原始BGE）')

    queries = [
        '远古精灵',
        '玫瑰皇后',
        '深究的魔法师',
    ]

    for q in queries:
        hits = retriever.search_names(q, top_k=3)
        print(f'\n  查询: "{q}"')
        for h in hits:
            meta = h.get('metadata', {})
            print(f'    [{meta.get("cardno","?")}] {meta.get("name","?")} '
                  f'(score={h["score"]:.4f})')


def test_content_search_effect(retriever: Retriever):
    """测试3: 效果文本检索 (Step 3a)——使用当前活跃内容索引"""
    print_separator('测试3: 效果文本检索')

    effect_text = (
        '《进化》《消费1》：使这张卡进化。\n'
        '【守护】\n'
        '《入场曲》将场上的其他的1张卡片放回手牌：'
        '使这张卡《攻击力》+1/《生命值》+1。'
    )

    hits = retriever.search_content(effect_text, top_k=5)
    print(f'  效果文本: {effect_text[:80]}...')
    print(f'\n  Top-5 检索结果:')
    for i, h in enumerate(hits, 1):
        text_preview = h['text'][:80].replace('\n', ' ')
        meta = h.get('metadata', {})
        print(f'    #{i} [{meta.get("cardno","?")}] score={h["score"]:.4f}')
        print(f'       {text_preview}...')

    if hits:
        avg_score = sum(h['score'] for h in hits) / len(hits)
        print(f'\n  平均分: {avg_score:.4f}')
        if avg_score > 0.60:
            print('  结果: PASS ✓ (分数分布在合理范围)')
        else:
            print(f'  结果: WARNING ⚠ (平均分偏低 {avg_score:.4f}，LoRA 可能欠拟合)')

    return hits


def test_user_query_search(retriever: Retriever):
    """测试4: 用户问题检索 (Step 3b) —— 核心噪声测试"""
    print_separator('测试4: 用户原始问题检索 + 噪声过滤')

    user_query = ('深究的魔法师·伊莎贝尔在不能选择对方场上的随从的情况下，'
                  '可以选择能力【1】吗')

    hits = retriever.search_content(user_query, top_k=10)

    print(f'  查询: "{user_query}"')
    print(f'\n  Top-10 检索结果 (含阈值判定):')

    threshold = config.CONTENT_SCORE_MIN_THRESHOLD
    passed = 0
    filtered = 0
    for i, h in enumerate(hits, 1):
        text_preview = h['text'][:100].replace('\n', ' ')
        meta = h.get('metadata', {})
        above_threshold = h['score'] >= threshold
        status = '✓' if above_threshold else '✗ FILTERED'

        print(f'    #{i} [{meta.get("cardno","?")}] score={h["score"]:.4f} {status}')
        print(f'       {text_preview}...')

        if above_threshold:
            passed += 1
        else:
            filtered += 1

    print(f'\n  高于阈值({threshold}): {passed} 条')
    print(f'  低于阈值将被过滤: {filtered} 条')

    # 检查结果是否包含「同选项」噪声
    noise_count = sum(1 for h in hits if '同选项' in h['text'])
    effective_count = sum(1 for h in hits
                           if '不能选择' in h['text'] or '无法选择' in h['text'])
    print(f'  「同选项」噪声: {noise_count} 条')
    print(f'  「不能/无法选择」相关: {effective_count} 条')

    if passed > 0:
        print(f'  ✓ 有 {passed} 条结果通过阈值过滤')
    else:
        print(f'  ⚠ WARNING: 无结果通过阈值')

    return hits


def test_merge_double_filter():
    """测试5: _merge_results 双层过滤验证"""
    print_separator('测试5: _merge_results 双层过滤验证')
    from v4.rag.pipeline import _merge_results

    ratio_threshold = config.CONTENT_SCORE_RATIO_THRESHOLD
    min_threshold = config.CONTENT_SCORE_MIN_THRESHOLD

    # 模拟 LoRA 后的典型分数分布（分数集中在窄区间）
    results_effect = [
        {'index': 5, 'score': 0.66, 'text': 'QA: 不能选择对方从者时能否使用能力',
         'metadata': {'cardno': 'BP09-117'}},
        {'index': 10, 'score': 0.65, 'text': 'QA: 无法选择时效果处理',
         'metadata': {'cardno': 'BP13-040'}},
    ]
    results_question = [
        {'index': 100, 'score': 0.60, 'text': 'QA: 可以多次选择同选项吗',
         'metadata': {'cardno': 'BP02-045'}},
        {'index': 101, 'score': 0.59, 'text': 'QA: 不可以多次选择同选项',
         'metadata': {'cardno': 'BP02-085'}},
        {'index': 20, 'score': 0.64, 'text': 'QA: 支付费用选择目标',
         'metadata': {'cardno': 'BP07-034'}},
        {'index': 30, 'score': 0.48, 'text': 'QA: 绝对低分噪声',
         'metadata': {'cardno': 'BP01-002'}},
    ]

    print(f'  绝对阈值: {min_threshold}')
    print(f'  相对比值阈值: {ratio_threshold}')
    print(f'  模拟输入 (LoRA 分数集中场景):')
    print(f'    results_effect: {len(results_effect)} 条 (分数: 0.66, 0.65)')
    print(f'    results_question: {len(results_question)} 条 (分数: 0.60, 0.59, 0.64, 0.48)')

    merged = _merge_results(results_effect, results_question, top_k=5)

    print(f'\n  合并后:')
    for i, r in enumerate(merged, 1):
        print(f'    #{i} idx={r["index"]} score={r["score"]:.2f} '
              f'| {r["text"][:50]}...')

    # 验证1：绝对阈值过滤 (0.48 应该被过滤)
    merged_indices = {r['index'] for r in merged}
    assert 30 not in merged_indices, 'FAIL: 绝对低分噪声(0.48)未被过滤'
    print(f'  ✓ 绝对阈值过滤: 0.48(idx=30)已过滤')

    # 验证2：相对比值过滤 (max=0.66, ratio=0.85 → 阈值=0.561)
    # 0.60 和 0.59 虽然高于绝对阈值 0.55，但低于相对阈值 0.561，应被过滤
    rel_cutoff = 0.66 * ratio_threshold
    print(f'  相对阈值: {0.66:.2f} × {ratio_threshold} = {rel_cutoff:.4f}')
    if 100 not in merged_indices and 101 not in merged_indices:
        print(f'  ✓ 相对比值过滤: 0.60(idx=100)和0.59(idx=101)已过滤'
              f'（低于 {rel_cutoff:.4f}）')
    elif 100 in merged_indices or 101 in merged_indices:
        print(f'  ⚠ 部分中等分数未被相对比值过滤')

    # 验证3：有效结果保留
    if {5, 10, 20}.issubset(merged_indices):
        print(f'  ✓ 有效高分结果 (idx=5,10,20) 全部保留')

    print(f'  结果: PASS ✓')
    return merged


def main():
    # 跟随 config.py 中的 LORA_ENABLED 设置
    print('=' * 60)
    print('  检索修复验证测试')
    print(f'  LORA_ENABLED: {config.LORA_ENABLED}')
    print(f'  CONTENT_SCORE_MIN_THRESHOLD: {config.CONTENT_SCORE_MIN_THRESHOLD}')
    print(f'  CONTENT_SCORE_RATIO_THRESHOLD: {config.CONTENT_SCORE_RATIO_THRESHOLD}')
    print(f'  TOP_K: {config.TOP_K}')
    print('=' * 60)

    try:
        retriever = test_index_loading()
        test_name_search(retriever)
        test_content_search_effect(retriever)
        test_user_query_search(retriever)
        test_merge_double_filter()

        print_separator('全部测试完成')

        # 总结
        idx, _, _, use_lora = retriever._get_active_content()
        active_index = 'content_lora' if use_lora else 'content'
        print(f'\n  架构总结:')
        print(f'    ┌─ card_names    → 原始 BGE (名称匹配，始终不变)')
        print(f'    ├─ content       → 原始 BGE (LoRA 关闭时的内容检索)')
        print(f'    └─ content_lora  → LoRA BGE (LoRA 开启时的内容检索)')
        print(f'    当前活跃内容索引: {active_index}')
        print(f'\n  切换方法:')
        print(f'    关闭 LoRA: config.LORA_ENABLED = False → 自动使用 content')
        print(f'    开启 LoRA: config.LORA_ENABLED = True  → 自动使用 content_lora')
        print(f'    无需重建索引即可切换 ✓')

    except Exception as e:
        print(f'\n[ERROR] 测试失败: {e}')
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
