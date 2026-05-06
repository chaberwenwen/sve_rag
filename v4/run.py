"""
项目入口 —— 统一命令行

用法:
  python run.py index              # 构建 FAISS 向量索引
  python run.py cli                # 启动命令行交互模式
  python run.py cli "BP01-001"     # 单次查询（自动选择全部卡牌）
  python run.py cli --lora         # 使用 LoRA 微调模型
  python run.py ui                 # 启动 Gradio Web 界面
  python run.py ui --lora          # Web UI 使用 LoRA 微调模型
"""

import sys
import os

# 将 v4/ 上级目录（sve_rag/）加入 Python path，使 data/ 和 ui/ 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 将 v4/ 自身加入 path，使 v4 包内模块可裸导入（如 config, rag 等）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def toggle_lora(enable: bool = None) -> bool:
    """
    切换 LoRA 开关。可在运行时反复调用。

    Args:
        enable: True=启用, False=关闭, None=翻转当前状态

    Returns:
        bool: 切换后的状态
    """
    import config
    from v4.rag.embedder import reload_embedder

    if enable is None:
        enable = not config.LORA_ENABLED

    if enable == config.LORA_ENABLED:
        print(f'[LoRA] 已{"启用" if enable else "关闭"}，无需重复切换')
        return enable

    config.LORA_ENABLED = enable
    reload_embedder()
    print(f'[LoRA] 已{"启用" if enable else "关闭"}微调模型')
    return enable


def _startup_lora():
    """启动时处理 --lora 参数。"""
    if '--lora' in sys.argv:
        toggle_lora(True)
        # 清理 --lora 参数，避免干扰后续 argparse
        sys.argv = [a for a in sys.argv if a != '--lora']


def cmd_index():
    """构建索引（支持 --names-only / --content-only / --lora 等参数）"""
    from indexer.build import build_all_indices, build_names_only, build_content_only

    if '--names-only' in sys.argv:
        sys.argv.remove('--names-only')
        build_names_only()
    elif '--content-only' in sys.argv:
        sys.argv.remove('--content-only')
        build_content_only()
    else:
        build_all_indices()


def _select_cards_interactive(candidate_cards: list[dict]) -> list[str]:
    """
    交互式卡牌选择：展示候选卡牌列表，用户输入编号选择。

    Args:
        candidate_cards: identify_cards() 返回的候选卡牌列表

    Returns:
        list[str]: 用户选择的卡牌编号列表
    """
    if not candidate_cards:
        print('（未识别到卡牌，将仅用原始问题进行内容检索）')
        return []

    print(f'\n识别到 {len(candidate_cards)} 张候选卡牌:')
    for i, card in enumerate(candidate_cards, 1):
        desc_preview = card.get('description', '')
        if len(desc_preview) > 60:
            desc_preview = desc_preview[:60] + '...'
        print(f'  [{i}] {card["name"]} ({card["cardno"]})'
              f'  | {desc_preview}')

    print('\n请选择用于检索的卡牌:')
    print('  - 输入编号（如 1,2,3），多个用逗号或空格分隔')
    print('  - 输入 all 或直接回车：使用全部候选卡牌')
    print('  - 输入 none：不使用任何卡牌（仅内容检索）')

    while True:
        try:
            choice = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('（已取消）')
            return []

        if choice == '' or choice.lower() == 'all':
            return [c['cardno'] for c in candidate_cards]
        if choice.lower() == 'none':
            return []

        # 解析编号
        parts = choice.replace(',', ' ').split()
        selected = []
        invalid = []
        for p in parts:
            try:
                idx = int(p) - 1
                if 0 <= idx < len(candidate_cards):
                    selected.append(candidate_cards[idx]['cardno'])
                else:
                    invalid.append(p)
            except ValueError:
                invalid.append(p)

        if invalid:
            print(f'  无效输入: {", ".join(invalid)}，请重新输入')
            continue

        if selected:
            return selected
        else:
            print('  未选择任何卡牌，请重新输入')


def _print_result_summary(result: dict):
    """打印查询结果的摘要信息（不含 LLM 回答，回答已在主流程中打印）。"""
    if result.get('cards'):
        print(f'\n识别的卡牌 ({len(result["cards"])} 张):')
        for c in result['cards']:
            print(f'  [{c["cardno"]}] {c["name"]}')

    if result.get('qa_results'):
        print(f'\n相关 Q&A ({len(result["qa_results"])} 条):')
        for qa in result['qa_results'][:3]:
            print(f'  [{qa["cardno"]}] Q: {qa["qa"]["question_cn"][:80]}...')

    print(f'\n检索结果: {len(result["search_results"])} 条')


def cmd_cli():
    """命令行交互 / 单次查询"""
    from v4.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()

    if len(sys.argv) > 2:
        # 单次查询模式：自动选择全部卡牌（向后兼容）
        query_text = ' '.join(sys.argv[2:])
        result = pipeline.query(query_text)
        print('\n回答:')
        print(result['answer'])
    else:
        # 交互模式：先识别卡牌 → 用户选择 → 检索
        import config as _cfg
        print('\n' + '=' * 50)
        print('SVE RAG 查询助手 (输入 exit 退出)')
        print('=' * 50)
        print('输入问题，例如:')
        print('  - BP01-001 的效果是什么')
        print('  - 次元超越的费用变化和桃乐丝的减费结算顺序是什么样的')
        print()
        print('命令:')
        print('  /lora on  - 启用 LoRA 微调模型')
        print('  /lora off - 关闭 LoRA，恢复原始 BGE')
        print(f'  当前 LoRA: {"✓ 启用" if _cfg.LORA_ENABLED else "✗ 关闭"}')
        print()

        while True:
            try:
                user_input = input('\n问题> ').strip()
            except (EOFError, KeyboardInterrupt):
                print('\n退出')
                break

            if user_input.lower() in ('exit', 'quit', 'q'):
                break
            if not user_input:
                continue

            # 运行时 LoRA 开关命令
            if user_input.lower() == '/lora on':
                toggle_lora(True)
                continue
            if user_input.lower() == '/lora off':
                toggle_lora(False)
                continue
            if user_input.lower() == '/lora':
                # 翻转当前状态
                new_state = toggle_lora(None)
                print(f'  当前 LoRA: {"✓ 启用" if new_state else "✗ 关闭"}')
                continue

            # Step A: 识别卡牌
            candidate_cards = pipeline.identify_cards(user_input)

            # Step B: 用户手动选择卡牌
            selected_cardnos = _select_cards_interactive(candidate_cards)

            # Step C: 用选中的卡牌执行完整 RAG 查询
            result = pipeline.query(user_input, selected_cardnos=selected_cardnos)

            print('\n' + '-' * 40)
            print('回答:')
            print(result['answer'])
            _print_result_summary(result)


def cmd_ui():
    """启动 Web UI"""
    from ui.app import launch
    launch()


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    # 在加载任何模型之前处理 --lora 参数
    _startup_lora()

    cmd = sys.argv[1].lower()

    if cmd == 'index':
        cmd_index()
    elif cmd == 'cli':
        cmd_cli()
    elif cmd == 'ui':
        cmd_ui()
    else:
        print(f'未知命令: {cmd}')
        print_usage()


if __name__ == '__main__':
    main()
