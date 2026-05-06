"""
Gradio Web 界面 —— 多 Tab 展示 RAG 各环节

面试要点：
  - 与 v2 废案2 的 UI 设计一致，但导入路径更新为共享模块模式
  - 多 Tab 展示 RAG 中间结果，方便调试和面试展示
  - 两步交互：先识别卡牌展示候选 → 用户勾选 → 再检索
  - 支持手动卡牌选择，避免噪音卡效应污染
  - 支持运行时 LoRA 开关切换（checkbox，可反复开关）
"""

import gradio as gr
import sys
import os

# 将 ui/ 上级目录（sve_rag/）加入 Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 全局 Pipeline 实例（延迟初始化）
# ============================================================
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from v4.rag.pipeline import RAGPipeline
        _pipeline = RAGPipeline()
    return _pipeline


# ============================================================
# Step 1: 识别卡牌
# ============================================================
def identify_handler(user_input: str):
    """仅识别卡牌，返回候选列表和 Checkbox 更新。"""
    if not user_input or not user_input.strip():
        return [], gr.update(choices=[], value=[])

    pipeline = _get_pipeline()
    candidates = pipeline.identify_cards(user_input)

    # 构建 checkbox 选项
    choices = []
    for c in candidates:
        cardno = c.get('cardno', '')
        name = c.get('name', '')
        desc = c.get('description', '')
        desc_short = desc[:40] + '...' if len(desc) > 40 else desc
        label = f'{name} ({cardno})'
        choices.append((label, cardno))

    # 默认全部选中
    default_values = [c['cardno'] for c in candidates]

    # 格式化候选卡牌信息展示
    cards_md = '### 识别到的候选卡牌\n\n'
    if candidates:
        for i, c in enumerate(candidates, 1):
            cardno = c.get('cardno', '')
            name = c.get('name', '')
            desc = c.get('description', '') or '（无效果描述）'
            card_type = c.get('card_type', '')
            card_class = c.get('class', '')
            cost = c.get('cost', '')
            cards_md += (
                f'**{i}. {name}** ({cardno})\n'
                f'> 职业: {card_class} | 类型: {card_type} | 费用: {cost}\n'
                f'> {desc[:200]}\n\n'
            )
    else:
        cards_md += '（未识别到卡牌，将仅用原始问题进行内容检索）\n'

    return (
        cards_md,
        gr.update(choices=choices, value=default_values,
                  interactive=True, visible=bool(choices)),
    )


# ============================================================
# Step 2: 查询（带卡牌选择）
# ============================================================
def query_handler(user_input: str, selected_cardnos: list):
    """处理用户查询，支持手动卡牌选择。"""
    if not user_input or not user_input.strip():
        return '请输入卡牌名称或编号', [], [], [], '', ''

    pipeline = _get_pipeline()
    result = pipeline.query(user_input, selected_cardnos=selected_cardnos)

    # 格式化卡牌信息
    cards_info = []
    for c in result['cards']:
        cardno = c.get('cardno', '')
        name = c.get('name', '')
        desc = c.get('description', '')
        cards_info.append(
            f'**{name}** ({cardno})\n> {desc[:200]}...'
            if len(desc) > 200
            else f'**{name}** ({cardno})\n> {desc}'
        )

    # 格式化 QA
    qa_info = []
    for qa in result['qa_results']:
        q = qa['qa'].get('question_cn', '')
        a = qa['qa'].get('answer_cn', '')
        qa_info.append(
            f'**[{qa["cardno"]}] {qa["name"]}**\nQ: {q}\nA: {a[:200]}...'
            if len(a) > 200
            else f'**[{qa["cardno"]}] {qa["name"]}**\nQ: {q}\nA: {a}'
        )

    # 格式化检索结果
    search_info = []
    for i, r in enumerate(result['search_results'], 1):
        meta = r['metadata']
        source = meta.get('source', '')
        score = r.get('score')
        text = r.get('text', '')[:300]
        cardno = meta.get('cardno', '')
        target_mark = ''
        if selected_cardnos and cardno in selected_cardnos:
            target_mark = ' ★目标卡'
        # 防御：score 可能为 None（来自 _merge_qa_results 追加的 QA 条目）
        if score is None:
            score_str = '关联QA'
        else:
            try:
                score_str = f'{score:.4f}'
            except (TypeError, ValueError) as e:
                print(f'[DEBUG] score format error: score={score!r}, type={type(score).__name__}, error={e}')
                score_str = f'{score}'
        search_info.append(
            f'**#{i}** 来源: {source} | cardno: {cardno} | 分数: {score_str}{target_mark}\n{text}...'
        )

    answer = result['answer']
    context = result.get('context', '')
    prompt = result.get('prompt', '')

    return (
        answer,
        cards_info if cards_info else ['未识别到卡牌'],
        qa_info if qa_info else ['无关联 Q&A'],
        search_info if search_info else ['无检索结果'],
        context,
        prompt,
    )


# ============================================================
# LoRA 切换（可在查询过程中反复开关）
# ============================================================
def toggle_lora_ui(enable: bool) -> str:
    """
    UI 中切换 LoRA 开关。
    调用 run.py 中的 toggle_lora() 统一逻辑，
    确保和 CLI 切换行为一致。
    """
    from v4.run import toggle_lora as _toggle
    _toggle(enable)
    return f'当前: **{"LoRA 微调模型 ✓" if enable else "原始 BGE 模型"}**'


# ============================================================
# UI 构建
# ============================================================
def create_ui():
    """创建 Gradio 界面 —— 两步交互：识别卡牌 → 选择卡牌 → 查询"""
    custom_css = """
    .scrollable-output .markdown-body,
    .scrollable-output .json-container,
    .scrollable-output textarea {
        overflow-y: auto !important;
    }
    .scrollable-md {
        height: 400px !important;
        overflow-y: auto !important;
    }
    .scrollable-md .markdown-body {
        max-height: 380px !important;
        overflow-y: auto !important;
    }
    .scrollable-json {
        height: 400px !important;
        overflow-y: auto !important;
    }
    .scrollable-json .json-container {
        max-height: 380px !important;
        overflow-y: auto !important;
    }
    .scrollable-text textarea {
        overflow-y: auto !important;
    }
    #candidate_cards_md {
        min-height: 100px;
    }
    """
    with gr.Blocks(title='SVE RAG 查询助手', css=custom_css) as ui:
        gr.Markdown('# 🃏 SVE 卡牌规则查询助手 (RAG v4)')
        gr.Markdown(
            '输入卡牌名称、编号或效果描述，AI 将结合数据库和规则书为你解答。\n\n'
            '**使用流程**：输入问题 → 点击「识别卡牌」→ 勾选需要的卡牌 → 点击「查询」'
        )

        # LoRA 开关（页面顶部，可随时切换，即时生效）
        with gr.Row():
            with gr.Column(scale=1):
                lora_toggle = gr.Checkbox(
                    label='启用 LoRA 微调模型',
                    value=False,
                    info='切换后即时生效，可在查询过程中反复开关',
                )
            with gr.Column(scale=3):
                lora_status = gr.Markdown('当前: **原始 BGE 模型**')

        # 输入区
        with gr.Row():
            with gr.Column(scale=3):
                user_input = gr.Textbox(
                    label='输入问题',
                    placeholder='例如: 次元超越的费用变化和桃乐丝的减费结算顺序是什么样的',
                    lines=2,
                )
            with gr.Column(scale=1):
                identify_btn = gr.Button('🔍 识别卡牌', variant='secondary', scale=1)

        # 卡牌选择区（初始隐藏，识别后显示）
        with gr.Group(visible=True) as card_select_group:
            gr.Markdown('### 请勾选用于检索的卡牌（默认全选）')
            card_checkbox = gr.CheckboxGroup(
                label='候选卡牌',
                choices=[],
                value=[],
                interactive=True,
                visible=False,
            )
            candidate_cards_md = gr.Markdown(
                '（输入问题后点击「识别卡牌」）',
                elem_id='candidate_cards_md',
            )

        # 查询按钮
        with gr.Row():
            query_btn = gr.Button('🚀 查询', variant='primary', scale=1)

        # 结果区
        with gr.Tabs():
            with gr.TabItem('💬 AI 回答'):
                answer_box = gr.Markdown(
                    label='AI 回答',
                    elem_classes=['scrollable-md'],
                )

            with gr.TabItem('🃏 识别的卡牌'):
                cards_box = gr.JSON(
                    label='卡牌列表',
                    elem_classes=['scrollable-json'],
                )

            with gr.TabItem('❓ 关联 Q&A'):
                qa_box = gr.JSON(
                    label='Q&A 列表',
                    elem_classes=['scrollable-json'],
                )

            with gr.TabItem('🔍 向量检索结果'):
                search_box = gr.JSON(
                    label='检索结果',
                    elem_classes=['scrollable-json'],
                )

            with gr.TabItem('📝 LLM Prompt'):
                with gr.Tabs():
                    with gr.TabItem('上下文'):
                        context_box = gr.Textbox(
                            label='构建的上下文',
                            lines=15,
                            max_lines=30,
                            elem_classes=['scrollable-text'],
                        )
                    with gr.TabItem('完整 Prompt'):
                        prompt_box = gr.Textbox(
                            label='完整 Prompt',
                            lines=20,
                            max_lines=40,
                            elem_classes=['scrollable-text'],
                        )

        # ============================================================
        # 事件绑定
        # ============================================================

        # LoRA 开关 → 实时切换模型
        lora_toggle.change(
            fn=toggle_lora_ui,
            inputs=[lora_toggle],
            outputs=[lora_status],
        )

        # Step 1: 识别卡牌 → 展示候选 + 更新 checkbox
        identify_btn.click(
            fn=identify_handler,
            inputs=[user_input],
            outputs=[candidate_cards_md, card_checkbox],
        )
        # 回车也触发识别
        user_input.submit(
            fn=identify_handler,
            inputs=[user_input],
            outputs=[candidate_cards_md, card_checkbox],
        )

        # Step 2: 查询 → 用选中的 cardnos 执行完整 RAG
        query_btn.click(
            fn=query_handler,
            inputs=[user_input, card_checkbox],
            outputs=[answer_box, cards_box, qa_box, search_box, context_box, prompt_box],
        )

    return ui


# ============================================================
# 启动入口
# ============================================================
def launch():
    """启动 Gradio Web UI"""
    ui = create_ui()
    print('启动 SVE RAG Web 界面...')
    print('本地访问: http://localhost:7860')
    ui.launch(server_name='127.0.0.1', server_port=7860, share=False)


if __name__ == '__main__':
    launch()
