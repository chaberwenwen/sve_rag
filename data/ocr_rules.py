"""
规则书图片 OCR 解析器
对 data/raw/QA_CN/规则书.jpg 进行 OCR 识别，提取中文规则文本并按章节分块。

OCR 方案（按优先级）:
  1. PaddleOCR — 中文识别精度最高（推荐）
  2. EasyOCR — 备选方案，安装更简单
  3. 纯手动模式 — 输出空模板供人工填写

依赖安装:
  pip install paddlepaddle paddleocr    # 方案1
  pip install easyocr                   # 方案2

使用方法:
    python data/ocr_rules.py                        # 自动检测 OCR 方案
    python data/ocr_rules.py --engine paddle        # 强制使用 PaddleOCR
    python data/ocr_rules.py --engine easyocr       # 强制使用 EasyOCR
    python data/ocr_rules.py --engine manual        # 生成空模板供人工填写

输入: data/raw/QA_CN/规则书.jpg
输出: data/rules/rules_chunks.zh.json
      格式: [{chapter, section, text, chunk_id}, ...]
"""
import json
import os
import re
import sys

# ============ 路径配置 ============
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # data/
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)               # 项目根目录
DATA_RULES_DIR = os.path.join(_PROJECT_DIR, 'data', 'rules')

# 规则书图片路径
RULES_IMAGE = os.path.join(_PROJECT_DIR, 'data', 'raw', 'QA_CN', '规则书.jpg')

# ============ 分块配置 ============
CHUNK_MIN_CHARS = 200
CHUNK_MAX_CHARS = 1500

# 中文章节标题检测正则
CHAPTER_CN_PATTERN = re.compile(r'^第[一二三四五六七八九十\d]+[章节篇部].*$')
SECTION_CN_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')


def detect_ocr_engine() -> str:
    """自动检测可用的 OCR 引擎"""
    try:
        import paddleocr
        print('  检测到 PaddleOCR，使用中文高精度 OCR')
        return 'paddle'
    except ImportError:
        pass

    try:
        import easyocr
        print('  检测到 EasyOCR，作为备选方案')
        return 'easyocr'
    except ImportError:
        pass

    print('  未检测到 OCR 库，请安装:')
    print('    pip install paddlepaddle paddleocr   (推荐)')
    print('    pip install easyocr                  (备选)')
    return 'none'


def ocr_paddle(image_path: str) -> list[dict]:
    """使用 PaddleOCR 识别图片中的文字"""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang='ch',       # 中文模型
        use_gpu=False,   # CPU 模式
    )
    result = ocr.ocr(image_path, textline_orientation=True)

    lines = []
    if result and result[0]:
        for line_info in result[0]:
            bbox = line_info[0]
            text = line_info[1][0]
            confidence = line_info[1][1]
            top_y = min(p[1] for p in bbox)
            lines.append({
                'text': text,
                'confidence': confidence,
                'y': top_y,
                'x': bbox[0][0],
            })

    lines.sort(key=lambda l: (round(l['y'], -1), l['x']))
    return lines


def ocr_easyocr(image_path: str) -> list[dict]:
    """使用 EasyOCR 识别图片中的文字"""
    import easyocr
    import numpy as np
    from PIL import Image

    reader = easyocr.Reader(
        ['ch_sim', 'en'],
        gpu=False,
    )

    pil_img = Image.open(image_path).convert('RGB')
    img_array = np.array(pil_img)
    result = reader.readtext(img_array)

    lines = []
    for bbox, text, confidence in result:
        top_y = min(p[1] for p in bbox)
        lines.append({
            'text': text,
            'confidence': confidence,
            'y': top_y,
            'x': bbox[0][0],
        })

    lines.sort(key=lambda l: (round(l['y'], -1), l['x']))
    return lines


def ocr_manual_template() -> list[dict]:
    """手动模式：生成空模板并提示用户手动填写规则文本"""
    print('\n[手动模式] 请手动将规则书文本填入 data/rules/manual_rules.txt')
    print('  格式要求:')
    print('    - 按章节组织，每行一个段落')
    print('    - 章节标题格式: "第1章 游戏概述" 或 "1.1 玩家数量"')
    print('    - 空行分隔不同段落')
    print('  完成后重新运行本脚本即可解析')
    return []


def postprocess_ocr_lines(lines: list[dict]) -> str:
    """
    OCR 后处理:
    1. 过滤低置信度行
    2. 合并同一段落内的行
    3. 返回原始文本供后续分块
    """
    if not lines:
        return ''

    filtered = [l for l in lines if l['confidence'] >= 0.5]
    if not filtered:
        filtered = lines

    # 保存原始 OCR 文本供调试
    debug_path = os.path.join(DATA_RULES_DIR, 'ocr_raw_output.txt')
    os.makedirs(DATA_RULES_DIR, exist_ok=True)
    with open(debug_path, 'w', encoding='utf-8') as f:
        for l in filtered:
            f.write(f'[y={l["y"]:.0f}, x={l["x"]:.0f}, conf={l["confidence"]:.3f}] {l["text"]}\n')
    print(f'  OCR 原始输出已保存: {debug_path}')

    # 合并成段落文本
    paragraphs = []
    current_para = []
    prev_y = None
    y_threshold = 15

    for line in filtered:
        y = round(line['y'], -1)
        if prev_y is not None and (y - prev_y) > y_threshold:
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
        current_para.append(line['text'])
        prev_y = y

    if current_para:
        paragraphs.append(' '.join(current_para))

    return '\n'.join(paragraphs)


def chunk_rules_text(full_text: str) -> list[dict]:
    """
    将规则文本按中文章节结构分块
    输出格式:
      [{chapter, section, text, chunk_id}, ...]
    """
    if not full_text.strip():
        return []

    lines = full_text.split('\n')
    chunks = []
    current_chapter = ''
    current_section = ''
    current_text = ''
    chunk_counter = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 跳过页码行
        if re.match(r'^\d+$', line) and len(line) < 5:
            continue

        # 检测中文章节标题： "第X章 XXXX"
        if CHAPTER_CN_PATTERN.match(line):
            if current_text.strip():
                chunks.append({
                    'chapter': current_chapter,
                    'section': current_section,
                    'text': current_text.strip(),
                    'chunk_id': f'rule-{chunk_counter}',
                })
                chunk_counter += 1
                current_text = ''

            ch_match = re.match(r'第([一二三四五六七八九十\d]+)[章节篇部]', line)
            if ch_match:
                cn_nums = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                           '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}
                ch_num = cn_nums.get(ch_match.group(1), ch_match.group(1))
                current_chapter = ch_num
            else:
                current_chapter = str(len(chunks) + 1)

            current_section = current_chapter
            current_text = line + '\n'
            continue

        # 检测小节标题： "1.1 XXXX" 或 "1.1.1 XXXX"
        section_match = SECTION_CN_PATTERN.match(line)
        if section_match:
            if current_text.strip():
                chunks.append({
                    'chapter': current_chapter,
                    'section': current_section,
                    'text': current_text.strip(),
                    'chunk_id': f'rule-{chunk_counter}',
                })
                chunk_counter += 1
                current_text = ''

            current_section = section_match.group(1)
            if not current_chapter:
                ch_part = current_section.split('.')[0]
                if ch_part.isdigit():
                    current_chapter = ch_part
            current_text = line + '\n'
            continue

        # 普通文本行
        if current_text:
            current_text += line + ' '
        else:
            current_text = line + ' '

    if current_text.strip():
        chunks.append({
            'chapter': current_chapter,
            'section': current_section,
            'text': current_text.strip(),
            'chunk_id': f'rule-{chunk_counter}',
        })

    refined_chunks = refine_chunks(chunks)
    for i, chunk in enumerate(refined_chunks):
        chunk['chunk_id'] = f'rule-{i}'

    return refined_chunks


def refine_chunks(chunks: list[dict]) -> list[dict]:
    """优化分块：合并过短块，切分过长块"""
    refined = []
    buffer = None

    for chunk in chunks:
        text_len = len(chunk['text'])

        if text_len < CHUNK_MIN_CHARS:
            if buffer is None:
                buffer = chunk
            else:
                buffer['text'] += '\n' + chunk['text']
        elif text_len > CHUNK_MAX_CHARS:
            if buffer:
                refined.append(buffer)
                buffer = None

            sentences = re.split(r'(?<=[。！？.!?])\s*', chunk['text'])
            temp_text = ''
            for sent in sentences:
                if not sent.strip():
                    continue
                if len(temp_text) + len(sent) < CHUNK_MAX_CHARS:
                    temp_text += sent + ' '
                else:
                    if temp_text:
                        refined.append({
                            'chapter': chunk['chapter'],
                            'section': chunk['section'],
                            'text': temp_text.strip(),
                            'chunk_id': '',
                        })
                    temp_text = sent + ' '
            if temp_text:
                refined.append({
                    'chapter': chunk['chapter'],
                    'section': chunk['section'],
                    'text': temp_text.strip(),
                    'chunk_id': '',
                })
        else:
            if buffer:
                combined = buffer['text'] + '\n' + chunk['text']
                if len(combined) <= CHUNK_MAX_CHARS:
                    buffer['text'] = combined
                else:
                    refined.append(buffer)
                    buffer = chunk
            else:
                buffer = chunk

    if buffer:
        refined.append(buffer)

    return refined


def parse_rules_image(image_path: str, engine: str = 'auto') -> list[dict]:
    """
    解析规则书图片的主流程
    engine: 'auto' | 'paddle' | 'easyocr' | 'manual'
    """
    if not os.path.exists(image_path):
        print(f'[错误] 规则书图片不存在: {image_path}')
        return []

    print(f'正在解析规则书图片: {image_path}')
    file_size = os.path.getsize(image_path)
    print(f'  文件大小: {file_size / 1024:.1f} KB')

    if engine == 'auto':
        engine = detect_ocr_engine()

    if engine == 'none':
        print('  无可用 OCR 引擎，生成空模板')
        return []

    print(f'  正在 OCR 识别 (引擎: {engine})...')
    if engine == 'paddle':
        lines = ocr_paddle(image_path)
    elif engine == 'easyocr':
        lines = ocr_easyocr(image_path)
    elif engine == 'manual':
        lines = ocr_manual_template()
        return []
    else:
        print(f'  未知 OCR 引擎: {engine}')
        return []

    if not lines:
        print('  OCR 未识别到任何文字')
        return []

    print(f'  OCR 识别到 {len(lines)} 个文本块')

    print('  正在后处理...')
    full_text = postprocess_ocr_lines(lines)

    print('  正在按章节分块...')
    chunks = chunk_rules_text(full_text)

    return chunks


def save_rules_chunks(chunks: list[dict]):
    """保存规则分块到 JSON（同时保存 .zh.json 和 .json 两份）"""
    if not chunks:
        print('  无规则块可保存')
        return None

    os.makedirs(DATA_RULES_DIR, exist_ok=True)

    # 保存中文规则（给 loader 使用）
    output_path = os.path.join(DATA_RULES_DIR, 'rules_chunks.zh.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f'  规则分块已保存: {output_path} ({len(chunks)} 块)')

    # 同步保存一份 rules_chunks.json（兼容旧配置）
    output_path2 = os.path.join(DATA_RULES_DIR, 'rules_chunks.json')
    with open(output_path2, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f'  规则分块已保存: {output_path2} ({len(chunks)} 块)')

    return output_path


def main():
    print('=' * 60)
    print('规则书图片 OCR 解析器')
    print('=' * 60)

    engine = 'auto'
    if '--engine' in sys.argv:
        idx = sys.argv.index('--engine')
        if idx + 1 < len(sys.argv):
            engine = sys.argv[idx + 1].lower()

    print(f'\nOCR 引擎: {engine}')
    print(f'规则书图片: {RULES_IMAGE}')

    chunks = parse_rules_image(RULES_IMAGE, engine)

    if chunks:
        save_rules_chunks(chunks)
        print('\n前 3 个块样例:')
        for c in chunks[:3]:
            text_preview = c['text'][:80].replace('\n', ' ')
            print(f'  [{c["chunk_id"]}] Ch.{c["chapter"]} Sec.{c["section"]}: {text_preview}...')
    else:
        print('\n未提取到规则内容')
        if engine == 'auto':
            print('提示: 尝试安装 OCR 库后重试')
            print('  pip install paddlepaddle paddleocr')
            print('  pip install easyocr')

    print(f'\n{"=" * 60}')
    print('完成!')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
