from __future__ import annotations

import re

import pandas as pd

from .utils import normalize_spaces


CORPUS_COLUMNS = [
    "paper_id",
    "title",
    "page",
    "section",
    "paragraph_id",
    "text_clean",
    "keep_or_drop",
    "element_tag",
    "priority",
    "note",
]


ELEMENT_KEYWORDS: dict[str, list[str]] = {
    "矿床": ["矿床", "磷矿", "矿区", "矿体", "矿段"],
    "矿体": ["矿体", "层状", "似层状", "产状", "厚度", "规模"],
    "地层": ["地层", "组", "段", "系", "统", "寒武", "震旦", "二叠"],
    "含矿层位": ["含矿层", "含磷岩系", "含矿岩系", "矿层", "层位"],
    "岩性": ["岩性", "磷块岩", "白云岩", "灰岩", "页岩", "泥岩", "砂岩"],
    "构造": ["构造", "断裂", "断层", "背斜", "褶皱", "构造带"],
    "沉积环境": ["沉积环境", "浅海", "台地", "潮坪", "陆棚", "盆地"],
    "地球化学": ["P2O5", "P₂O₅", "五氧化二磷", "稀土", "微量元素", "元素", "含量"],
    "资源属性": ["品位", "资源量", "储量", "厚度", "规模", "资源潜力"],
    "找矿标志": ["找矿", "控矿", "有利", "指示", "评价指标", "远景", "预测"],
    "成矿时代": ["成矿时代", "成矿期", "寒武纪", "震旦纪", "二叠纪"],
    "成矿区带": ["成矿带", "矿集区", "成矿区", "远景区"],
    "勘查工程": ["钻孔", "槽探", "剖面", "样品", "勘查", "工程"],
}


def split_paragraphs(raw_text: str) -> list[str]:
    """按自然段切分文本；若没有空行，则按较长句群保留为段落。"""
    raw = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []

    raw_paragraphs = [normalize_spaces(p) for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    if len(raw_paragraphs) > 1:
        return raw_paragraphs

    text = normalize_spaces(raw)
    if not text:
        return []

    single = text.replace("\n", "")
    if len(single) <= 550:
        return [single]

    sentences = re.split(r"(?<=[。！？；;])", single)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) > 450:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        chunks.append(current)
    return chunks


def clean_text(text: str) -> str:
    """清洗文本，去除多余空格、参考文献编号和明显无效字符。"""
    text = normalize_spaces(text)
    text = re.sub(r"\[[0-9,\-\s]+\]", "", text)
    text = re.sub(r"（\s*[0-9,\-\s]+\s*）", "", text)
    text = re.sub(r"\(\s*[0-9,\-\s]+\s*\)", "", text)
    text = re.sub(r"[�]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tag_geological_elements(text: str) -> list[str]:
    """根据关键词判断段落涉及的地质要素。"""
    tags = []
    for tag, keywords in ELEMENT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    return tags


def judge_keep_or_drop(text: str) -> str:
    """判断段落是否保留。"""
    if not text or len(text) < 12:
        return "drop"
    if re.search(r"^(参考文献|致谢|基金项目|作者简介|关键词[:：])", text):
        return "drop"
    return "keep" if tag_geological_elements(text) else "drop"


def judge_priority(tags: list[str], text: str) -> str:
    high_tags = {"矿床", "矿体", "含矿层位", "资源属性", "找矿标志"}
    mid_tags = {"地层", "岩性", "构造", "沉积环境", "地球化学"}
    if high_tags.intersection(tags) or any(word in text for word in ["控制", "有利", "重要指标", "资源潜力"]):
        return "高"
    if mid_tags.intersection(tags):
        return "中"
    return "低"


def build_corpus_table(
    paper_id: str,
    title: str,
    page: str,
    section: str,
    raw_text: str,
) -> pd.DataFrame:
    """生成标准语料表。"""
    rows = []
    for idx, paragraph in enumerate(split_paragraphs(raw_text), start=1):
        text_clean = clean_text(paragraph)
        if not text_clean:
            continue
        tags = tag_geological_elements(text_clean)
        keep_or_drop = judge_keep_or_drop(text_clean)
        rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "page": page,
                "section": section,
                "paragraph_id": f"{paper_id or 'PAPER'}-{idx:03d}",
                "text_clean": text_clean,
                "keep_or_drop": keep_or_drop,
                "element_tag": "；".join(tags) if tags else "未识别",
                "priority": judge_priority(tags, text_clean),
                "note": "" if tags else "未识别到地质关键词，建议人工确认",
            }
        )
    return pd.DataFrame(rows, columns=CORPUS_COLUMNS)
