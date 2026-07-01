from src.corpus_processor import build_corpus_table, clean_text, split_paragraphs, tag_geological_elements


def test_build_corpus_table_from_example_text():
    text = "羊场磷矿床位于滇东北地区。含矿岩性主要为磷块岩和白云质磷块岩。"
    df = build_corpus_table("P001", "示例", "1", "正文", text)

    assert len(df) == 1
    assert df.loc[0, "keep_or_drop"] == "keep"
    assert "矿床" in df.loc[0, "element_tag"]
    assert "岩性" in df.loc[0, "element_tag"]


def test_clean_text_removes_reference_marks():
    assert clean_text("矿体赋存于含磷岩系[1-2]。") == "矿体赋存于含磷岩系。"


def test_split_paragraphs_uses_blank_lines():
    assert split_paragraphs("第一段。\n\n第二段。") == ["第一段。", "第二段。"]


def test_tag_geological_elements_detects_multiple_tags():
    tags = tag_geological_elements("P2O5 含量和矿层厚度是评价资源潜力的重要指标。")
    assert "地球化学" in tags
    assert "资源属性" in tags

