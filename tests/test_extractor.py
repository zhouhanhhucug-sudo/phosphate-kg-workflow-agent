from src.corpus_processor import build_corpus_table
from src.extractor import run_extraction


def test_run_extraction_on_example_text():
    text = (
        "羊场磷矿床位于滇东北地区，矿体主要赋存于寒武系梅树村组含磷岩系中。"
        "含矿岩性主要为磷块岩、白云质磷块岩和含磷白云岩。"
        "P2O5 含量和矿层厚度是评价磷矿资源潜力的重要指标。"
    )
    corpus = build_corpus_table("P001", "示例", "1", "正文", text)
    result = run_extraction(corpus)

    entities = result["entities"]
    relations = result["relations"]

    assert "羊场磷矿床" in entities["entity_name"].tolist()
    assert "磷块岩" in entities["entity_name"].tolist()
    assert "located_in" in relations["relation"].tolist()
    assert "has_lithology" in relations["relation"].tolist()

