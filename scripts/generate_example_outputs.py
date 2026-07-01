from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.candidate_analyzer import analyze_candidate_area, build_candidate_markdown
from src.corpus_processor import build_corpus_table
from src.cypher_generator import generate_all_cypher
from src.exporter import export_project_zip, save_dataframe, save_text
from src.extractor import run_extraction
from src.gnn_data_builder import build_edges, build_model_plan, build_node_types, build_nodes, build_relation_types
from src.hop_analyzer import build_hop_analysis_template, build_hop_query
from src.ontology_checker import build_ontology_review


OUTPUT_DIR = ROOT / "data" / "output"


def main() -> None:
    text = (ROOT / "data" / "examples" / "example_text.txt").read_text(encoding="utf-8")
    corpus = build_corpus_table("P001", "羊场磷矿床示例", "1", "示例段落", text)
    extraction = run_extraction(corpus)
    entities = extraction["entities"]
    relations = extraction["relations"]
    attributes = extraction["attributes"]
    rules = extraction["rules"]
    review = build_ontology_review(entities, relations)
    core_node = str(entities.iloc[0]["entity_name"]) if not entities.empty else "羊场磷矿床"
    cypher_files = generate_all_cypher(entities, relations, core_node)
    hop_query = build_hop_query(core_node, 3)
    hop_markdown = build_hop_analysis_template(core_node, "1-3")
    nodes = build_nodes(entities)
    edges = build_edges(relations, nodes)
    node_types = build_node_types(entities)
    relation_types = build_relation_types(relations)
    model_plan = build_model_plan()

    candidate = analyze_candidate_area(
        "羊场外围候选区",
        {
            "赋矿层位": "寒武系梅树村组含磷岩系匹配，具备有利层位条件",
            "岩性组合": "发育磷块岩、白云质磷块岩和含磷白云岩",
            "构造条件": "背斜和断裂构造发育，对矿体展布有一定控制作用",
            "沉积环境": "浅海台地沉积环境有利于磷质沉积",
            "地球化学指标": "P2O5 含量为重要评价指标，具体数值需补充",
            "资源属性": "矿层厚度被作为资源潜力评价指标",
            "找矿标志": "含磷岩系、磷块岩和浅海台地环境可作为综合找矿线索",
        },
    )
    candidate_markdown = build_candidate_markdown(candidate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_dataframe(corpus, OUTPUT_DIR / "corpus.csv")
    save_dataframe(entities, OUTPUT_DIR / "entities.csv")
    save_dataframe(relations, OUTPUT_DIR / "relations.csv")
    save_dataframe(attributes, OUTPUT_DIR / "attributes.csv")
    save_dataframe(rules, OUTPUT_DIR / "rules.csv")
    save_dataframe(review, OUTPUT_DIR / "ontology_review.csv")
    save_dataframe(nodes, OUTPUT_DIR / "nodes.csv")
    save_dataframe(edges, OUTPUT_DIR / "edges.csv")
    save_dataframe(node_types, OUTPUT_DIR / "node_types.csv")
    save_dataframe(relation_types, OUTPUT_DIR / "relation_types.csv")

    for file_name, content in cypher_files.items():
        save_text(content, OUTPUT_DIR / file_name)
    save_text(hop_query, OUTPUT_DIR / "hop_query.cypher")
    save_text(hop_markdown, OUTPUT_DIR / "hop_analysis.md")
    save_text(candidate_markdown, OUTPUT_DIR / "candidate_analysis.md")
    save_text(model_plan, OUTPUT_DIR / "model_plan.md")
    export_project_zip(OUTPUT_DIR, OUTPUT_DIR / "phosphate_kg_outputs.zip")
    print(f"Example outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
