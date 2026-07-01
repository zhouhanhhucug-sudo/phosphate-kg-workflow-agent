import pandas as pd

from src.gnn_data_builder import build_edges, build_nodes, build_relation_types


def test_build_gnn_tables():
    entities = pd.DataFrame(
        [
            {
                "entity_name": "羊场磷矿床",
                "entity_type": "Deposit",
                "entity_type_cn": "矿床",
                "source_text": "证据句",
                "confidence": 0.9,
            },
            {
                "entity_name": "滇东北地区",
                "entity_type": "Region",
                "entity_type_cn": "地区",
                "source_text": "证据句",
                "confidence": 0.8,
            },
        ]
    )
    relations = pd.DataFrame(
        [
            {
                "subject": "羊场磷矿床",
                "object": "滇东北地区",
                "relation": "located_in",
                "relation_cn": "位于",
                "evidence": "证据句",
                "confidence": 0.8,
            }
        ]
    )

    nodes = build_nodes(entities)
    edges = build_edges(relations, nodes)
    relation_types = build_relation_types(relations)

    assert len(nodes) == 2
    assert edges.loc[0, "source"].startswith("N")
    assert relation_types.loc[0, "relation_type"] == "located_in"
