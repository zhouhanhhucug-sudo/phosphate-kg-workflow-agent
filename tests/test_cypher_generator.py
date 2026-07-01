import pandas as pd

from src.cypher_generator import generate_all_cypher


def test_generate_all_cypher_contains_nodes_and_relations():
    entities = pd.DataFrame(
        [
            {
                "entity_name": "羊场磷矿床",
                "entity_type": "Deposit",
                "entity_type_cn": "矿床",
                "entity_subtype": "",
                "paper_id": "P001",
                "paragraph_id": "P001-001",
                "source_text": "羊场磷矿床位于滇东北地区。",
                "confidence": 0.9,
            },
            {
                "entity_name": "滇东北地区",
                "entity_type": "Region",
                "entity_type_cn": "地区",
                "entity_subtype": "",
                "paper_id": "P001",
                "paragraph_id": "P001-001",
                "source_text": "羊场磷矿床位于滇东北地区。",
                "confidence": 0.8,
            },
        ]
    )
    relations = pd.DataFrame(
        [
            {
                "subject": "羊场磷矿床",
                "relation": "located_in",
                "relation_cn": "位于",
                "object": "滇东北地区",
                "evidence": "羊场磷矿床位于滇东北地区。",
                "paper_id": "P001",
                "paragraph_id": "P001-001",
                "confidence": 0.8,
            }
        ]
    )
    cypher = generate_all_cypher(entities, relations, "羊场磷矿床")

    assert "MERGE (n:Entity:Deposit" in cypher["nodes.cypher"]
    assert "LOCATED_IN" in cypher["relations.cypher"]
    assert "1-3 跳邻域" in cypher["query.cypher"]

