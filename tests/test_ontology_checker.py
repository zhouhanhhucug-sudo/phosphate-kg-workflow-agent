import pandas as pd

from src.ontology_checker import check_entity_types, check_relation_types


def test_check_entity_types_suggests_mapping():
    entities = pd.DataFrame(
        [
            {"entity_id": "E1", "entity_name": "断裂", "entity_type": "断裂", "need_review": False},
            {"entity_id": "E2", "entity_name": "羊场磷矿床", "entity_type": "Deposit", "need_review": False},
        ]
    )
    review = check_entity_types(entities, {"Deposit": "矿床", "TectonicSetting": "构造背景"}, {"断裂": "TectonicSetting"})

    invalid = review[review["item_id"] == "E1"].iloc[0]
    assert invalid["suggested_value"] == "TectonicSetting"
    assert invalid["need_review"]


def test_check_relation_types_validates_known_relation():
    relations = pd.DataFrame(
        [
            {
                "relation_id": "R1",
                "relation": "located_in",
                "direction_check": "通过",
                "need_review": False,
            }
        ]
    )
    review = check_relation_types(relations, {"located_in": "位于"}, {})

    assert bool(review.loc[0, "is_valid"])

