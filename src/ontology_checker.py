from __future__ import annotations

import pandas as pd


REVIEW_COLUMNS = [
    "review_id",
    "item_kind",
    "item_id",
    "field",
    "original_value",
    "is_valid",
    "suggested_value",
    "action",
    "message",
    "need_review",
]


def _suggest(value: str, mapping: dict[str, str], allowed: dict[str, str]) -> str:
    value = str(value or "").strip()
    if value in allowed:
        return value
    if value in mapping:
        return mapping[value]
    for key, target in mapping.items():
        if key and key in value:
            return target
    for allowed_key, cn in allowed.items():
        if value == cn or value in str(cn):
            return allowed_key
    return ""


def check_entity_types(entities_df: pd.DataFrame, entity_types: dict, entity_mapping: dict) -> pd.DataFrame:
    """检查实体类型是否符合本体。"""
    rows = []
    if entities_df is None or entities_df.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)

    for _, row in entities_df.iterrows():
        original = str(row.get("entity_type", ""))
        suggested = _suggest(original, entity_mapping, entity_types)
        valid = original in entity_types
        rows.append(
            {
                "review_id": f"OE-{len(rows) + 1:04d}",
                "item_kind": "entity",
                "item_id": row.get("entity_id", ""),
                "field": "entity_type",
                "original_value": original,
                "is_valid": valid,
                "suggested_value": original if valid else suggested,
                "action": "通过" if valid else ("建议映射" if suggested else "人工确认"),
                "message": "实体类型符合本体" if valid else "实体类型不在本体枚举中",
                "need_review": not valid or bool(row.get("need_review", False)),
            }
        )

    duplicates = entities_df[entities_df.duplicated(subset=["entity_name"], keep=False)]
    for name in sorted(duplicates["entity_name"].dropna().astype(str).unique()):
        rows.append(
            {
                "review_id": f"OE-{len(rows) + 1:04d}",
                "item_kind": "entity",
                "item_id": "多个",
                "field": "entity_name",
                "original_value": name,
                "is_valid": False,
                "suggested_value": name,
                "action": "人工确认",
                "message": "存在同名实体，建议确认是否为同义或重复实体",
                "need_review": True,
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def check_relation_types(relations_df: pd.DataFrame, relation_types: dict, relation_mapping: dict) -> pd.DataFrame:
    """检查关系类型是否符合本体。"""
    rows = []
    if relations_df is None or relations_df.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)

    for _, row in relations_df.iterrows():
        original = str(row.get("relation", ""))
        suggested = _suggest(original, relation_mapping, relation_types)
        valid = original in relation_types
        direction_ok = str(row.get("direction_check", "")) == "通过"
        rows.append(
            {
                "review_id": f"OR-{len(rows) + 1:04d}",
                "item_kind": "relation",
                "item_id": row.get("relation_id", ""),
                "field": "relation",
                "original_value": original,
                "is_valid": valid and direction_ok,
                "suggested_value": original if valid else suggested,
                "action": "通过" if valid and direction_ok else ("建议映射" if suggested else "人工确认"),
                "message": "关系类型符合本体" if valid else "关系类型不在本体枚举中",
                "need_review": (not valid) or (not direction_ok) or bool(row.get("need_review", False)),
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def apply_mapping(df: pd.DataFrame, review_df: pd.DataFrame) -> pd.DataFrame:
    """根据校验结果应用映射建议。"""
    if df is None or df.empty or review_df is None or review_df.empty:
        return df
    mapped = df.copy()
    for _, review in review_df.iterrows():
        if review.get("action") != "建议映射" or not review.get("suggested_value"):
            continue
        item_id = review.get("item_id")
        field = review.get("field")
        if field in mapped.columns:
            id_col = "entity_id" if str(item_id).startswith("E") else "relation_id"
            if id_col in mapped.columns:
                mapped.loc[mapped[id_col] == item_id, field] = review.get("suggested_value")
    return mapped


def build_ontology_review(entities_df: pd.DataFrame, relations_df: pd.DataFrame) -> pd.DataFrame:
    """生成本体校验表。"""
    from .utils import load_config

    entity_types = load_config("entity_types.json")
    relation_types = load_config("relation_types.json")
    entity_mapping = load_config("entity_mapping.json")
    relation_mapping = load_config("relation_mapping.json")
    entity_review = check_entity_types(entities_df, entity_types, entity_mapping)
    relation_review = check_relation_types(relations_df, relation_types, relation_mapping)
    return pd.concat([entity_review, relation_review], ignore_index=True)

