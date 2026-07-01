from __future__ import annotations

import re

import pandas as pd


def _escape(value: object) -> str:
    return str(value or "").replace("\\", "\\\\").replace("'", "\\'")


def _safe_label(value: object, fallback: str = "Entity") -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "", str(value or ""))
    return text if text and not text[0].isdigit() else fallback


def _safe_relation(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(value or "associated_with")).upper()
    return text or "ASSOCIATED_WITH"


def generate_constraints(entities_df: pd.DataFrame) -> str:
    """生成 Neo4j 约束和索引语句。"""
    labels = sorted(set(entities_df.get("entity_type", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    lines = [
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE;",
        "CREATE INDEX entity_type_index IF NOT EXISTS FOR (n:Entity) ON (n.type);",
    ]
    for label in labels:
        safe = _safe_label(label)
        lines.append(f"CREATE INDEX {safe.lower()}_name_index IF NOT EXISTS FOR (n:{safe}) ON (n.name);")
    return "\n".join(lines)


def generate_node_cypher(entities_df: pd.DataFrame) -> str:
    """生成节点创建 Cypher。"""
    if entities_df is None or entities_df.empty:
        return "// 暂无实体节点"
    lines = []
    for _, row in entities_df.drop_duplicates(subset=["entity_name", "entity_type"]).iterrows():
        label = _safe_label(row.get("entity_type"))
        name = _escape(row.get("entity_name"))
        lines.append(
            "MERGE (n:Entity:{label} {{name: '{name}'}}) "
            "SET n.type = '{etype}', n.type_cn = '{etype_cn}', n.subtype = '{subtype}', "
            "n.paper_id = '{paper_id}', n.paragraph_id = '{paragraph_id}', "
            "n.source_text = '{source_text}', n.confidence = {confidence};".format(
                label=label,
                name=name,
                etype=_escape(row.get("entity_type")),
                etype_cn=_escape(row.get("entity_type_cn")),
                subtype=_escape(row.get("entity_subtype")),
                paper_id=_escape(row.get("paper_id")),
                paragraph_id=_escape(row.get("paragraph_id")),
                source_text=_escape(row.get("source_text")),
                confidence=float(row.get("confidence", 0) or 0),
            )
        )
    return "\n".join(lines)


def generate_relation_cypher(relations_df: pd.DataFrame) -> str:
    """生成关系创建 Cypher。"""
    if relations_df is None or relations_df.empty:
        return "// 暂无关系"
    lines = []
    for _, row in relations_df.drop_duplicates(subset=["subject", "relation", "object"]).iterrows():
        relation = _safe_relation(row.get("relation"))
        lines.append(
            "MATCH (s:Entity {{name: '{subject}'}}), (o:Entity {{name: '{object}'}}) "
            "MERGE (s)-[r:{relation}]->(o) "
            "SET r.relation_cn = '{relation_cn}', r.evidence = '{evidence}', "
            "r.paper_id = '{paper_id}', r.paragraph_id = '{paragraph_id}', r.confidence = {confidence};".format(
                subject=_escape(row.get("subject")),
                object=_escape(row.get("object")),
                relation=relation,
                relation_cn=_escape(row.get("relation_cn")),
                evidence=_escape(row.get("evidence")),
                paper_id=_escape(row.get("paper_id")),
                paragraph_id=_escape(row.get("paragraph_id")),
                confidence=float(row.get("confidence", 0) or 0),
            )
        )
    return "\n".join(lines)


def generate_query_cypher(core_node: str) -> str:
    """生成 1 跳和 1-3 跳查询 Cypher。"""
    core = _escape(core_node)
    return f"""// 1 跳邻域
MATCH (n:Entity {{name: '{core}'}})-[r]-(m:Entity)
RETURN n, r, m
LIMIT 100;

// 1-3 跳邻域
MATCH path = (n:Entity {{name: '{core}'}})-[*1..3]-(m:Entity)
RETURN path
LIMIT 200;

// 按关系类型统计邻域
MATCH (n:Entity {{name: '{core}'}})-[r*1..3]-(m:Entity)
UNWIND r AS rel
RETURN type(rel) AS relation_type, count(*) AS relation_count
ORDER BY relation_count DESC;"""


def generate_all_cypher(entities_df: pd.DataFrame, relations_df: pd.DataFrame, core_node: str) -> dict[str, str]:
    """生成全部 Cypher 文件内容。"""
    return {
        "constraints.cypher": generate_constraints(entities_df),
        "nodes.cypher": generate_node_cypher(entities_df),
        "relations.cypher": generate_relation_cypher(relations_df),
        "query.cypher": generate_query_cypher(core_node),
        "import.cypher": "\n\n".join(
            [
                generate_constraints(entities_df),
                generate_node_cypher(entities_df),
                generate_relation_cypher(relations_df),
            ]
        ),
    }

