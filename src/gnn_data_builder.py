from __future__ import annotations

import pandas as pd


def build_nodes(entities_df: pd.DataFrame) -> pd.DataFrame:
    """将实体表转换为 GNN 节点表。"""
    if entities_df is None or entities_df.empty:
        return pd.DataFrame(columns=["node_id", "name", "node_type", "node_type_cn", "source", "confidence"])
    nodes = entities_df.drop_duplicates(subset=["entity_name", "entity_type"]).copy()
    nodes["node_id"] = [f"N{idx:05d}" for idx in range(1, len(nodes) + 1)]
    return nodes.rename(
        columns={
            "entity_name": "name",
            "entity_type": "node_type",
            "entity_type_cn": "node_type_cn",
            "source_text": "source",
        }
    )[["node_id", "name", "node_type", "node_type_cn", "source", "confidence"]]


def build_edges(relations_df: pd.DataFrame, nodes_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """将关系表转换为 GNN 边表。"""
    if relations_df is None or relations_df.empty:
        return pd.DataFrame(columns=["source", "target", "source_name", "target_name", "relation_type", "relation_type_cn", "evidence", "confidence"])

    name_to_id = {}
    if nodes_df is not None and not nodes_df.empty:
        name_to_id = dict(zip(nodes_df["name"], nodes_df["node_id"]))

    edges = relations_df.copy()
    edges["source"] = edges["subject"].map(name_to_id).fillna(edges["subject"])
    edges["target"] = edges["object"].map(name_to_id).fillna(edges["object"])
    return edges.rename(
        columns={
            "subject": "source_name",
            "object": "target_name",
            "relation": "relation_type",
            "relation_cn": "relation_type_cn",
        }
    )[["source", "target", "source_name", "target_name", "relation_type", "relation_type_cn", "evidence", "confidence"]]


def build_relation_types(relations_df: pd.DataFrame) -> pd.DataFrame:
    """生成关系类型表。"""
    if relations_df is None or relations_df.empty:
        return pd.DataFrame(columns=["relation_type", "relation_type_cn", "count"])
    return (
        relations_df.groupby(["relation", "relation_cn"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"relation": "relation_type", "relation_cn": "relation_type_cn"})
    )


def build_node_types(entities_df: pd.DataFrame) -> pd.DataFrame:
    """生成节点类型表。"""
    if entities_df is None or entities_df.empty:
        return pd.DataFrame(columns=["node_type", "node_type_cn", "count"])
    return (
        entities_df.groupby(["entity_type", "entity_type_cn"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"entity_type": "node_type", "entity_type_cn": "node_type_cn"})
    )


def build_model_plan() -> str:
    """生成 GNN 建模数据说明。"""
    return """# GNN 数据准备说明

## 当前产物
- `nodes.csv`：知识图谱节点表，可作为 R-GCN / CompGCN 的实体输入。
- `edges.csv`：多关系边表，包含 source、target、relation_type 和 evidence。
- `node_types.csv`：节点类型枚举与数量统计。
- `relation_types.csv`：关系类型枚举与数量统计。

## 建模建议
1. 先对 `node_id` 和 `relation_type` 进行整数编码。
2. 将 `confidence` 和 `need_review` 作为样本权重或过滤条件。
3. 仅使用具备证据句且通过本体校验的关系训练初版模型。
4. 若用于矿产预测，应补充空间单元、正负样本标签和区域地质变量。

## 注意事项
当前应用只生成建模数据，不训练 R-GCN / CompGCN。训练前需要人工复核实体消歧、关系方向和候选区标签。
"""

