from __future__ import annotations


def build_hop_query(core_node: str, max_hop: int = 3) -> str:
    """生成 1-N 跳 Neo4j 查询语句。"""
    max_hop = max(1, min(int(max_hop), 5))
    escaped = str(core_node).replace("'", "\\'")
    return f"""MATCH path = (core:Entity {{name: '{escaped}'}})-[*1..{max_hop}]-(neighbor:Entity)
RETURN path, length(path) AS hop_count
ORDER BY hop_count ASC
LIMIT 200;"""


def build_hop_analysis_template(core_node: str, hop_range: str = "1-3") -> str:
    """生成 1-3 跳邻域分析说明。"""
    return f"""# {core_node} {hop_range} 跳邻域分析

## 分析目标
围绕核心节点“{core_node}”检索与其直接或间接相关的矿床、地层、岩性、构造、沉积环境、地球化学指标和找矿标志，用于解释成矿要素组合。

## 解读框架
1. 1 跳邻域：优先解释与核心节点直接相连的控矿因素、赋矿层位、岩性组合和资源属性。
2. 2 跳邻域：识别通过地层、构造或沉积环境连接起来的间接成矿要素。
3. 3 跳邻域：用于发现潜在的区域背景、成矿区带或找矿标志组合，结果需要人工复核。

## 输出建议
- 将 `controlled_by`、`hosted_by`、`has_lithology`、`has_geochemical_feature` 作为重点关系。
- 对置信度低或 `need_review=true` 的路径单独标记。
- 不把缺少证据句的路径作为确定性结论。
"""

