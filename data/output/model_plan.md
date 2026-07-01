# GNN 数据准备说明

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
