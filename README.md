# 沉积型磷矿知识图谱工作流智能体

## 项目简介

本项目是一个面向沉积型磷矿找矿预测研究的知识图谱工作流 Web 应用原型，支持文献语料整理、知识抽取、本体校验、Neo4j Cypher 脚本生成、1-3 跳邻域分析、候选区成矿要素解释和 GNN 建模数据准备。

第一版默认使用离线规则抽取，不依赖 API Key，不强制连接 Neo4j，适合本地演示、阶段汇报和后续二次开发。

## 目录结构

```text
phosphate_kg_workflow_agent/
├── app.py
├── README.md
├── requirements.txt
├── config/
├── data/
│   ├── examples/
│   ├── input/
│   └── output/
├── src/
└── tests/
```

## 安装环境

```bash
cd phosphate_kg_workflow_agent
pip install -r requirements.txt
```

## 启动应用

```bash
streamlit run app.py
```

启动后在浏览器中打开 Streamlit 给出的本地地址。

## 使用流程

1. 打开“语料整理”，粘贴文本或上传 TXT / Markdown / CSV。
2. 点击“生成语料表”，得到标准语料表。
3. 打开“知识抽取”，点击“开始知识抽取”，生成实体、关系、属性和找矿规则。
4. 打开“本体校验”，检查实体类型和关系类型。
5. 打开“Neo4j 脚本生成”，生成约束、节点、关系和查询 Cypher。
6. 打开“1-3 跳分析”，生成邻域查询语句和解释模板。
7. 打开“候选区解释”，输入成矿要素证据并生成 Markdown 解释。
8. 打开“GNN 数据准备”，生成 nodes、edges、node_types 和 relation_types。
9. 打开“数据导出”，导出 CSV、JSON、Markdown、Cypher 和 ZIP。

## 输入 CSV 字段

上传 CSV 时，若包含以下字段，将直接作为语料表读取：

- `paper_id`
- `title`
- `page`
- `section`
- `paragraph_id`
- `text_clean`
- `keep_or_drop`
- `element_tag`
- `priority`
- `note`

若 CSV 中包含 `raw_text` 字段，系统会自动生成标准语料表。

## 输出文件

数据导出模块会将结果保存到 `data/output/`，常见文件包括：

- `corpus.csv`
- `entities.csv`
- `relations.csv`
- `attributes.csv`
- `rules.csv`
- `ontology_review.csv`
- `import.cypher`
- `query.cypher`
- `candidate_analysis.md`
- `nodes.csv`
- `edges.csv`
- `node_types.csv`
- `relation_types.csv`
- `model_plan.md`
- `phosphate_kg_outputs.zip`

## 测试

```bash
pytest
```

## 生成示例输出

```bash
python scripts/generate_example_outputs.py
```

## 注意事项

当前版本主要用于原型演示和数据准备，不保证自动抽取结果完全准确。所有 `need_review=true` 或校验结果为“需确认”的内容都需要人工复核。

项目不包含组会汇报、周报、开题报告、简历等写作功能。
