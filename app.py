from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.candidate_analyzer import FACTOR_WEIGHTS, analyze_candidate_area, build_candidate_markdown
from src.corpus_processor import CORPUS_COLUMNS, build_corpus_table
from src.cypher_generator import generate_all_cypher
from src.exporter import export_project_zip, save_dataframe, save_text
from src.extractor import ATTRIBUTE_COLUMNS, ENTITY_COLUMNS, RELATION_COLUMNS, RULE_COLUMNS, run_extraction
from src.gnn_data_builder import build_edges, build_model_plan, build_node_types, build_nodes, build_relation_types
from src.hop_analyzer import build_hop_analysis_template, build_hop_query
from src.ontology_checker import REVIEW_COLUMNS, apply_mapping, build_ontology_review
from src.utils import dataframe_to_csv_bytes, dataframe_to_json_bytes, ensure_output_dir


ROOT = Path(__file__).resolve().parent
EXAMPLE_TEXT_PATH = ROOT / "data" / "examples" / "example_text.txt"

PAGES = [
    "项目首页",
    "语料整理",
    "知识抽取",
    "本体校验",
    "Neo4j 脚本生成",
    "1-3 跳分析",
    "候选区解释",
    "GNN 数据准备",
    "数据导出",
    "使用说明",
]


def empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def init_state() -> None:
    defaults = {
        "corpus_df": empty_df(CORPUS_COLUMNS),
        "entities_df": empty_df(ENTITY_COLUMNS),
        "relations_df": empty_df(RELATION_COLUMNS),
        "attributes_df": empty_df(ATTRIBUTE_COLUMNS),
        "rules_df": empty_df(RULE_COLUMNS),
        "ontology_review_df": empty_df(REVIEW_COLUMNS),
        "cypher_files": {},
        "hop_query": "",
        "hop_markdown": "",
        "candidate_result": {},
        "candidate_markdown": "",
        "nodes_df": pd.DataFrame(),
        "edges_df": pd.DataFrame(),
        "node_types_df": pd.DataFrame(),
        "relation_types_df": pd.DataFrame(),
        "model_plan": "",
        "nav_page": "项目首页",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px 14px;
            background: #ffffff;
        }
        div[data-testid="stSidebar"] {
            background: #f8faf9;
        }
        .workflow-step {
            display: inline-block;
            border: 1px solid #d7ded9;
            border-radius: 8px;
            padding: 8px 10px;
            margin: 0 6px 8px 0;
            background: #ffffff;
            color: #1f2933;
            font-size: 0.92rem;
        }
        .status-line {
            border-left: 4px solid #2f855a;
            padding: 10px 12px;
            background: #f5fbf7;
            margin: 8px 0 16px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_example_text() -> str:
    return EXAMPLE_TEXT_PATH.read_text(encoding="utf-8")


def load_example_pipeline() -> None:
    corpus_df = build_corpus_table("P001", "羊场磷矿床示例", "1", "示例段落", load_example_text())
    extraction = run_extraction(corpus_df)
    st.session_state.corpus_df = corpus_df
    st.session_state.entities_df = extraction["entities"]
    st.session_state.relations_df = extraction["relations"]
    st.session_state.attributes_df = extraction["attributes"]
    st.session_state.rules_df = extraction["rules"]
    st.session_state.ontology_review_df = build_ontology_review(extraction["entities"], extraction["relations"])
    core = first_entity_name()
    st.session_state.cypher_files = generate_all_cypher(extraction["entities"], extraction["relations"], core)
    st.session_state.hop_query = build_hop_query(core, 3)
    st.session_state.hop_markdown = build_hop_analysis_template(core, "1-3")
    build_gnn_state()


def first_entity_name() -> str:
    entities = st.session_state.entities_df
    if not entities.empty and "entity_name" in entities:
        return str(entities.iloc[0]["entity_name"])
    return "羊场磷矿床"


def section_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.markdown(f"<div class='status-line'>{subtitle}</div>", unsafe_allow_html=True)


def show_download_buttons(df: pd.DataFrame, stem: str) -> None:
    left, right = st.columns(2)
    with left:
        st.download_button(
            "下载 CSV",
            data=dataframe_to_csv_bytes(df),
            file_name=f"{stem}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=df.empty,
        )
    with right:
        st.download_button(
            "下载 JSON",
            data=dataframe_to_json_bytes(df),
            file_name=f"{stem}.json",
            mime="application/json",
            use_container_width=True,
            disabled=df.empty,
        )


def read_uploaded_file(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        last_error = ""
        for encoding in ("utf-8-sig", "utf-8", "gbk"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
                return normalize_uploaded_dataframe(df), ""
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        return None, f"CSV 读取失败：{last_error}"

    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            text = raw.decode(encoding)
            return None, text
        except UnicodeDecodeError:
            continue
    return None, "文件编码无法识别"


def normalize_uploaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if set(CORPUS_COLUMNS).issubset(df.columns):
        return df[CORPUS_COLUMNS].copy()
    if "raw_text" in df.columns:
        frames = []
        for idx, row in df.iterrows():
            frames.append(
                build_corpus_table(
                    str(row.get("paper_id", f"P{idx + 1:03d}")),
                    str(row.get("title", "上传文献")),
                    str(row.get("page", "")),
                    str(row.get("section", "")),
                    str(row.get("raw_text", "")),
                )
            )
        return pd.concat(frames, ignore_index=True) if frames else empty_df(CORPUS_COLUMNS)

    text_col = df.columns[0]
    return build_corpus_table("UPLOAD", "上传 CSV", "", "", "\n\n".join(df[text_col].dropna().astype(str).tolist()))


def sidebar() -> str:
    st.sidebar.title("工作流")
    page = st.sidebar.radio("模块导航", PAGES, key="nav_page", label_visibility="collapsed")
    st.sidebar.divider()
    corpus_count = len(st.session_state.corpus_df)
    entity_count = len(st.session_state.entities_df)
    relation_count = len(st.session_state.relations_df)
    review_count = (
        int(st.session_state.ontology_review_df["need_review"].sum())
        if not st.session_state.ontology_review_df.empty and "need_review" in st.session_state.ontology_review_df
        else 0
    )
    st.sidebar.metric("语料段落", corpus_count)
    st.sidebar.metric("实体", entity_count)
    st.sidebar.metric("关系", relation_count)
    st.sidebar.metric("待复核", review_count)
    st.sidebar.divider()
    if st.sidebar.button("载入示例数据", use_container_width=True):
        load_example_pipeline()
        st.toast("示例数据已载入")
    return page


def page_home() -> None:
    section_header("沉积型磷矿知识图谱工作流智能体", "Phosphate KG Workflow Agent")
    metrics = st.columns(5)
    metrics[0].metric("文献数量", st.session_state.corpus_df["paper_id"].nunique() if not st.session_state.corpus_df.empty else 0)
    metrics[1].metric("语料段落", len(st.session_state.corpus_df))
    metrics[2].metric("实体数量", len(st.session_state.entities_df))
    metrics[3].metric("关系数量", len(st.session_state.relations_df))
    review_count = int(st.session_state.ontology_review_df["need_review"].sum()) if not st.session_state.ontology_review_df.empty else 0
    metrics[4].metric("需确认项", review_count)

    steps = ["文献输入", "语料整理", "知识抽取", "本体校验", "Cypher 生成", "1-3 跳分析", "候选区解释", "GNN 数据准备", "结果导出"]
    st.markdown("".join(f"<span class='workflow-step'>{step}</span>" for step in steps), unsafe_allow_html=True)

    left, mid, right = st.columns(3)
    with left:
        if st.button("新建项目", use_container_width=True):
            for key in [
                "corpus_df",
                "entities_df",
                "relations_df",
                "attributes_df",
                "rules_df",
                "ontology_review_df",
                "nodes_df",
                "edges_df",
            ]:
                st.session_state[key] = pd.DataFrame()
            st.session_state.cypher_files = {}
            st.session_state.nav_page = "语料整理"
            st.rerun()
    with mid:
        if st.button("使用示例数据", use_container_width=True):
            load_example_pipeline()
            st.session_state.nav_page = "知识抽取"
            st.rerun()
    with right:
        if st.button("进入语料整理", use_container_width=True):
            st.session_state.nav_page = "语料整理"
            st.rerun()

    if not st.session_state.entities_df.empty:
        st.subheader("当前实体预览")
        st.dataframe(st.session_state.entities_df.head(8), use_container_width=True, hide_index=True)


def page_corpus() -> None:
    section_header("语料整理")
    with st.form("corpus_form"):
        c1, c2 = st.columns([1, 2])
        paper_id = c1.text_input("文献编号", value="P001")
        title = c2.text_input("文献标题", value="羊场磷矿床示例")
        c3, c4 = st.columns(2)
        page = c3.text_input("页码", value="1")
        section = c4.text_input("章节", value="示例段落")
        raw_text = st.text_area("原始文本", value=load_example_text(), height=220)
        uploaded = st.file_uploader("上传 TXT / Markdown / CSV", type=["txt", "md", "csv"])
        submitted = st.form_submit_button("生成语料表", use_container_width=True)

    if submitted:
        if uploaded is not None:
            uploaded_df, payload = read_uploaded_file(uploaded)
            if uploaded_df is not None:
                st.session_state.corpus_df = uploaded_df
            elif payload and not payload.startswith("文件编码"):
                st.session_state.corpus_df = build_corpus_table(paper_id, title, page, section, payload)
            else:
                st.error(payload)
        else:
            st.session_state.corpus_df = build_corpus_table(paper_id, title, page, section, raw_text)

    if not st.session_state.corpus_df.empty:
        st.session_state.corpus_df = st.data_editor(
            st.session_state.corpus_df,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            key="corpus_editor",
        )
        show_download_buttons(st.session_state.corpus_df, "corpus")


def page_extraction() -> None:
    section_header("知识抽取")
    if st.session_state.corpus_df.empty:
        st.info("请先生成或载入语料表。")
        return

    c1, c2 = st.columns([1, 2])
    mode = c1.selectbox("抽取模式", ["规则抽取", "大模型抽取（未配置时回退规则）"])
    c2.write("")
    c2.write("")
    if st.button("开始知识抽取", use_container_width=True):
        if mode.startswith("大模型"):
            st.info("当前未接入 API Key，已使用离线规则抽取。")
        extraction = run_extraction(st.session_state.corpus_df)
        st.session_state.entities_df = extraction["entities"]
        st.session_state.relations_df = extraction["relations"]
        st.session_state.attributes_df = extraction["attributes"]
        st.session_state.rules_df = extraction["rules"]

    tabs = st.tabs(["实体表", "关系表", "属性表", "找矿规则表"])
    table_specs = [
        ("entities_df", "entities"),
        ("relations_df", "relations"),
        ("attributes_df", "attributes"),
        ("rules_df", "rules"),
    ]
    for tab, (state_key, stem) in zip(tabs, table_specs):
        with tab:
            df = st.session_state[state_key]
            if df.empty:
                st.info("暂无数据。")
            else:
                st.session_state[state_key] = st.data_editor(df, use_container_width=True, num_rows="dynamic", hide_index=True, key=f"{state_key}_editor")
                show_download_buttons(st.session_state[state_key], stem)


def page_ontology() -> None:
    section_header("本体校验")
    if st.session_state.entities_df.empty and st.session_state.relations_df.empty:
        st.info("请先执行知识抽取。")
        return

    c1, c2 = st.columns(2)
    if c1.button("执行本体校验", use_container_width=True):
        st.session_state.ontology_review_df = build_ontology_review(st.session_state.entities_df, st.session_state.relations_df)
        if not st.session_state.relations_df.empty and not st.session_state.ontology_review_df.empty:
            valid_relation_ids = set(
                st.session_state.ontology_review_df[
                    (st.session_state.ontology_review_df["item_kind"] == "relation")
                    & (st.session_state.ontology_review_df["is_valid"])
                ]["item_id"]
            )
            st.session_state.relations_df.loc[:, "ontology_check"] = st.session_state.relations_df["relation_id"].apply(
                lambda item_id: "通过" if item_id in valid_relation_ids else "需确认"
            )
    if c2.button("应用映射建议", use_container_width=True):
        review = st.session_state.ontology_review_df
        st.session_state.entities_df = apply_mapping(st.session_state.entities_df, review[review["item_kind"] == "entity"] if not review.empty else review)
        st.session_state.relations_df = apply_mapping(st.session_state.relations_df, review[review["item_kind"] == "relation"] if not review.empty else review)
        st.success("已应用可自动处理的映射建议。")

    review_df = st.session_state.ontology_review_df
    if review_df.empty:
        st.info("暂无校验结果。")
    else:
        st.dataframe(review_df, use_container_width=True, hide_index=True)
        show_download_buttons(review_df, "ontology_review")


def page_cypher() -> None:
    section_header("Neo4j 脚本生成")
    if st.session_state.entities_df.empty:
        st.info("请先执行知识抽取。")
        return
    names = st.session_state.entities_df["entity_name"].dropna().astype(str).unique().tolist()
    core_node = st.selectbox("核心节点", names, index=0 if names else None)
    if st.button("生成 Neo4j 脚本", use_container_width=True):
        st.session_state.cypher_files = generate_all_cypher(st.session_state.entities_df, st.session_state.relations_df, core_node)

    if st.session_state.cypher_files:
        tabs = st.tabs(list(st.session_state.cypher_files.keys()))
        for tab, (file_name, content) in zip(tabs, st.session_state.cypher_files.items()):
            with tab:
                st.code(content, language="cypher")
                st.download_button("下载脚本", data=content.encode("utf-8"), file_name=file_name, mime="text/plain", use_container_width=True)


def page_hop() -> None:
    section_header("1-3 跳分析")
    if st.session_state.entities_df.empty:
        st.info("请先执行知识抽取。")
        return
    names = st.session_state.entities_df["entity_name"].dropna().astype(str).unique().tolist()
    c1, c2 = st.columns([2, 1])
    core_node = c1.selectbox("核心节点", names, index=0)
    max_hop = c2.slider("最大跳数", 1, 5, 3)
    if st.button("生成 1-3 跳分析", use_container_width=True):
        st.session_state.hop_query = build_hop_query(core_node, max_hop)
        st.session_state.hop_markdown = build_hop_analysis_template(core_node, f"1-{max_hop}")

    if st.session_state.hop_query:
        st.code(st.session_state.hop_query, language="cypher")
        st.download_button("下载 query.cypher", st.session_state.hop_query.encode("utf-8"), "query.cypher", "text/plain", use_container_width=True)
    if st.session_state.hop_markdown:
        st.markdown(st.session_state.hop_markdown)
        st.download_button("下载 hop_analysis.md", st.session_state.hop_markdown.encode("utf-8"), "hop_analysis.md", "text/markdown", use_container_width=True)


def page_candidate() -> None:
    section_header("候选区解释")
    candidate_area = st.text_input("候选区名称", value="羊场外围候选区")
    defaults = {
        "赋矿层位": "寒武系梅树村组含磷岩系匹配，具备有利层位条件",
        "岩性组合": "发育磷块岩、白云质磷块岩和含磷白云岩",
        "构造条件": "背斜和断裂构造发育，对矿体展布有一定控制作用",
        "沉积环境": "浅海台地沉积环境有利于磷质沉积",
        "地球化学指标": "P2O5 含量为重要评价指标，具体数值需补充",
        "资源属性": "矿层厚度被作为资源潜力评价指标",
        "找矿标志": "含磷岩系、磷块岩和浅海台地环境可作为综合找矿线索",
    }
    factors = {}
    cols = st.columns(2)
    for idx, factor in enumerate(FACTOR_WEIGHTS.keys()):
        with cols[idx % 2]:
            factors[factor] = st.text_area(factor, value=defaults[factor], height=90)

    if st.button("生成候选区解释", use_container_width=True):
        st.session_state.candidate_result = analyze_candidate_area(candidate_area, factors)
        st.session_state.candidate_markdown = build_candidate_markdown(st.session_state.candidate_result)

    if st.session_state.candidate_result:
        result = st.session_state.candidate_result
        c1, c2, c3 = st.columns(3)
        c1.metric("综合评分", result["score"])
        c2.metric("潜力等级", result["level"])
        c3.metric("需人工复核", "是" if result["need_review"] else "否")
        st.dataframe(pd.DataFrame(result["details"]), use_container_width=True, hide_index=True)
        st.markdown(st.session_state.candidate_markdown)
        st.download_button("下载 candidate_analysis.md", st.session_state.candidate_markdown.encode("utf-8"), "candidate_analysis.md", "text/markdown", use_container_width=True)


def build_gnn_state() -> None:
    nodes_df = build_nodes(st.session_state.entities_df)
    edges_df = build_edges(st.session_state.relations_df, nodes_df)
    st.session_state.nodes_df = nodes_df
    st.session_state.edges_df = edges_df
    st.session_state.node_types_df = build_node_types(st.session_state.entities_df)
    st.session_state.relation_types_df = build_relation_types(st.session_state.relations_df)
    st.session_state.model_plan = build_model_plan()


def page_gnn() -> None:
    section_header("GNN 数据准备")
    if st.session_state.entities_df.empty:
        st.info("请先执行知识抽取。")
        return
    if st.button("生成 GNN 数据", use_container_width=True):
        build_gnn_state()

    tabs = st.tabs(["nodes.csv", "edges.csv", "node_types.csv", "relation_types.csv", "model_plan.md"])
    with tabs[0]:
        st.dataframe(st.session_state.nodes_df, use_container_width=True, hide_index=True)
        show_download_buttons(st.session_state.nodes_df, "nodes")
    with tabs[1]:
        st.dataframe(st.session_state.edges_df, use_container_width=True, hide_index=True)
        show_download_buttons(st.session_state.edges_df, "edges")
    with tabs[2]:
        st.dataframe(st.session_state.node_types_df, use_container_width=True, hide_index=True)
        show_download_buttons(st.session_state.node_types_df, "node_types")
    with tabs[3]:
        st.dataframe(st.session_state.relation_types_df, use_container_width=True, hide_index=True)
        show_download_buttons(st.session_state.relation_types_df, "relation_types")
    with tabs[4]:
        st.markdown(st.session_state.model_plan or build_model_plan())
        st.download_button("下载 model_plan.md", (st.session_state.model_plan or build_model_plan()).encode("utf-8"), "model_plan.md", "text/markdown", use_container_width=True)


def save_all_outputs() -> Path:
    output_dir = ensure_output_dir()
    tables = {
        "corpus.csv": st.session_state.corpus_df,
        "entities.csv": st.session_state.entities_df,
        "relations.csv": st.session_state.relations_df,
        "attributes.csv": st.session_state.attributes_df,
        "rules.csv": st.session_state.rules_df,
        "ontology_review.csv": st.session_state.ontology_review_df,
        "nodes.csv": st.session_state.nodes_df,
        "edges.csv": st.session_state.edges_df,
        "node_types.csv": st.session_state.node_types_df,
        "relation_types.csv": st.session_state.relation_types_df,
    }
    for file_name, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            save_dataframe(df, output_dir / file_name)

    if not st.session_state.cypher_files and not st.session_state.entities_df.empty:
        st.session_state.cypher_files = generate_all_cypher(st.session_state.entities_df, st.session_state.relations_df, first_entity_name())
    for file_name, content in st.session_state.cypher_files.items():
        save_text(content, output_dir / file_name)

    if st.session_state.hop_query:
        save_text(st.session_state.hop_query, output_dir / "hop_query.cypher")
    if st.session_state.hop_markdown:
        save_text(st.session_state.hop_markdown, output_dir / "hop_analysis.md")
    if st.session_state.candidate_markdown:
        save_text(st.session_state.candidate_markdown, output_dir / "candidate_analysis.md")
    save_text(st.session_state.model_plan or build_model_plan(), output_dir / "model_plan.md")
    return output_dir


def page_export() -> None:
    section_header("数据导出")
    output_dir = ROOT / "data" / "output"
    if st.button("导出当前结果", use_container_width=True):
        save_all_outputs()
        st.success(f"已保存到 {output_dir}")

    if st.button("打包下载全部文件", use_container_width=True):
        output_dir = save_all_outputs()
        zip_path = output_dir / "phosphate_kg_outputs.zip"
        export_project_zip(output_dir, zip_path)
        st.session_state.export_zip_path = zip_path

    zip_path = st.session_state.get("export_zip_path")
    if zip_path and Path(zip_path).exists():
        st.download_button(
            "下载 ZIP",
            data=Path(zip_path).read_bytes(),
            file_name="phosphate_kg_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if output_dir.exists():
        files = sorted([file.name for file in output_dir.iterdir() if file.is_file()])
        st.dataframe(pd.DataFrame({"output_file": files}), use_container_width=True, hide_index=True)


def page_help() -> None:
    section_header("使用说明")
    st.markdown(
        """
        ## 运行方式
        ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```

        ## 标准流程
        1. 在“语料整理”中粘贴文本或上传 TXT / Markdown / CSV。
        2. 在“知识抽取”中生成实体、关系、属性和找矿规则。
        3. 在“本体校验”中检查类型和关系是否符合内置本体。
        4. 生成 Neo4j Cypher、邻域分析、候选区解释和 GNN 数据。
        5. 在“数据导出”中保存 CSV、JSON、Markdown、Cypher 和 ZIP 文件。

        ## 边界
        当前版本为本地原型，默认使用规则抽取，不训练图神经网络，不强制连接 Neo4j，不生成组会、周报、开题报告或简历类写作内容。
        """
    )


def main() -> None:
    st.set_page_config(page_title="沉积型磷矿知识图谱工作流智能体", layout="wide")
    init_state()
    apply_theme()
    page = sidebar()
    page_map = {
        "项目首页": page_home,
        "语料整理": page_corpus,
        "知识抽取": page_extraction,
        "本体校验": page_ontology,
        "Neo4j 脚本生成": page_cypher,
        "1-3 跳分析": page_hop,
        "候选区解释": page_candidate,
        "GNN 数据准备": page_gnn,
        "数据导出": page_export,
        "使用说明": page_help,
    }
    page_map[page]()


if __name__ == "__main__":
    main()
