from __future__ import annotations

import html
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

PROJECT_TITLE = "沉积型磷矿知识图谱工作流智能体"
PROJECT_SUBTITLE = "面向沉积型磷矿找矿预测的知识图谱构建与分析平台"

PAGES = [
    "项目首页",
    "文献输入",
    "语料整理",
    "知识抽取",
    "本体校验",
    "Cypher生成",
    "1-3跳变分析",
    "候选区解释",
    "GNN数据准备",
    "结果导出",
    "使用说明",
]


def load_example_text() -> str:
    return EXAMPLE_TEXT_PATH.read_text(encoding="utf-8")


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
        "current_page": "项目首页",
        "extraction_mode": "规则抽取",
        "project_name": "默认项目",
        "input_paper_id": "P001",
        "input_title": "羊场磷矿床示例",
        "input_page": "1",
        "input_section": "示例段落",
        "input_raw_text": load_example_text(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to_page(page_name: str) -> None:
    st.session_state.current_page = page_name
    st.rerun()


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111F;
            --bg-soft: #0B1220;
            --panel: #111827;
            --panel-2: #162033;
            --line: rgba(0, 212, 255, 0.18);
            --line-strong: rgba(0, 212, 255, 0.42);
            --cyan: #00D4FF;
            --purple: #7C3AED;
            --green: #22C55E;
            --orange: #F59E0B;
            --red: #EF4444;
            --muted: #94A3B8;
            --text: #E5F2FF;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(0, 212, 255, 0.10), transparent 26%),
                radial-gradient(circle at 78% 8%, rgba(124, 58, 237, 0.16), transparent 30%),
                linear-gradient(135deg, #07111F 0%, #0B1220 45%, #111827 100%);
            color: var(--text);
            font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.1rem;
            padding-bottom: 3rem;
        }

        header[data-testid="stHeader"] {
            background: rgba(7, 17, 31, 0.72);
            backdrop-filter: blur(14px);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(7, 17, 31, 0.98), rgba(11, 18, 32, 0.98));
            border-right: 1px solid rgba(0, 212, 255, 0.18);
            box-shadow: 10px 0 32px rgba(0, 0, 0, 0.22);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #D8ECFF !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 8px;
            padding: 9px 10px;
            margin: 5px 0;
            background: rgba(17, 24, 39, 0.72);
            transition: all 0.15s ease;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover,
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            border-color: rgba(0, 212, 255, 0.68);
            background: linear-gradient(90deg, rgba(0, 212, 255, 0.18), rgba(124, 58, 237, 0.12));
            box-shadow: 0 0 18px rgba(0, 212, 255, 0.12);
        }

        h1, h2, h3, h4, h5, h6, p, li, label, span {
            letter-spacing: 0;
        }

        h1, h2, h3 {
            color: #F8FBFF;
        }

        .top-shell {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 18px;
            align-items: center;
            padding: 22px 24px;
            margin-bottom: 18px;
            border: 1px solid rgba(0, 212, 255, 0.22);
            border-radius: 8px;
            background:
                linear-gradient(115deg, rgba(17, 24, 39, 0.94), rgba(22, 32, 51, 0.82)),
                linear-gradient(90deg, rgba(0, 212, 255, 0.08), rgba(124, 58, 237, 0.10));
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.08);
        }

        .top-title {
            margin: 0;
            font-size: clamp(1.45rem, 2.4vw, 2.25rem);
            font-weight: 800;
            color: #F8FBFF;
        }

        .top-subtitle {
            margin-top: 8px;
            color: var(--muted);
            font-size: 0.98rem;
        }

        .status-grid {
            display: grid;
            gap: 8px;
            min-width: 260px;
        }

        .status-pill {
            display: flex;
            justify-content: space-between;
            gap: 14px;
            padding: 8px 10px;
            border: 1px solid rgba(0, 212, 255, 0.18);
            border-radius: 8px;
            background: rgba(2, 6, 23, 0.46);
            color: #D8ECFF;
            font-size: 0.86rem;
        }

        .status-pill strong {
            color: var(--cyan);
            font-weight: 700;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(300px, 0.85fr);
            gap: 18px;
            margin-bottom: 18px;
        }

        .dashboard-card, .hero-card, .page-title-card, .info-card, .code-card, .file-card {
            border: 1px solid rgba(0, 212, 255, 0.18);
            border-radius: 8px;
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.95), rgba(22, 32, 51, 0.88));
            box-shadow: 0 0 18px rgba(0, 212, 255, 0.08);
        }

        .hero-card {
            min-height: 230px;
            padding: 26px;
            position: relative;
            overflow: hidden;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            inset: auto -20% -60% 30%;
            height: 190px;
            background: radial-gradient(circle, rgba(0, 212, 255, 0.20), transparent 58%);
            pointer-events: none;
        }

        .hero-kicker, .card-kicker {
            color: var(--cyan);
            font-weight: 800;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 12px 0 12px;
            font-size: clamp(1.65rem, 2.7vw, 2.5rem);
            line-height: 1.18;
            font-weight: 850;
            color: #F8FBFF;
        }

        .hero-copy {
            color: #B8C7DA;
            font-size: 1rem;
            line-height: 1.75;
            max-width: 760px;
        }

        .page-title-card {
            padding: 18px 20px;
            margin-bottom: 16px;
        }

        .page-title-card h2 {
            margin: 7px 0 8px;
            font-size: 1.65rem;
        }

        .page-title-card p, .info-card p {
            margin: 0;
            color: #AFC1D6;
            line-height: 1.7;
        }

        .info-card {
            padding: 17px 18px;
            margin-bottom: 14px;
        }

        .info-card h3 {
            margin: 0 0 10px;
            font-size: 1.03rem;
            color: #F8FBFF;
        }

        .info-card ul {
            margin: 0;
            padding-left: 1.1rem;
            color: #B8C7DA;
            line-height: 1.8;
        }

        .metric-card {
            padding: 16px 16px 14px;
            border: 1px solid rgba(0, 212, 255, 0.18);
            border-radius: 8px;
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.94), rgba(2, 6, 23, 0.62));
            box-shadow: 0 0 18px rgba(0, 212, 255, 0.08);
            min-height: 112px;
        }

        .metric-label {
            color: var(--cyan);
            font-size: 0.86rem;
            font-weight: 750;
        }

        .metric-value {
            margin-top: 8px;
            color: #FFFFFF;
            font-size: 2.15rem;
            line-height: 1;
            font-weight: 850;
        }

        .metric-note {
            margin-top: 8px;
            color: var(--muted);
            font-size: 0.78rem;
        }

        .workflow-row {
            display: flex;
            flex-wrap: wrap;
            align-items: stretch;
            gap: 9px;
            margin: 10px 0 6px;
        }

        .workflow-node {
            flex: 1 1 118px;
            min-width: 118px;
            padding: 12px 10px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            background: rgba(17, 24, 39, 0.82);
            color: #D8ECFF;
            text-align: center;
            font-size: 0.9rem;
        }

        .workflow-node.active {
            border-color: rgba(0, 212, 255, 0.78);
            color: var(--cyan);
            box-shadow: 0 0 18px rgba(0, 212, 255, 0.16);
        }

        .workflow-arrow {
            display: flex;
            align-items: center;
            color: rgba(0, 212, 255, 0.62);
            font-weight: 800;
        }

        .section-title {
            margin: 22px 0 10px;
            color: #F8FBFF;
            font-size: 1.08rem;
            font-weight: 760;
        }

        .stButton > button, .stDownloadButton > button, button[kind="primary"] {
            border: 1px solid rgba(0, 212, 255, 0.42) !important;
            border-radius: 8px !important;
            background: linear-gradient(90deg, rgba(0, 212, 255, 0.96), rgba(124, 58, 237, 0.92)) !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            box-shadow: 0 0 18px rgba(0, 212, 255, 0.18) !important;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(0, 212, 255, 0.88) !important;
            box-shadow: 0 0 26px rgba(0, 212, 255, 0.28) !important;
        }

        div[data-testid="stForm"], div[data-testid="stExpander"], div[data-testid="stDataFrame"] {
            border: 1px solid rgba(0, 212, 255, 0.14);
            border-radius: 8px;
            background: rgba(17, 24, 39, 0.60);
        }

        textarea, input, div[data-baseweb="select"] > div, div[data-baseweb="textarea"] textarea {
            background-color: rgba(2, 6, 23, 0.72) !important;
            border-color: rgba(0, 212, 255, 0.20) !important;
            color: #E5F2FF !important;
        }

        input[type="radio"], input[type="checkbox"] {
            accent-color: #00D4FF !important;
        }

        div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(0, 212, 255, 0.32);
            border-radius: 8px;
            padding: 10px;
            background: rgba(2, 6, 23, 0.42);
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(0, 212, 255, 0.18);
            border-radius: 8px;
            padding: 12px 14px;
            background: rgba(17, 24, 39, 0.82);
            box-shadow: 0 0 18px rgba(0, 212, 255, 0.08);
        }

        div[data-testid="stMetric"] label, div[data-testid="stMetric"] div {
            color: #E5F2FF !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid rgba(0, 212, 255, 0.16);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            background: rgba(17, 24, 39, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.14);
            color: #D8ECFF;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(0, 212, 255, 0.15);
            border-color: rgba(0, 212, 255, 0.42);
            color: var(--cyan);
        }

        pre, code {
            background-color: #020617 !important;
            color: #D8ECFF !important;
            border-radius: 8px !important;
        }

        .rank-high { color: var(--green); border-color: rgba(34, 197, 94, 0.40); }
        .rank-mid { color: var(--orange); border-color: rgba(245, 158, 11, 0.40); }
        .rank-low { color: #94A3B8; border-color: rgba(148, 163, 184, 0.30); }
        .rank-review { color: var(--red); border-color: rgba(239, 68, 68, 0.42); }

        .file-card {
            padding: 15px;
            margin-bottom: 12px;
        }

        .file-card .file-name {
            color: var(--cyan);
            font-weight: 800;
            margin-bottom: 5px;
        }

        .file-card .file-meta {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.6;
        }

        @media (max-width: 900px) {
            .top-shell, .hero-grid {
                grid-template-columns: 1fr;
            }
            .status-grid {
                min-width: 0;
            }
            .workflow-arrow {
                display: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def esc(value: object) -> str:
    return html.escape(str(value))


def review_count() -> int:
    review_df = st.session_state.ontology_review_df
    if review_df.empty or "need_review" not in review_df:
        return 0
    return int(review_df["need_review"].fillna(False).sum())


def corpus_priority_counts() -> tuple[int, int, int]:
    corpus_df = st.session_state.corpus_df
    if corpus_df.empty or "priority" not in corpus_df:
        return 0, 0, 0
    priorities = corpus_df["priority"].astype(str)
    return int((priorities == "高").sum()), int((priorities == "中").sum()), int((priorities == "低").sum())


def render_app_header() -> None:
    st.markdown(
        f"""
        <div class="top-shell">
            <div>
                <h1 class="top-title">{PROJECT_TITLE}</h1>
                <div class="top-subtitle">{PROJECT_SUBTITLE}</div>
            </div>
            <div class="status-grid">
                <div class="status-pill"><span>系统状态</span><strong>运行中</strong></div>
                <div class="status-pill"><span>当前模式</span><strong>{esc(st.session_state.extraction_mode)}</strong></div>
                <div class="status-pill"><span>当前项目</span><strong>{esc(st.session_state.project_name)}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_title(label: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="page-title-card">
            <div class="card-kicker">{esc(label)}</div>
            <h2>{esc(title)}</h2>
            <p>{esc(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body_html: str, label: str = "") -> None:
    kicker = f'<div class="card-kicker">{esc(label)}</div>' if label else ""
    st.markdown(
        f"""
        <div class="info-card">
            {kicker}
            <h3>{esc(title)}</h3>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: object, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{esc(label)}</div>
            <div class="metric-value">{esc(value)}</div>
            <div class="metric-note">{esc(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{esc(title)}</div>', unsafe_allow_html=True)


def workflow_html(active_step: str = "") -> str:
    steps = ["文献输入", "语料整理", "知识抽取", "本体校验", "Cypher 生成", "1-3 跳分析", "候选区解释", "GNN 数据准备", "结果导出"]
    parts = ['<div class="workflow-row">']
    for idx, step in enumerate(steps):
        cls = "workflow-node active" if step == active_step else "workflow-node"
        parts.append(f'<div class="{cls}">{esc(step)}</div>')
        if idx < len(steps) - 1:
            parts.append('<div class="workflow-arrow">→</div>')
    parts.append("</div>")
    return "".join(parts)


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


def reset_project_state() -> None:
    for key, columns in {
        "corpus_df": CORPUS_COLUMNS,
        "entities_df": ENTITY_COLUMNS,
        "relations_df": RELATION_COLUMNS,
        "attributes_df": ATTRIBUTE_COLUMNS,
        "rules_df": RULE_COLUMNS,
        "ontology_review_df": REVIEW_COLUMNS,
    }.items():
        st.session_state[key] = empty_df(columns)
    st.session_state.cypher_files = {}
    st.session_state.hop_query = ""
    st.session_state.hop_markdown = ""
    st.session_state.candidate_result = {}
    st.session_state.candidate_markdown = ""
    st.session_state.nodes_df = pd.DataFrame()
    st.session_state.edges_df = pd.DataFrame()
    st.session_state.node_types_df = pd.DataFrame()
    st.session_state.relation_types_df = pd.DataFrame()
    st.session_state.model_plan = ""


def first_entity_name() -> str:
    entities = st.session_state.entities_df
    if not entities.empty and "entity_name" in entities:
        return str(entities.iloc[0]["entity_name"])
    return "羊场磷矿床"


def highlight_need_review(df: pd.DataFrame):
    if df.empty or "need_review" not in df.columns:
        return df

    def style_row(row: pd.Series) -> list[str]:
        flag = str(row.get("need_review", "")).lower() in {"true", "1", "yes", "是"}
        return ["background-color: rgba(245, 158, 11, 0.20); color: #FDE68A;" if flag else "" for _ in row]

    return df.style.apply(style_row, axis=1)


def one_hop_query(core_node: str) -> str:
    safe = str(core_node).replace("'", "\\'")
    return f"""MATCH (n:Entity {{name: '{safe}'}})-[r]-(m:Entity)
RETURN n, r, m
LIMIT 100;"""


def sidebar() -> str:
    st.sidebar.markdown("## 工作流导航")
    if st.session_state.current_page not in PAGES:
        st.session_state.current_page = "项目首页"

    selected_page = st.sidebar.radio(
        "功能导航",
        PAGES,
        index=PAGES.index(st.session_state.current_page),
    )
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

    st.sidebar.divider()
    st.sidebar.metric("语料段落", len(st.session_state.corpus_df))
    st.sidebar.metric("实体", len(st.session_state.entities_df))
    st.sidebar.metric("关系", len(st.session_state.relations_df))
    st.sidebar.metric("待复核", review_count())
    st.sidebar.divider()
    if st.sidebar.button("载入样本数据", use_container_width=True):
        load_example_pipeline()
        go_to_page("知识抽取")
    return st.session_state.current_page


def page_home() -> None:
    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="hero-kicker">MINERAL KNOWLEDGE GRAPH PLATFORM</div>
                <div class="hero-title">{PROJECT_TITLE}</div>
                <div class="hero-copy">
                    将地质文献中的成矿知识转化为可分析、可入库、可建模的知识图谱数据。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        info_card(
            "当前流程",
            workflow_html("文献输入"),
            "WORKFLOW",
        )

    section_title("数据总览")
    metric_cols = st.columns(5)
    paper_count = st.session_state.corpus_df["paper_id"].nunique() if not st.session_state.corpus_df.empty else 0
    metrics = [
        ("文献数量", paper_count, "已登记来源"),
        ("语料段落", len(st.session_state.corpus_df), "标准语料表"),
        ("实体数量", len(st.session_state.entities_df), "知识图谱节点"),
        ("关系数量", len(st.session_state.relations_df), "知识图谱边"),
        ("待确认", review_count(), "人工复核项"),
    ]
    for col, (label, value, note) in zip(metric_cols, metrics):
        with col:
            metric_card(label, value, note)

    section_title("工作流流程")
    st.markdown(workflow_html("文献输入"), unsafe_allow_html=True)

    section_title("快速开始")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("使用样本数据", use_container_width=True):
            load_example_pipeline()
            go_to_page("知识抽取")
    with c2:
        if st.button("进入文献输入", use_container_width=True):
            go_to_page("文献输入")
    with c3:
        if st.button("开始语料整理", use_container_width=True):
            go_to_page("语料整理")
    with c4:
        if st.button("生成 Neo4j 脚本", use_container_width=True):
            go_to_page("Cypher生成")
    with c5:
        if st.button("导出结果", use_container_width=True):
            go_to_page("结果导出")

    if not st.session_state.entities_df.empty:
        section_title("当前实体预览")
        st.dataframe(st.session_state.entities_df.head(8), use_container_width=True, hide_index=True)


def page_literature_input() -> None:
    page_title(
        "DATA INPUT",
        "文献输入",
        "录入论文段落、地质报告或勘查资料，为后续语料整理和知识抽取保留来源信息。",
    )
    left, right = st.columns([1.7, 0.9])
    with left:
        section_title("文献信息输入")
        with st.form("literature_input_form"):
            c1, c2 = st.columns([1, 2])
            paper_id = c1.text_input("文献编号", value=st.session_state.input_paper_id)
            title = c2.text_input("文献标题", value=st.session_state.input_title)
            c3, c4 = st.columns(2)
            page = c3.text_input("页码", value=st.session_state.input_page)
            section = c4.text_input("章节", value=st.session_state.input_section)
            raw_text = st.text_area("文本输入", value=st.session_state.input_raw_text, height=320)
            uploaded = st.file_uploader("文件上传", type=["txt", "md", "csv"])
            submitted = st.form_submit_button("保存文献", use_container_width=True)

        if submitted:
            uploaded_df, payload = (None, "")
            if uploaded is not None:
                uploaded_df, payload = read_uploaded_file(uploaded)
            if uploaded_df is not None:
                st.session_state.corpus_df = uploaded_df
                st.success("已从 CSV 生成语料表。")
            elif payload and not payload.startswith("文件编码") and not payload.startswith("CSV 读取失败"):
                raw_text = payload
            elif payload:
                st.error(payload)

            st.session_state.input_paper_id = paper_id
            st.session_state.input_title = title
            st.session_state.input_page = page
            st.session_state.input_section = section
            st.session_state.input_raw_text = raw_text
            st.success("文献信息已保存。")

    with right:
        info_card(
            "操作说明",
            """
            <ul>
                <li>支持输入论文段落、地质报告文本和勘查资料摘录。</li>
                <li>建议保留页码、章节和来源信息，便于证据句追溯。</li>
                <li>CSV 可包含 raw_text 字段或标准语料表字段。</li>
            </ul>
            """,
            "INPUT GUIDE",
        )
        if st.button("载入示例文本", use_container_width=True):
            st.session_state.input_paper_id = "P001"
            st.session_state.input_title = "羊场磷矿床示例"
            st.session_state.input_page = "1"
            st.session_state.input_section = "示例段落"
            st.session_state.input_raw_text = load_example_text()
            st.rerun()


def page_corpus() -> None:
    page_title(
        "CORPUS PROCESSING",
        "语料整理",
        "将已保存的文献文本切分、清洗并标注为标准语料表。",
    )
    st.markdown(workflow_html("语料整理"), unsafe_allow_html=True)

    high_count, mid_count, low_count = corpus_priority_counts()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("语料段落", len(st.session_state.corpus_df), "已生成记录")
    with c2:
        metric_card("高优先级", high_count, "矿床/矿体/找矿信息")
    with c3:
        metric_card("中优先级", mid_count, "地层/岩性/构造信息")
    with c4:
        metric_card("低优先级", low_count, "背景性描述")

    section_title("语料处理状态")
    info_card(
        "处理输入",
        f"<p>当前文献：{esc(st.session_state.input_title)}；文本长度：{len(st.session_state.input_raw_text)} 字。</p>",
        "STATUS",
    )
    if st.button("生成语料表", use_container_width=True):
        if not st.session_state.input_raw_text.strip():
            st.warning("请先在文献输入页面保存文本。")
        else:
            st.session_state.corpus_df = build_corpus_table(
                st.session_state.input_paper_id,
                st.session_state.input_title,
                st.session_state.input_page,
                st.session_state.input_section,
                st.session_state.input_raw_text,
            )
            st.success("语料表已生成。")

    high_count, mid_count, low_count = corpus_priority_counts()
    if not st.session_state.corpus_df.empty:
        section_title(
            f"语料表预览：已生成语料段落 {len(st.session_state.corpus_df)} 条，其中高优先级 {high_count} 条，中优先级 {mid_count} 条，低优先级 {low_count} 条"
        )
        st.session_state.corpus_df = st.data_editor(
            st.session_state.corpus_df,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            key="corpus_editor",
        )
        section_title("下载导出")
        show_download_buttons(st.session_state.corpus_df, "corpus")


def page_extraction() -> None:
    page_title(
        "KNOWLEDGE EXTRACTION",
        "知识抽取",
        "从标准语料表中抽取实体、关系、属性和找矿规则，并保留证据句。",
    )
    if st.session_state.corpus_df.empty:
        st.info("请先生成或载入语料表。")
        return

    section_title("抽取模式")
    mode = st.radio("抽取模式", ["规则抽取", "大模型抽取（未配置时回退规则）"], horizontal=True)
    st.session_state.extraction_mode = mode
    if st.button("开始知识抽取", use_container_width=True):
        if mode.startswith("大模型"):
            st.info("当前未接入 API Key，已使用离线规则抽取。")
        extraction = run_extraction(st.session_state.corpus_df)
        st.session_state.entities_df = extraction["entities"]
        st.session_state.relations_df = extraction["relations"]
        st.session_state.attributes_df = extraction["attributes"]
        st.session_state.rules_df = extraction["rules"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("实体数量", len(st.session_state.entities_df), "entities")
    with c2:
        metric_card("关系数量", len(st.session_state.relations_df), "relations")
    with c3:
        metric_card("属性数量", len(st.session_state.attributes_df), "attributes")
    with c4:
        metric_card("规则数量", len(st.session_state.rules_df), "rules")

    section_title("抽取结果预览")
    tabs = st.tabs(["实体", "关系", "属性", "规则"])
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
                st.session_state[state_key] = st.data_editor(
                    df,
                    use_container_width=True,
                    num_rows="dynamic",
                    hide_index=True,
                    key=f"{state_key}_editor",
                )
                show_download_buttons(st.session_state[state_key], stem)


def page_ontology() -> None:
    page_title(
        "ONTOLOGY CHECK",
        "本体校验",
        "检查实体类型、关系类型和方向是否符合沉积型磷矿知识图谱本体。",
    )
    if st.session_state.entities_df.empty and st.session_state.relations_df.empty:
        st.info("请先执行知识抽取。")
        return

    info_card(
        "校验说明",
        "<p>系统会检查类型枚举、映射建议、重复实体和需人工确认项。need_review=true 的记录会以警告色显示。</p>",
        "CHECK RULES",
    )
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
    entity_review = review_df[review_df["item_kind"] == "entity"] if not review_df.empty else pd.DataFrame()
    relation_review = review_df[review_df["item_kind"] == "relation"] if not review_df.empty else pd.DataFrame()
    stats = [
        ("合规实体", int(entity_review["is_valid"].fillna(False).sum()) if not entity_review.empty else 0),
        ("需映射实体", int((~entity_review["is_valid"].fillna(False)).sum()) if not entity_review.empty else 0),
        ("合规关系", int(relation_review["is_valid"].fillna(False).sum()) if not relation_review.empty else 0),
        ("需映射关系", int((~relation_review["is_valid"].fillna(False)).sum()) if not relation_review.empty else 0),
        ("需人工确认", review_count()),
    ]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, stats):
        with col:
            metric_card(label, value, "ontology")

    section_title("本体校验表格")
    if review_df.empty:
        st.info("暂无校验结果。")
    else:
        st.dataframe(highlight_need_review(review_df), use_container_width=True, hide_index=True)
        show_download_buttons(review_df, "ontology_review")


def page_cypher() -> None:
    page_title(
        "CYPHER GENERATOR",
        "Cypher生成",
        "生成 Neo4j 约束、节点、关系和邻域查询脚本。",
    )
    if st.session_state.entities_df.empty:
        st.info("请先执行知识抽取。")
        return

    names = st.session_state.entities_df["entity_name"].dropna().astype(str).unique().tolist()
    c1, c2 = st.columns([1.5, 1])
    with c1:
        core_node = st.selectbox("核心节点", names, index=0 if names else None)
    with c2:
        script_label = st.selectbox(
            "Cypher 类型选择",
            ["节点创建脚本", "关系创建脚本", "约束和索引脚本", "1 跳查询脚本", "1-3 跳查询脚本"],
        )

    if st.button("生成 Cypher", use_container_width=True):
        st.session_state.cypher_files = generate_all_cypher(st.session_state.entities_df, st.session_state.relations_df, core_node)

    if not st.session_state.cypher_files:
        st.session_state.cypher_files = generate_all_cypher(st.session_state.entities_df, st.session_state.relations_df, core_node)

    script_map = {
        "节点创建脚本": ("nodes.cypher", st.session_state.cypher_files.get("nodes.cypher", "")),
        "关系创建脚本": ("relations.cypher", st.session_state.cypher_files.get("relations.cypher", "")),
        "约束和索引脚本": ("constraints.cypher", st.session_state.cypher_files.get("constraints.cypher", "")),
        "1 跳查询脚本": ("one_hop_query.cypher", one_hop_query(core_node)),
        "1-3 跳查询脚本": ("query.cypher", st.session_state.cypher_files.get("query.cypher", "")),
    }
    file_name, content = script_map[script_label]

    section_title("代码展示区")
    st.code(content, language="cypher")
    c3, c4 = st.columns(2)
    with c3:
        if st.button("复制按钮", use_container_width=True):
            st.toast("代码块右上角可直接复制脚本。")
    with c4:
        st.download_button("下载 .cypher", data=content.encode("utf-8"), file_name=file_name, mime="text/plain", use_container_width=True)


def page_hop() -> None:
    page_title(
        "GRAPH ANALYSIS",
        "1-3跳变分析",
        "围绕核心节点生成 1-3 跳邻域查询，解释直接成矿要素、间接控矿因素和区域背景信息。",
    )
    if st.session_state.entities_df.empty:
        st.info("请先执行知识抽取。")
        return

    names = st.session_state.entities_df["entity_name"].dropna().astype(str).unique().tolist()
    type_options = sorted(st.session_state.entities_df["entity_type_cn"].dropna().astype(str).unique().tolist())
    c1, c2 = st.columns([2, 1])
    core_node = c1.selectbox("核心节点选择", names, index=0)
    hop_count = c2.radio("跳数选择", [1, 2, 3], format_func=lambda item: f"{item}跳", horizontal=True, index=2)
    focus_types = st.multiselect("关注实体类型", type_options, default=type_options[: min(4, len(type_options))])

    if st.button("生成分析", use_container_width=True):
        st.session_state.hop_query = build_hop_query(core_node, hop_count)
        st.session_state.hop_markdown = build_hop_analysis_template(core_node, f"1-{hop_count}")

    focus_text = "、".join(focus_types) if focus_types else "全部实体类型"
    c3, c4 = st.columns(2)
    with c3:
        info_card(
            "关键路径说明",
            f"<p>核心节点为 {esc(core_node)}，当前关注 {esc(focus_text)}。优先检查 hosted_by、controlled_by、has_lithology、has_geochemical_feature 等关系路径。</p>",
            "PATHS",
        )
    with c4:
        info_card(
            "找矿意义说明",
            "<p>1跳用于识别直接成矿要素，2跳用于发现间接控矿因素，3跳用于补充区域背景和潜在找矿线索。</p>",
            "INTERPRETATION",
        )

    st.markdown(
        """
        <div class="info-card">
            <div class="card-kicker">ANALYSIS OUTPUT</div>
            <h3>分层解释</h3>
            <ul>
                <li>1跳直接成矿要素：赋矿层位、岩性组合、构造控制和地球化学指标。</li>
                <li>2跳间接控矿因素：通过地层、构造或沉积环境连接的复合要素。</li>
                <li>3跳区域背景信息：成矿区带、区域构造和潜在找矿标志组合。</li>
                <li>综合找矿意义：用于候选区证据链组织和后续 GNN 数据准备。</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("对应 Cypher 查询")
    query = st.session_state.hop_query or build_hop_query(core_node, hop_count)
    st.code(query, language="cypher")
    st.download_button("下载 query.cypher", query.encode("utf-8"), "query.cypher", "text/plain", use_container_width=True)


def page_candidate() -> None:
    page_title(
        "CANDIDATE INTERPRETATION",
        "候选区解释",
        "面向候选区评价，组织成矿要素、推荐等级、证据链和不确定性说明。",
    )
    candidate_area = st.text_input("候选区名称", value="羊场外围候选区")

    factor_defaults = {
        "地层条件": "寒武系梅树村组地层条件匹配，具备有利层位条件",
        "含矿层位": "含磷岩系发育，含矿层位清晰",
        "岩性组合": "发育磷块岩、白云质磷块岩和含磷白云岩",
        "构造背景": "背斜和断裂构造发育，对矿体展布有一定控制作用",
        "沉积环境": "浅海台地沉积环境有利于磷质沉积",
        "地球化学异常": "P2O5 含量为重要评价指标，具体数值需补充",
        "找矿标志": "含磷岩系、磷块岩和浅海台地环境可作为综合找矿线索",
    }
    checked: dict[str, bool] = {}
    evidence: dict[str, str] = {}
    section_title("成矿要素勾选区")
    cols = st.columns(2)
    for idx, (factor, default_text) in enumerate(factor_defaults.items()):
        with cols[idx % 2]:
            checked[factor] = st.checkbox(factor, value=True)
            evidence[factor] = st.text_area(f"{factor}证据", value=default_text, height=84, disabled=not checked[factor])

    analyzer_factors = {
        "赋矿层位": "；".join([evidence[name] for name in ["地层条件", "含矿层位"] if checked[name]]) or "未提供",
        "岩性组合": evidence["岩性组合"] if checked["岩性组合"] else "未提供",
        "构造条件": evidence["构造背景"] if checked["构造背景"] else "未提供",
        "沉积环境": evidence["沉积环境"] if checked["沉积环境"] else "未提供",
        "地球化学指标": evidence["地球化学异常"] if checked["地球化学异常"] else "未提供",
        "资源属性": "需结合矿层厚度、品位和资源量进一步确认",
        "找矿标志": evidence["找矿标志"] if checked["找矿标志"] else "未提供",
    }

    if st.button("生成候选区解释", use_container_width=True):
        st.session_state.candidate_result = analyze_candidate_area(candidate_area, analyzer_factors)
        st.session_state.candidate_markdown = build_candidate_markdown(st.session_state.candidate_result)

    if st.session_state.candidate_result:
        result = st.session_state.candidate_result
        level = str(result.get("level", "需确认"))
        if "高" in level:
            level_cls = "rank-high"
        elif "中" in level:
            level_cls = "rank-mid"
        elif "低" in level:
            level_cls = "rank-low"
        else:
            level_cls = "rank-review"

        st.markdown(
            f"""
            <div class="info-card {level_cls}">
                <div class="card-kicker">RECOMMENDATION</div>
                <h3>推荐等级：{esc(level)}</h3>
                <p>综合评分：{esc(result.get("score", 0))} / 100；需人工复核：{"是" if result.get("need_review") else "否"}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            info_card("推荐理由", "<p>推荐等级由赋矿层位、岩性组合、构造背景、沉积环境、地球化学指标和找矿标志共同评分得到。</p>")
            info_card("不确定性说明", "<p>缺少具体数值、空间位置或原文证据的要素仍需人工确认，不作为最终找矿结论。</p>")
        with c2:
            info_card("证据链", "<p>" + "<br>".join(esc(item["evidence"]) for item in result.get("details", []) if item.get("evidence")) + "</p>")
        st.dataframe(pd.DataFrame(result["details"]), use_container_width=True, hide_index=True)
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
    page_title(
        "GNN DATA BUILDER",
        "GNN数据准备",
        "生成 R-GCN / CompGCN 建模所需的节点表、边表、节点类型表、关系类型表和说明文件。",
    )
    if st.session_state.entities_df.empty:
        st.info("请先执行知识抽取。")
        return

    info_card(
        "模型数据准备中心",
        "<p>当前系统暂不训练模型，只生成 R-GCN / CompGCN 建模所需的节点表、边表和说明文件。</p>",
        "MODEL DATA",
    )
    if st.button("生成 GNN 数据", use_container_width=True):
        build_gnn_state()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("节点表", len(st.session_state.nodes_df), "nodes.csv")
    with c2:
        metric_card("边表", len(st.session_state.edges_df), "edges.csv")
    with c3:
        metric_card("节点类型", len(st.session_state.node_types_df), "node_types.csv")
    with c4:
        metric_card("关系类型", len(st.session_state.relation_types_df), "relation_types.csv")

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


def file_description(file_name: str) -> tuple[str, str]:
    suffix = Path(file_name).suffix.lower().lstrip(".") or "file"
    descriptions = {
        "corpus.csv": "标准语料表",
        "entities.csv": "实体抽取结果",
        "relations.csv": "关系抽取结果",
        "attributes.csv": "属性抽取结果",
        "rules.csv": "找矿规则结果",
        "ontology_review.csv": "本体校验结果",
        "import.cypher": "Neo4j 入库脚本",
        "query.cypher": "邻域查询脚本",
        "candidate_analysis.md": "候选区解释报告",
        "nodes.csv": "GNN 节点表",
        "edges.csv": "GNN 边表",
        "model_plan.md": "模型数据说明",
    }
    return suffix.upper(), descriptions.get(file_name, "项目输出文件")


def page_export() -> None:
    page_title(
        "EXPORT CENTER",
        "结果导出",
        "集中管理 CSV、JSON、Markdown、Cypher 和 ZIP 输出文件。",
    )
    output_dir = ROOT / "data" / "output"
    c1, c2 = st.columns(2)
    if c1.button("导出当前结果", use_container_width=True):
        save_all_outputs()
        st.success(f"已保存到 {output_dir}")

    if c2.button("一键打包下载", use_container_width=True):
        output_dir = save_all_outputs()
        zip_path = output_dir / "phosphate_kg_outputs.zip"
        export_project_zip(output_dir, zip_path)
        st.session_state.export_zip_path = zip_path

    files = sorted([file for file in output_dir.iterdir() if file.is_file()]) if output_dir.exists() else []
    info_card(
        "数据完整性检查",
        f"<p>已生成文件 {len(files)} 个；语料 {len(st.session_state.corpus_df)} 条，实体 {len(st.session_state.entities_df)} 个，关系 {len(st.session_state.relations_df)} 条。</p>",
        "INTEGRITY",
    )

    zip_path = st.session_state.get("export_zip_path")
    if zip_path and Path(zip_path).exists():
        st.download_button(
            "下载 ZIP",
            data=Path(zip_path).read_bytes(),
            file_name="phosphate_kg_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )

    section_title("已生成文件列表")
    if not files:
        st.info("暂无输出文件。")
        return

    for file in files:
        file_type, desc = file_description(file.name)
        st.markdown(
            f"""
            <div class="file-card">
                <div class="file-name">{esc(file.name)}</div>
                <div class="file-meta">文件类型：{esc(file_type)}；文件说明：{esc(desc)}；大小：{file.stat().st_size} bytes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            f"下载 {file.name}",
            data=file.read_bytes(),
            file_name=file.name,
            mime="application/octet-stream",
            use_container_width=True,
            key=f"download_{file.name}",
        )


def page_help() -> None:
    page_title(
        "USER GUIDE",
        "使用说明",
        "本地运行、标准流程和系统边界说明。",
    )
    info_card(
        "运行方式",
        """
        <p><code>pip install -r requirements.txt</code></p>
        <p><code>streamlit run app.py</code></p>
        """,
        "RUN",
    )
    info_card(
        "标准流程",
        """
        <ul>
            <li>文献输入：保存文本和来源信息。</li>
            <li>语料整理：生成标准语料表。</li>
            <li>知识抽取：生成实体、关系、属性和找矿规则。</li>
            <li>本体校验：检查类型与关系映射。</li>
            <li>图谱分析与导出：生成 Cypher、候选区解释、GNN 数据和 ZIP 文件。</li>
        </ul>
        """,
        "FLOW",
    )
    info_card(
        "系统边界",
        "<p>当前版本为本地原型，默认使用规则抽取，不训练图神经网络，不强制连接 Neo4j。</p>",
        "BOUNDARY",
    )


def main() -> None:
    st.set_page_config(page_title=PROJECT_TITLE, layout="wide")
    init_state()
    inject_custom_css()
    page = sidebar()
    render_app_header()
    page_map = {
        "项目首页": page_home,
        "文献输入": page_literature_input,
        "语料整理": page_corpus,
        "知识抽取": page_extraction,
        "本体校验": page_ontology,
        "Cypher生成": page_cypher,
        "1-3跳变分析": page_hop,
        "候选区解释": page_candidate,
        "GNN数据准备": page_gnn,
        "结果导出": page_export,
        "使用说明": page_help,
    }
    page_map.get(page, page_home)()


if __name__ == "__main__":
    main()
