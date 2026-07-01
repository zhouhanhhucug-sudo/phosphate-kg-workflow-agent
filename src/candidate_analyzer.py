from __future__ import annotations


FACTOR_WEIGHTS = {
    "赋矿层位": 20,
    "岩性组合": 15,
    "构造条件": 15,
    "沉积环境": 15,
    "地球化学指标": 15,
    "资源属性": 10,
    "找矿标志": 10,
}


def analyze_candidate_area(candidate_area: str, factors: dict) -> dict:
    """根据输入的成矿要素生成候选区解释。"""
    score = 0
    details = []
    for factor, weight in FACTOR_WEIGHTS.items():
        value = str(factors.get(factor, "未提供")).strip()
        status = "缺少证据"
        factor_score = 0
        if value and value not in {"未提供", "无", "不明确"}:
            if any(word in value for word in ["有利", "发育", "明显", "较高", "匹配", "具备", "含磷"]):
                factor_score = weight
                status = "有利"
            elif any(word in value for word in ["一般", "局部", "弱", "不稳定"]):
                factor_score = round(weight * 0.5, 1)
                status = "中等"
            elif any(word in value for word in ["不利", "缺失", "低", "差"]):
                factor_score = 0
                status = "不利"
            else:
                factor_score = round(weight * 0.4, 1)
                status = "需确认"
        score += factor_score
        details.append({"factor": factor, "evidence": value, "status": status, "score": factor_score, "weight": weight})

    if score >= 75:
        level = "高潜力"
    elif score >= 50:
        level = "中等潜力"
    elif score >= 30:
        level = "低-中潜力"
    else:
        level = "低潜力或证据不足"

    return {
        "candidate_area": candidate_area,
        "score": round(score, 1),
        "level": level,
        "details": details,
        "need_review": any(item["status"] in {"需确认", "缺少证据"} for item in details),
    }


def build_candidate_markdown(candidate_result: dict) -> str:
    """生成候选区 Markdown 解释文本。"""
    area = candidate_result.get("candidate_area", "未命名候选区")
    lines = [
        f"# {area} 成矿要素解释",
        "",
        f"- 综合评分：{candidate_result.get('score', 0)} / 100",
        f"- 潜力等级：{candidate_result.get('level', '未评价')}",
        f"- 是否需要人工复核：{'是' if candidate_result.get('need_review') else '否'}",
        "",
        "## 要素证据",
    ]
    for item in candidate_result.get("details", []):
        lines.append(
            f"- {item['factor']}：{item['status']}，得分 {item['score']}/{item['weight']}。证据：{item['evidence'] or '未提供'}"
        )
    lines.extend(
        [
            "",
            "## 解释结论",
            "该结果基于当前输入要素的规则化评分生成，仅用于候选区解释和汇报展示。所有缺少原文证据、空间数据或人工确认的条目均不应作为最终找矿结论。",
        ]
    )
    return "\n".join(lines)

