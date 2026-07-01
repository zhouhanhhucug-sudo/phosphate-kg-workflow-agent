from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .utils import evidence_sentence, split_sentences


ENTITY_COLUMNS = [
    "entity_id",
    "entity_name",
    "entity_type",
    "entity_type_cn",
    "entity_subtype",
    "source_text",
    "paper_id",
    "paragraph_id",
    "confidence",
    "need_review",
]

RELATION_COLUMNS = [
    "relation_id",
    "subject",
    "subject_type",
    "relation",
    "relation_cn",
    "object",
    "object_type",
    "evidence",
    "paper_id",
    "paragraph_id",
    "direction_check",
    "ontology_check",
    "confidence",
    "need_review",
]

ATTRIBUTE_COLUMNS = [
    "attribute_id",
    "entity_name",
    "attribute_name",
    "attribute_value",
    "unit",
    "evidence",
    "paper_id",
    "paragraph_id",
    "confidence",
    "need_review",
]

RULE_COLUMNS = [
    "rule_id",
    "rule_text",
    "related_entities",
    "evidence",
    "rule_type",
    "confidence",
    "need_review",
]


ENTITY_TYPE_CN = {
    "Deposit": "矿床",
    "Orebody": "矿体",
    "StratigraphicUnit": "地层单元",
    "OreBearingHorizon": "含矿层位",
    "Lithology": "岩性",
    "TectonicSetting": "构造背景",
    "SedimentaryEnvironment": "沉积环境",
    "GeochemicalIndicator": "地球化学指标",
    "ResourceAttribute": "资源属性",
    "ProspectingIndicator": "找矿标志",
    "MetallogenicAge": "成矿时代",
    "MetallogenicBelt": "成矿区带",
    "ExplorationEngineering": "勘查工程",
    "Region": "地区",
    "Mineral": "矿物",
    "Element": "元素",
}

RELATION_CN = {
    "located_in": "位于/分布于",
    "contains": "包含",
    "hosted_by": "赋存于",
    "controlled_by": "受控于",
    "associated_with": "相关",
    "formed_in": "形成于",
    "developed_in": "发育于",
    "has_lithology": "具有岩性",
    "has_indicator": "具有找矿标志",
    "has_geochemical_feature": "具有地球化学特征",
    "has_resource": "具有资源属性",
    "belongs_to": "属于",
    "adjacent_to": "邻近",
    "overlies": "覆于",
    "underlies": "伏于",
    "indicates": "指示",
    "favorable_for": "有利于",
    "restricts": "制约",
    "derived_from": "来源于",
    "same_as": "同义实体",
}


@dataclass(frozen=True)
class EntityHit:
    name: str
    entity_type: str
    subtype: str = ""
    confidence: float = 0.82
    need_review: bool = False


LITHOLOGY_TERMS = [
    "白云质磷块岩",
    "含磷白云岩",
    "磷块岩",
    "白云岩",
    "碳酸盐岩",
    "灰岩",
    "页岩",
    "泥岩",
    "砂岩",
]

TECTONIC_TERMS = ["断裂构造", "断裂", "断层", "背斜", "褶皱", "构造带"]
SEDIMENTARY_TERMS = ["浅海台地沉积环境", "浅海台地", "碳酸盐台地", "潮坪", "陆棚", "盆地沉积环境"]
GEOCHEM_TERMS = ["P2O5 含量", "P2O5含量", "P₂O₅含量", "P₂O₅ 含量", "五氧化二磷含量", "稀土元素", "微量元素"]
RESOURCE_TERMS = ["矿层厚度", "资源量", "储量", "品位", "厚度", "矿石量", "资源潜力"]
PROSPECTING_TERMS = ["找矿标志", "控矿因素", "有利部位", "评价指标", "远景区", "磷质沉积"]
EXPLORATION_TERMS = ["钻孔", "槽探", "剖面", "样品", "勘查工程"]
MINERAL_TERMS = ["磷灰石", "胶磷矿", "白云石", "方解石"]


def _unique_hits(hits: Iterable[EntityHit]) -> list[EntityHit]:
    unique: OrderedDict[tuple[str, str], EntityHit] = OrderedDict()
    for hit in hits:
        name = hit.name.strip(" ，。；、()（）")
        if not name or len(name) < 2:
            continue
        key = (name, hit.entity_type)
        if key not in unique:
            unique[key] = EntityHit(name, hit.entity_type, hit.subtype, hit.confidence, hit.need_review)
    return list(unique.values())


def _regex_hits(text: str) -> list[EntityHit]:
    hits: list[EntityHit] = []

    for match in re.finditer(r"[\u4e00-\u9fffA-Za-z0-9]{2,18}(?:磷矿床|磷矿区|矿床)", text):
        name = match.group(0)
        if any(skip in name for skip in ["沉积型磷矿床", "磷矿床位"]):
            continue
        hits.append(EntityHit(name, "Deposit", confidence=0.88))

    for match in re.finditer(r"[\u4e00-\u9fffA-Za-z0-9]{1,10}(?:地区|矿集区|成矿带|远景区|盆地|研究区)", text):
        hits.append(EntityHit(match.group(0), "Region" if "成矿带" not in match.group(0) else "MetallogenicBelt", confidence=0.78))

    strat_patterns = [
        r"[\u4e00-\u9fffA-Za-z0-9]{1,8}系[\u4e00-\u9fffA-Za-z0-9]{1,8}组",
        r"[\u4e00-\u9fffA-Za-z0-9]{1,12}(?:组|段|系|统|阶)",
    ]
    for pattern in strat_patterns:
        for match in re.finditer(pattern, text):
            name = match.group(0)
            if name in {"岩系"}:
                continue
            hits.append(EntityHit(name, "StratigraphicUnit", confidence=0.80))

    if "矿体" in text:
        hits.append(EntityHit("矿体", "Orebody", "泛指", confidence=0.70, need_review=True))

    for term in ["含磷岩系", "含矿岩系", "含矿层位", "含矿层", "矿层"]:
        if term in text:
            hits.append(EntityHit(term, "OreBearingHorizon", confidence=0.78))

    for term in LITHOLOGY_TERMS:
        if term in text:
            hits.append(EntityHit(term, "Lithology", confidence=0.90))

    for term in TECTONIC_TERMS:
        if term in text:
            hits.append(EntityHit(term, "TectonicSetting", confidence=0.88))

    for term in SEDIMENTARY_TERMS:
        if term in text:
            entity_name = term if "环境" in term else f"{term}沉积环境"
            hits.append(EntityHit(entity_name, "SedimentaryEnvironment", confidence=0.86))

    for term in GEOCHEM_TERMS:
        if term in text:
            normalized = term.replace("P₂O₅", "P2O5").replace("P2O5含量", "P2O5 含量")
            hits.append(EntityHit(normalized, "GeochemicalIndicator", confidence=0.86))

    for term in RESOURCE_TERMS:
        if term in text:
            hits.append(EntityHit(term, "ResourceAttribute", confidence=0.78))

    for term in PROSPECTING_TERMS:
        if term in text:
            hits.append(EntityHit(term, "ProspectingIndicator", confidence=0.74, need_review=term in {"磷质沉积"}))

    for term in EXPLORATION_TERMS:
        if term in text:
            hits.append(EntityHit(term, "ExplorationEngineering", confidence=0.82))

    for term in MINERAL_TERMS:
        if term in text:
            hits.append(EntityHit(term, "Mineral", confidence=0.86))

    for match in re.finditer(r"(?:寒武纪|震旦纪|二叠纪|奥陶纪|志留纪|泥盆纪|石炭纪|三叠纪|侏罗纪|白垩纪)", text):
        hits.append(EntityHit(match.group(0), "MetallogenicAge", confidence=0.70, need_review=True))

    return _unique_hits(hits)


def extract_entities(text: str, paper_id: str, paragraph_id: str) -> pd.DataFrame:
    """抽取实体。第一版使用关键词和正则规则。"""
    rows = []
    for idx, hit in enumerate(_regex_hits(text), start=1):
        rows.append(
            {
                "entity_id": f"E-{paragraph_id}-{idx:03d}",
                "entity_name": hit.name,
                "entity_type": hit.entity_type,
                "entity_type_cn": ENTITY_TYPE_CN.get(hit.entity_type, "未映射"),
                "entity_subtype": hit.subtype,
                "source_text": evidence_sentence(text, hit.name.replace("沉积环境", "")),
                "paper_id": paper_id,
                "paragraph_id": paragraph_id,
                "confidence": round(hit.confidence, 2),
                "need_review": hit.need_review,
            }
        )
    return pd.DataFrame(rows, columns=ENTITY_COLUMNS)


def _entities_by_type(entities: pd.DataFrame, entity_type: str) -> list[str]:
    if entities.empty or "entity_type" not in entities:
        return []
    return entities.loc[entities["entity_type"] == entity_type, "entity_name"].dropna().astype(str).unique().tolist()


def _first_entity(entities: pd.DataFrame, entity_type: str) -> str | None:
    values = _entities_by_type(entities, entity_type)
    return values[0] if values else None


def _add_relation(
    rows: list[dict],
    subject: str | None,
    subject_type: str,
    relation: str,
    obj: str | None,
    object_type: str,
    evidence: str,
    paper_id: str,
    paragraph_id: str,
    confidence: float = 0.78,
    need_review: bool = False,
) -> None:
    if not subject or not obj or subject == obj:
        return
    key = (subject, relation, obj)
    if any((row["subject"], row["relation"], row["object"]) == key for row in rows):
        return
    rows.append(
        {
            "relation_id": f"R-{paragraph_id}-{len(rows) + 1:03d}",
            "subject": subject,
            "subject_type": subject_type,
            "relation": relation,
            "relation_cn": RELATION_CN.get(relation, relation),
            "object": obj,
            "object_type": object_type,
            "evidence": evidence,
            "paper_id": paper_id,
            "paragraph_id": paragraph_id,
            "direction_check": "通过" if not need_review else "需确认",
            "ontology_check": "待校验",
            "confidence": round(confidence, 2),
            "need_review": need_review,
        }
    )


def extract_relations(text: str, entities: pd.DataFrame, paper_id: str, paragraph_id: str) -> pd.DataFrame:
    """抽取关系。第一版使用规则模板。"""
    rows: list[dict] = []
    deposits = _entities_by_type(entities, "Deposit")
    core = deposits[0] if deposits else _first_entity(entities, "Region")

    for sentence in split_sentences(text):
        sentence_entities = entities[entities["entity_name"].apply(lambda name: str(name) in sentence)] if not entities.empty else pd.DataFrame()
        sentence_core = core
        sentence_deposits = _entities_by_type(sentence_entities, "Deposit")
        if sentence_deposits:
            sentence_core = sentence_deposits[0]

        if any(word in sentence for word in ["位于", "分布于", "地处"]):
            for region in _entities_by_type(sentence_entities, "Region"):
                _add_relation(rows, sentence_core, "Deposit", "located_in", region, "Region", sentence, paper_id, paragraph_id, 0.86)

        if any(word in sentence for word in ["赋存于", "产于", "赋存", "容矿", "含矿层位"]):
            for obj_type in ["StratigraphicUnit", "OreBearingHorizon"]:
                for obj in _entities_by_type(sentence_entities, obj_type):
                    _add_relation(rows, sentence_core, "Deposit", "hosted_by", obj, obj_type, sentence, paper_id, paragraph_id, 0.84)

        if any(word in sentence for word in ["岩性", "主要为", "由", "组成"]):
            for lithology in _entities_by_type(sentence_entities, "Lithology"):
                _add_relation(rows, sentence_core, "Deposit", "has_lithology", lithology, "Lithology", sentence, paper_id, paragraph_id, 0.86)

        if any(word in sentence for word in ["控制", "影响", "受"]):
            for tectonic in _entities_by_type(sentence_entities, "TectonicSetting"):
                _add_relation(rows, sentence_core, "Deposit", "controlled_by", tectonic, "TectonicSetting", sentence, paper_id, paragraph_id, 0.82)

        if any(word in sentence for word in ["发育"]):
            for tectonic in _entities_by_type(sentence_entities, "TectonicSetting"):
                _add_relation(rows, sentence_core, "Deposit", "developed_in", tectonic, "TectonicSetting", sentence, paper_id, paragraph_id, 0.68, True)

        if any(word in sentence for word in ["重要指标", "评价", "指示"]):
            indicators = _entities_by_type(sentence_entities, "GeochemicalIndicator") + _entities_by_type(sentence_entities, "ResourceAttribute")
            targets = _entities_by_type(sentence_entities, "ProspectingIndicator") or ["资源潜力"]
            for indicator in indicators:
                for target in targets:
                    _add_relation(rows, indicator, "GeochemicalIndicator", "indicates", target, "ProspectingIndicator", sentence, paper_id, paragraph_id, 0.80)

        if any(word in sentence for word in ["有利于", "有利"]):
            for environment in _entities_by_type(sentence_entities, "SedimentaryEnvironment"):
                targets = _entities_by_type(sentence_entities, "ProspectingIndicator") or ["磷质沉积"]
                for target in targets:
                    _add_relation(rows, environment, "SedimentaryEnvironment", "favorable_for", target, "ProspectingIndicator", sentence, paper_id, paragraph_id, 0.84)

        for geochem in _entities_by_type(sentence_entities, "GeochemicalIndicator"):
            _add_relation(rows, sentence_core, "Deposit", "has_geochemical_feature", geochem, "GeochemicalIndicator", sentence, paper_id, paragraph_id, 0.66, True)

        for resource in _entities_by_type(sentence_entities, "ResourceAttribute"):
            if resource != "资源潜力":
                _add_relation(rows, sentence_core, "Deposit", "has_resource", resource, "ResourceAttribute", sentence, paper_id, paragraph_id, 0.66, True)

    return pd.DataFrame(rows, columns=RELATION_COLUMNS)


def extract_attributes(text: str, entities: pd.DataFrame, paper_id: str, paragraph_id: str) -> pd.DataFrame:
    """抽取属性，例如品位、厚度、资源量、P2O5 含量等。"""
    rows = []
    core_entity = _first_entity(entities, "Deposit") or _first_entity(entities, "Region") or "未指定对象"
    patterns = [
        ("P2O5 含量", r"(?:P2O5|P₂O₅|五氧化二磷)\s*含量(?:为|达|约|平均为)?\s*([0-9.]+(?:\s*[-~至]\s*[0-9.]+)?\s*%?)", "%"),
        ("品位", r"品位(?:为|达|约|平均为)?\s*([0-9.]+(?:\s*[-~至]\s*[0-9.]+)?\s*%?)", "%"),
        ("矿层厚度", r"(?:矿层厚度|厚度)(?:为|达|约|平均为)?\s*([0-9.]+(?:\s*[-~至]\s*[0-9.]+)?\s*(?:m|米)?)", "m"),
        ("资源量", r"(?:资源量|储量)(?:为|达|约)?\s*([0-9.]+(?:\s*[-~至]\s*[0-9.]+)?\s*(?:万吨|亿吨|t)?)", ""),
    ]
    for attr_name, pattern, default_unit in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(1).strip()
            rows.append(
                {
                    "attribute_id": f"A-{paragraph_id}-{len(rows) + 1:03d}",
                    "entity_name": core_entity,
                    "attribute_name": attr_name,
                    "attribute_value": value,
                    "unit": default_unit if not re.search(r"[%米吨m]", value) else "",
                    "evidence": evidence_sentence(text, match.group(0)),
                    "paper_id": paper_id,
                    "paragraph_id": paragraph_id,
                    "confidence": 0.86,
                    "need_review": False,
                }
            )

    for indicator in ["P2O5 含量", "矿层厚度", "品位", "资源量"]:
        if indicator.replace(" ", "") in text.replace(" ", "") and not any(row["attribute_name"] == indicator for row in rows):
            rows.append(
                {
                    "attribute_id": f"A-{paragraph_id}-{len(rows) + 1:03d}",
                    "entity_name": core_entity,
                    "attribute_name": indicator,
                    "attribute_value": "文中提及，未给出具体数值",
                    "unit": "",
                    "evidence": evidence_sentence(text, indicator.split()[0]),
                    "paper_id": paper_id,
                    "paragraph_id": paragraph_id,
                    "confidence": 0.62,
                    "need_review": True,
                }
            )

    return pd.DataFrame(rows, columns=ATTRIBUTE_COLUMNS)


def extract_rules(text: str, entities: pd.DataFrame, paper_id: str, paragraph_id: str) -> pd.DataFrame:
    """抽取找矿规则。"""
    rows = []
    rule_keywords = ["有利于", "控制", "受", "指示", "重要指标", "评价", "找矿标志", "控矿"]
    entity_names = entities["entity_name"].dropna().astype(str).tolist() if not entities.empty else []
    for sentence in split_sentences(text):
        if not any(keyword in sentence for keyword in rule_keywords):
            continue
        related = [name for name in entity_names if name in sentence]
        rule_type = "控矿规则" if any(word in sentence for word in ["控制", "受控", "受"]) else "找矿评价规则"
        rows.append(
            {
                "rule_id": f"RULE-{paragraph_id}-{len(rows) + 1:03d}",
                "rule_text": sentence,
                "related_entities": "；".join(related),
                "evidence": sentence,
                "rule_type": rule_type,
                "confidence": 0.78 if related else 0.62,
                "need_review": not bool(related),
            }
        )
    return pd.DataFrame(rows, columns=RULE_COLUMNS)


def _add_missing_relation_objects(entities: pd.DataFrame, relations: pd.DataFrame) -> pd.DataFrame:
    if relations.empty:
        return entities
    existing = set(entities["entity_name"].astype(str).tolist()) if not entities.empty else set()
    rows = entities.to_dict("records") if not entities.empty else []
    for _, relation in relations.iterrows():
        obj = str(relation.get("object", ""))
        obj_type = str(relation.get("object_type", "ProspectingIndicator"))
        if obj and obj not in existing:
            rows.append(
                {
                    "entity_id": f"E-AUTO-{len(rows) + 1:03d}",
                    "entity_name": obj,
                    "entity_type": obj_type,
                    "entity_type_cn": ENTITY_TYPE_CN.get(obj_type, "未映射"),
                    "entity_subtype": "关系补全",
                    "source_text": relation.get("evidence", ""),
                    "paper_id": relation.get("paper_id", ""),
                    "paragraph_id": relation.get("paragraph_id", ""),
                    "confidence": 0.60,
                    "need_review": True,
                }
            )
            existing.add(obj)
    return pd.DataFrame(rows, columns=ENTITY_COLUMNS)


def _deduplicate_entities(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=ENTITY_COLUMNS)
    df = df.sort_values(["entity_name", "confidence"], ascending=[True, False])
    df = df.drop_duplicates(subset=["entity_name", "entity_type"], keep="first").copy()
    df["entity_id"] = [f"E-{idx:04d}" for idx in range(1, len(df) + 1)]
    return df[ENTITY_COLUMNS].reset_index(drop=True)


def _deduplicate_relations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=RELATION_COLUMNS)
    df = df.drop_duplicates(subset=["subject", "relation", "object", "evidence"], keep="first").copy()
    df["relation_id"] = [f"R-{idx:04d}" for idx in range(1, len(df) + 1)]
    return df[RELATION_COLUMNS].reset_index(drop=True)


def run_extraction(corpus_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """对语料表批量执行知识抽取。"""
    entity_frames = []
    relation_frames = []
    attribute_frames = []
    rule_frames = []

    if corpus_df is None or corpus_df.empty:
        return {
            "entities": pd.DataFrame(columns=ENTITY_COLUMNS),
            "relations": pd.DataFrame(columns=RELATION_COLUMNS),
            "attributes": pd.DataFrame(columns=ATTRIBUTE_COLUMNS),
            "rules": pd.DataFrame(columns=RULE_COLUMNS),
        }

    keep_df = corpus_df[corpus_df.get("keep_or_drop", "keep").astype(str).str.lower().ne("drop")]
    for _, row in keep_df.iterrows():
        text = str(row.get("text_clean", ""))
        paper_id = str(row.get("paper_id", ""))
        paragraph_id = str(row.get("paragraph_id", "P-000"))
        entities = extract_entities(text, paper_id, paragraph_id)
        relations = extract_relations(text, entities, paper_id, paragraph_id)
        attributes = extract_attributes(text, entities, paper_id, paragraph_id)
        rules = extract_rules(text, entities, paper_id, paragraph_id)
        entity_frames.append(entities)
        relation_frames.append(relations)
        attribute_frames.append(attributes)
        rule_frames.append(rules)

    entities_df = pd.concat(entity_frames, ignore_index=True) if entity_frames else pd.DataFrame(columns=ENTITY_COLUMNS)
    relations_df = pd.concat(relation_frames, ignore_index=True) if relation_frames else pd.DataFrame(columns=RELATION_COLUMNS)
    attributes_df = pd.concat(attribute_frames, ignore_index=True) if attribute_frames else pd.DataFrame(columns=ATTRIBUTE_COLUMNS)
    rules_df = pd.concat(rule_frames, ignore_index=True) if rule_frames else pd.DataFrame(columns=RULE_COLUMNS)

    entities_df = _add_missing_relation_objects(entities_df, relations_df)
    return {
        "entities": _deduplicate_entities(entities_df),
        "relations": _deduplicate_relations(relations_df),
        "attributes": attributes_df.reset_index(drop=True),
        "rules": rules_df.reset_index(drop=True),
    }

