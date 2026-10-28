from __future__ import annotations

import io
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="계절가전 리뷰 분석", page_icon="📊", layout="wide")

ALIASES = {
    "review_id": ["review_id", "reviewid", "Index", "index", "ReviewID", "리뷰ID"],
    "aspect_category": ["aspect_category", "aspect", "대분류", "속성"],
    "sub_reason": ["sub_reason", "subreason", "Aspect", "세부분류", "세부원인"],
    "sentiment": ["sentiment", "SentimentPolarity", "감성", "polarity"],
    "evidence_text": ["evidence_text", "SentimentText", "evidence", "근거문장", "근거"],
    "RawText_clean": [
        "RawText_clean",
        "raw_text",
        "review_text",
        "text",
        "리뷰",
        "원문",
        "내용",
    ],
    "product": ["product", "product_name", "ProductName", "제품", "상품명"],
    "brand_normalized": ["brand_normalized", "brand", "브랜드", "제조사"],
    "ReviewScore": ["ReviewScore", "review_score", "rating", "별점", "평점"],
    "RDate": ["RDate", "review_date", "date", "작성일", "날짜"],
}
SUPPORTED = ["csv", "parquet", "pq", "xlsx", "xls", "json", "jsonl"]


def read_table(name: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        for enc in ("utf-8-sig", "utf-8", "cp949"):
            try:
                return pd.read_csv(io.BytesIO(payload), encoding=enc)
            except UnicodeDecodeError:
                pass
        raise ValueError(f"CSV 인코딩을 읽지 못했습니다: {name}")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(io.BytesIO(payload))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(payload))
    if suffix == ".json":
        obj = json.loads(payload.decode("utf-8"))
        if isinstance(obj, dict) and any(
            isinstance(v, (dict, list)) for v in obj.values()
        ):
            return pd.DataFrame({"__json_payload__": [obj]})
        if isinstance(obj, dict):
            return pd.DataFrame([obj])
        return pd.DataFrame({"__json_payload__": [obj]})
    if suffix == ".jsonl":
        return pd.read_json(io.BytesIO(payload), lines=True)
    raise ValueError(f"지원하지 않는 파일: {name}")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lookup = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for canonical, choices in ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in choices:
            original = lookup.get(alias.lower())
            if original is not None:
                rename[original] = canonical
                break
    df = df.rename(columns=rename)
    if "review_id" in df:
        df["review_id"] = df["review_id"].astype("string").str.strip()
    for col in [
        "aspect_category",
        "sub_reason",
        "sentiment",
        "evidence_text",
        "RawText_clean",
        "product",
        "brand_normalized",
    ]:
        if col in df:
            df[col] = df[col].fillna("").astype(str).str.strip()
    if "sentiment" in df:
        df["sentiment"] = df["sentiment"].replace(
            {
                "positive": "긍정",
                "negative": "부정",
                "neutral": "중립",
                "pos": "긍정",
                "neg": "부정",
                "neu": "중립",
                "Positive": "긍정",
                "Negative": "부정",
                "Neutral": "중립",
            }
        )
    return df


def read_folder(folder: str) -> list[tuple[str, pd.DataFrame]]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {root}")
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower().lstrip(".") in SUPPORTED
        and not p.name.startswith("~$")
    ]
    if not files:
        raise FileNotFoundError("폴더에서 지원하는 데이터 파일을 찾지 못했습니다.")
    tables, errors = [], []
    for p in files:
        try:
            tables.append(
                (p.relative_to(root).as_posix(), read_table(p.name, p.read_bytes()))
            )
        except Exception as e:
            errors.append(f"{p.name}: {e}")
    if not tables:
        raise ValueError("파일을 읽지 못했습니다. " + " / ".join(errors))
    return tables


def top_terms(texts: pd.Series, limit: int = 12) -> pd.DataFrame:
    stop = set(
        "정말 너무 진짜 그냥 조금 생각보다 제품 사용 구매 주문 배송 리뷰 고객 만족 좋아요 좋습니다 합니다 했어요 입니다 그리고 그런데 있는 같아요 사용중 사용해요".split()
    )
    counts: Counter[str] = Counter()
    for text in texts.fillna("").astype(str):
        words = re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
        counts.update(w for w in words if w not in stop and not w.isdigit())
    return pd.DataFrame(counts.most_common(limit), columns=["키워드", "언급 수"])


def natural_key(value: str):
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def summary_payload(frame: pd.DataFrame) -> dict:
    if "__json_payload__" in frame.columns and len(frame):
        value = frame.iloc[0]["__json_payload__"]
        return value if isinstance(value, dict) else {}
    if len(frame):
        return frame.iloc[0].to_dict()
    return {}


def class_metrics_from_pairs(pred: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    return class_metrics_for_level(pred, gold, "sub_reason")


def class_metrics_for_level(
    pred: pd.DataFrame, gold: pd.DataFrame, level: str
) -> pd.DataFrame:
    keys = ["review_id", level]
    if not set(keys).issubset(pred.columns) or not set(keys).issubset(gold.columns):
        return pd.DataFrame()
    p = pred[keys].drop_duplicates()
    g = gold[keys].drop_duplicates()
    pset = set(map(tuple, p.itertuples(index=False, name=None)))
    gset = set(map(tuple, g.itertuples(index=False, name=None)))
    rows = []
    for reason in sorted({s for _, s in pset | gset}, key=natural_key):
        pk = {x for x in pset if x[1] == reason}
        gk = {x for x in gset if x[1] == reason}
        tp, fp, fn = len(pk & gk), len(pk - gk), len(gk - pk)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        rows.append(
            {
                level: reason,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def summary_row(
    experiment: str, summary: dict, metrics: pd.DataFrame | None = None
) -> dict:
    sub = summary.get("sub_reason", summary.get("subreason", {}))
    category = summary.get("aspect_category", summary.get("category", {}))
    if not isinstance(sub, dict):
        sub = {}
    if not isinstance(category, dict):
        category = {}
    if not sub and metrics is not None:
        m = normalize_metric_columns(metrics)
        if {"TP", "FP", "FN"}.issubset(m.columns):
            tp, fp, fn = (
                int(pd.to_numeric(m[c], errors="coerce").fillna(0).sum())
                for c in ["TP", "FP", "FN"]
            )
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            sub = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                ),
            }
    return {
        "experiment": experiment,
        "sub_tp": sub.get("tp", sub.get("TP")),
        "sub_fp": sub.get("fp", sub.get("FP")),
        "sub_fn": sub.get("fn", sub.get("FN")),
        "sub_precision": sub.get("precision", sub.get("Precision")),
        "sub_recall": sub.get("recall", sub.get("Recall")),
        "sub_f1": sub.get("f1", sub.get("F1")),
        "category_precision": category.get("precision", category.get("Precision")),
        "category_recall": category.get("recall", category.get("Recall")),
        "category_f1": category.get("f1", category.get("F1")),
        "sentiment_accuracy": summary.get("sentiment_accuracy"),
    }


def get_experiment_tables(tables):
    normalized = [(name.replace("\\", "/"), normalize(df)) for name, df in tables]
    predictions, prediction_priority, metrics, summaries, gold_by_experiment = (
        {},
        {},
        {},
        {},
        {},
    )
    comparison = None
    metadata = None
    common_gold = None
    safety_summary = None
    safety_examples = None

    def experiment_from_path(name: str):
        parts = Path(name).parts
        lowered = [part.lower() for part in parts]
        if "output" in lowered:
            index = lowered.index("output")
            if index + 1 < len(parts):
                return parts[index + 1]
        return None

    for name, frame in normalized:
        base = Path(name).name.lower()
        low = name.lower()
        exp_name = experiment_from_path(name)
        if exp_name:
            if "metrics_subreason" in base or "metrics_sub_reason" in base:
                metrics[exp_name] = frame
            elif "summary" in base and base.endswith(".json"):
                summaries[exp_name] = summary_payload(frame)
            elif "gold_aspect" in base:
                gold_by_experiment[exp_name] = frame
            elif "safety_tag_summary" in base:
                safety_summary = frame
            elif "flagged_safety" in base:
                safety_examples = frame
            elif {"review_id", "sub_reason"}.issubset(frame.columns):
                # Prefer a clearly named predictions file; accept legacy W4 and W6 names.
                priority = (
                    0
                    if base in {"predictions.parquet", "model_predictions.parquet"}
                    else (1 if "w6_experiment" in base else 2)
                )
                if (
                    exp_name not in predictions
                    or priority < prediction_priority[exp_name]
                ):
                    predictions[exp_name] = frame
                    prediction_priority[exp_name] = priority
            continue

        if "comparison_summary" in base:
            comparison = frame
        elif "metrics_subreason_ex1" in base:
            metrics["실험 1"] = frame
        elif "metrics_subreason_ex2" in base:
            metrics["실험 2"] = frame
        elif "metrics_subreason_ex3" in base:
            metrics["실험 3"] = frame
        elif "safety_tag_summary" in base:
            safety_summary = frame
        elif "flagged_safety" in base:
            safety_examples = frame
        elif "w5_gold_aspects" in base or "gold_aspects" in base:
            common_gold = frame
        elif "w6_experiment" in base and {"review_id", "sub_reason"}.issubset(
            frame.columns
        ):
            match = re.search(r"experiment[_-]?(\d+)", base)
            label = f"실험 {match.group(1)}" if match else Path(base).stem
            predictions[label] = frame
        elif "metrics_subreason" in base:
            match = re.search(r"ex[_-]?(\d+)", base)
            label = f"실험 {match.group(1)}" if match else Path(base).stem
            metrics[label] = frame
        elif {"review_id", "RawText_clean"}.issubset(frame.columns):
            metadata = frame

    # W5 gold is fixed across prompt experiments; allow one shared copy under common/.
    if common_gold is None:
        common_gold = next(
            (
                frame
                for name, frame in normalized
                if "w5_gold_aspects" in Path(name).name.lower()
                and experiment_from_path(name) is None
            ),
            None,
        )
    for experiment, pred in predictions.items():
        gold = gold_by_experiment.get(experiment, common_gold)
        if gold is not None and (
            experiment not in metrics or "sub_reason" not in metrics[experiment].columns
        ):
            metrics[experiment] = class_metrics_from_pairs(pred, gold)
    for experiment, metric in metrics.items():
        if "sub_reason" not in metric.columns:
            # Restore a Parquet-saved index when the label column was serialized as the index.
            metric = metric.reset_index()
            first = metric.columns[0]
            if (
                first.lower() in {"index", "level_0"}
                and metric[first].astype(str).str.contains("[가-힣]").any()
            ):
                metric = metric.rename(columns={first: "sub_reason"})
            metrics[experiment] = metric

    order = sorted(set(predictions) | set(metrics) | set(summaries), key=natural_key)
    predictions = {name: predictions[name] for name in order if name in predictions}
    metrics = {name: metrics[name] for name in order if name in metrics}
    gold_by_experiment = {
        name: gold_by_experiment[name] for name in order if name in gold_by_experiment
    }

    summary_rows = [
        summary_row(name, summaries.get(name, {}), metrics.get(name)) for name in order
    ]
    summary_rows = [
        row
        for row in summary_rows
        if any(
            row.get(c) is not None for c in ["sub_f1", "sub_precision", "sub_recall"]
        )
    ]
    if summary_rows:
        comparison = pd.DataFrame(summary_rows)
    elif comparison is not None:
        comparison.columns = [str(c).strip() for c in comparison.columns]
        exp_col = next(
            (c for c in ["experiment", "실험"] if c in comparison.columns), None
        )
        if exp_col:

            def legacy_label(value):
                match = re.search(
                    r"(?:experiment|exp|실험)[ _-]?(\d+)", str(value), re.I
                )
                return f"실험 {match.group(1)}" if match else str(value)

            comparison[exp_col] = comparison[exp_col].map(legacy_label)
            comparison = comparison.sort_values(
                exp_col, key=lambda col: col.map(natural_key), kind="stable"
            )

    return (
        normalized,
        predictions,
        metrics,
        comparison,
        metadata,
        safety_summary,
        safety_examples,
        common_gold,
        gold_by_experiment,
        order,
    )


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "sub_reason": "세부 원인",
        "tp": "TP",
        "fp": "FP",
        "fn": "FN",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }
    df = df.rename(
        columns={c: aliases[c.lower()] for c in df.columns if c.lower() in aliases}
    )
    return df


def merge_metadata(pred: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.DataFrame:
    out = pred.copy()
    if metadata is not None and "review_id" in metadata:
        cols = [
            c
            for c in [
                "review_id",
                "RawText_clean",
                "product",
                "brand_normalized",
                "ReviewScore",
                "RDate",
            ]
            if c in metadata
        ]
        meta = metadata[cols].drop_duplicates("review_id")
        out = out.merge(meta, on="review_id", how="left", suffixes=("", "_meta"))
    for c in [
        "RawText_clean",
        "product",
        "brand_normalized",
        "evidence_text",
        "aspect_category",
        "sentiment",
    ]:
        if c not in out:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)
    return out


def make_review_cases(pred: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Build one drill-down row for every TP, FP, and FN aspect pair."""
    key = ["review_id", "sub_reason"]
    p = pred.drop_duplicates(key).copy()
    g = gold.drop_duplicates(key).copy()
    for frame in [p, g]:
        for col in ["aspect_category", "sub_reason", "sentiment", "evidence_text"]:
            if col not in frame:
                frame[col] = ""
            frame[col] = frame[col].fillna("").astype(str)
    pred_by_review = {rid: group for rid, group in p.groupby("review_id", sort=False)}
    gold_by_review = {rid: group for rid, group in g.groupby("review_id", sort=False)}
    gold_keys = set(zip(g["review_id"], g["sub_reason"]))
    pred_keys = set(zip(p["review_id"], p["sub_reason"]))
    rows = []
    for _, row in p.iterrows():
        rid, sub = row["review_id"], row["sub_reason"]
        status = "TP" if (rid, sub) in gold_keys else "FP"
        gg = gold_by_review.get(rid, pd.DataFrame())
        gold_labels = (
            "; ".join(
                f"{r.aspect_category + ' / ' if r.aspect_category else ''}{r.sub_reason} ({r.sentiment})"
                for r in gg.itertuples()
            )
            or "정답 aspect 없음"
        )
        gold_evidence = (
            "\n".join(
                f"{r.sub_reason}: {r.evidence_text}"
                for r in gg.itertuples()
                if r.evidence_text
            )
            or ""
        )
        sentiment_gold = (
            gg.loc[gg["sub_reason"] == sub, "sentiment"].iloc[0]
            if status == "TP" and not gg.loc[gg["sub_reason"] == sub].empty
            else ""
        )
        rows.append(
            {
                "review_id": rid,
                "status": status,
                "filter_category": row.aspect_category,
                "filter_sub_reason": sub,
                "filter_sentiment": row.sentiment,
                "pred_category": row.aspect_category,
                "pred_sub_reason": sub,
                "pred_sentiment": row.sentiment,
                "pred_evidence": row.evidence_text,
                "gold_category": row.aspect_category if status == "TP" else "",
                "gold_sub_reason": sub if status == "TP" else "",
                "gold_sentiment": sentiment_gold,
                "gold_evidence": (
                    next(
                        iter(gg.loc[gg["sub_reason"] == sub, "evidence_text"].tolist()),
                        "",
                    )
                    if status == "TP"
                    else gold_evidence
                ),
                "other_gold": gold_labels if status == "FP" else "",
                "other_pred": "",
                "why": (
                    (
                        "정답"
                        if row.sentiment == sentiment_gold
                        else "Aspect는 맞았지만 감성 예측이 Gold와 다릅니다."
                    )
                    if status == "TP"
                    else f"과잉 예측(모델이 '{sub}'를 예측했지만 Gold에 같은 aspect가 없습니다.)"
                ),
            }
        )
    for _, row in g.iterrows():
        rid, sub = row["review_id"], row["sub_reason"]
        if (rid, sub) in pred_keys:
            continue
        pp = pred_by_review.get(rid, pd.DataFrame())
        pred_labels = (
            "; ".join(
                f"{r.aspect_category + ' / ' if r.aspect_category else ''}{r.sub_reason} ({r.sentiment})"
                for r in pp.itertuples()
            )
            or "모델이 aspect를 예측하지 않음"
        )
        rows.append(
            {
                "review_id": rid,
                "status": "FN",
                "filter_category": row.aspect_category,
                "filter_sub_reason": sub,
                "filter_sentiment": row.sentiment,
                "pred_category": "",
                "pred_sub_reason": "",
                "pred_sentiment": "",
                "pred_evidence": "",
                "gold_category": row.aspect_category,
                "gold_sub_reason": sub,
                "gold_sentiment": row.sentiment,
                "gold_evidence": row.evidence_text,
                "other_gold": "",
                "other_pred": pred_labels,
                "why": f"Gold에는 '{sub}'가 있지만 모델 예측에서 누락됐습니다.",
            }
        )
    out = pd.DataFrame(rows)
    return out


def render_developer_drilldown(
    pred: pd.DataFrame,
    gold: pd.DataFrame | None,
    metadata: pd.DataFrame | None,
    experiment: str,
):
    st.subheader("개발자용 리뷰 Drill-down · TP / FP / FN")
    st.caption(
        "평가 단위를 Aspect_category(대분류) 또는 Sub_reason(소분류)로 선택해 TP·FP·FN 사례의 리뷰 원문, 모델 예측, Gold 정답과 근거 문구를 확인합니다."
    )
    if gold is None:
        st.info("개발자용 정오답 Drill-down을 보려면 Gold 데이터가 필요합니다.")
        return
    gold = normalize(gold)
    evaluation_level = st.radio(
        "평가 단위",
        ["Sub_reason(소분류)", "Aspect_category(대분류)"],
        horizontal=True,
        key="dev_case_level",
    )
    st.caption(
        "Sub_reason(소분류)는 세부 항목 단위의 TP·FP·FN을 확인합니다. "
        "Aspect_category(대분류)는 소분류를 상위 범주로 묶어 판정하므로, "
        "세부 항목은 달라도 대분류가 맞았는지 살펴볼 때 대분류를 선택하세요."
    )
    pred_eval, gold_eval = pred.copy(), gold.copy()
    if evaluation_level == "Aspect_category(대분류)":
        pred_eval["sub_reason"] = pred_eval["aspect_category"]
        gold_eval["sub_reason"] = gold_eval["aspect_category"]
    cases = make_review_cases(pred_eval, gold_eval)
    display_metadata = merge_metadata(pred, metadata)
    if "review_id" in display_metadata:
        meta_cols = [
            c
            for c in ["review_id", "RawText_clean", "product"]
            if c in display_metadata
        ]
        cases = cases.merge(
            display_metadata[meta_cols].drop_duplicates("review_id"),
            on="review_id",
            how="left",
        )
    for col in ["RawText_clean", "product"]:
        if col not in cases:
            cases[col] = ""
        cases[col] = cases[col].fillna("").astype(str)

    c1, c2, c3, c4, c5 = st.columns([1.1, 1.2, 1.25, 1.1, 1.3])
    with c1:
        products = sorted(x for x in cases["product"].unique() if x.strip())
        product_filter = st.selectbox(
            "제품", ["전체"] + products, key="dev_case_product"
        )
    with c2:
        categories = sorted(x for x in cases["filter_category"].unique() if x.strip())
        category_filter = st.selectbox(
            "Aspect_category(대분류)", ["전체"] + categories, key="dev_case_category"
        )
    with c3:
        reasons = (
            sorted(x for x in cases["filter_sub_reason"].unique() if x.strip())
            if evaluation_level == "Sub_reason(소분류)"
            else []
        )
        reason_filter = st.selectbox(
            "Sub_reason(소분류)",
            ["전체"] + reasons,
            key="dev_case_reason",
            disabled=evaluation_level != "Sub_reason(소분류)",
        )
    with c4:
        sentiments = [
            x for x in ["긍정", "부정", "중립"] if x in set(cases["filter_sentiment"])
        ]
        sentiment_filter = st.selectbox(
            "감성", ["전체"] + sentiments, key="dev_case_sentiment"
        )
    with c5:
        statuses = st.multiselect(
            "평가 결과",
            ["TP · 정답", "FP · 과잉", "FN · 누락"],
            default=["TP · 정답", "FP · 과잉", "FN · 누락"],
            key="dev_case_status",
        )

    status_map = {"TP · 정답": "TP", "FP · 과잉": "FP", "FN · 누락": "FN"}
    filtered = cases.copy()
    filters = [
        ("product", product_filter),
        ("filter_category", category_filter),
        ("filter_sub_reason", reason_filter),
        ("filter_sentiment", sentiment_filter),
    ]
    for col, value in filters:
        if value != "전체":
            filtered = filtered[filtered[col].astype(str) == value]
    filtered = filtered[filtered["status"].isin([status_map[x] for x in statuses])]
    filtered["_status_order"] = filtered["status"].map({"TP": 0, "FP": 1, "FN": 2})
    filtered = (
        filtered.sort_values(
            ["review_id", "_status_order", "filter_sub_reason"], kind="stable"
        )
        .drop(columns="_status_order")
        .reset_index(drop=True)
    )
    if filtered.empty:
        st.info("선택한 조건에 맞는 사례가 없습니다.")
        return
    review_ids = sorted(filtered["review_id"].dropna().astype(str).unique())
    page_size = 3
    total_pages = (len(review_ids) + page_size - 1) // page_size
    signature = (
        experiment,
        evaluation_level,
        product_filter,
        category_filter,
        reason_filter,
        sentiment_filter,
        tuple(statuses),
    )
    if st.session_state.get("dev_case_filter_signature") != signature:
        st.session_state["dev_case_page"] = 0
        st.session_state["dev_case_filter_signature"] = signature
    page_index = min(st.session_state.get("dev_case_page", 0), total_pages - 1)
    nav_l, nav_mid, nav_r = st.columns([1, 2, 1])
    with nav_l:
        if st.button("← 이전", disabled=page_index == 0, key="dev_case_prev"):
            st.session_state["dev_case_page"] = page_index - 1
            st.rerun()
    with nav_mid:
        st.markdown(
            f"<div style='text-align:center;padding-top:0.4rem'>페이지 <b>{page_index + 1} / {total_pages}</b> · {len(review_ids):,}개 리뷰 · 페이지당 최대 {page_size}개</div>",
            unsafe_allow_html=True,
        )
    with nav_r:
        if st.button(
            "다음 →", disabled=page_index >= total_pages - 1, key="dev_case_next"
        ):
            st.session_state["dev_case_page"] = page_index + 1
            st.rerun()
    selected_ids = review_ids[page_index * page_size : (page_index + 1) * page_size]
    for review_no, rid in enumerate(selected_ids, start=page_index * page_size + 1):
        review_cases = filtered[filtered["review_id"].astype(str) == rid]
        first = review_cases.iloc[0]
        with st.container(border=True):
            st.markdown(
                f"### 리뷰 {review_no} · ID {rid} · {first['product'] or '제품 정보 없음'}"
            )
            st.text_area(
                "리뷰 원문",
                first["RawText_clean"] or "원문을 찾지 못했습니다.",
                height=100,
                disabled=True,
                key=f"dev_raw_{experiment}_{rid}",
            )
            for i, (_, row) in enumerate(review_cases.iterrows(), start=1):
                status_label = {
                    "TP": "TP · 정답",
                    "FP": "FP · 과잉",
                    "FN": "FN · 누락",
                }[row["status"]]
                label = (
                    row["filter_sub_reason"]
                    if evaluation_level == "Aspect_category(대분류)"
                    else f"{row['filter_category']} / {row['filter_sub_reason']}"
                )
                with st.expander(
                    f"{status_label} · {label} · {row['filter_sentiment']}",
                    expanded=True,
                ):
                    st.write(f"**판정 설명:** {row['why']}")
                    left, right = st.columns(2)
                    with left:
                        st.markdown("**모델 예측**")
                        pred_label = (
                            row["pred_category"]
                            if evaluation_level == "Aspect_category(대분류)"
                            else f"{row['pred_category']} / {row['pred_sub_reason']}"
                        )
                        st.write(
                            f"{pred_label} · {row['pred_sentiment']}"
                            if row["pred_sub_reason"]
                            else "해당 Gold 항목을 모델이 예측하지 않음"
                        )
                        st.text_area(
                            "모델 근거 문구",
                            row["pred_evidence"] or "(근거 문구 없음)",
                            height=70,
                            disabled=True,
                            key=f"dev_pred_{experiment}_{rid}_{i}",
                        )
                    with right:
                        st.markdown("**Gold 정답**")
                        gold_label = (
                            row["gold_category"]
                            if evaluation_level == "Aspect_category(대분류)"
                            else f"{row['gold_category']} / {row['gold_sub_reason']}"
                        )
                        st.write(
                            f"{gold_label} · {row['gold_sentiment']}"
                            if row["gold_sub_reason"]
                            else row["other_gold"] or "Gold 정답 없음"
                        )
                        st.text_area(
                            "Gold 근거 문구",
                            row["gold_evidence"] or "(근거 문구 없음)",
                            height=70,
                            disabled=True,
                            key=f"dev_gold_{experiment}_{rid}_{i}",
                        )


def apply_risk_subreason_selection(table_key: str, options_key: str):
    """Copy the selected risk-table row into the business drill-down filters."""
    state = st.session_state.get(table_key)
    if state is None:
        return
    selection = getattr(state, "selection", None)
    selected_rows = getattr(selection, "rows", []) if selection is not None else []
    if not selected_rows and isinstance(state, dict):
        selected_rows = state.get("selection", {}).get("rows", [])
    options = st.session_state.get(options_key, [])
    if selected_rows and options:
        row_index = selected_rows[0]
        if 0 <= row_index < len(options):
            st.session_state["biz_reason"] = options[row_index]
            st.session_state["biz_sentiment"] = "부정"


st.title("계절가전 고객 리뷰 대시보드")
with st.sidebar:
    st.header("데이터 불러오기")
    mode = st.radio("불러오는 방법", ["파일 업로드", "폴더에서 불러오기"])
    if mode == "파일 업로드":
        st.caption(
            "폴더 업로드는 상대 경로를 유지할 수 없으므로, 실험별 파일명에 실험 이름을 넣어 주세요."
        )
        st.caption(
            "권장: 폴더 모드에서 data/common 및 data/output/<실험명> 구조를 읽습니다."
        )
        uploads = st.file_uploader(
            "관련 파일을 한 번에 선택", type=SUPPORTED, accept_multiple_files=True
        )
        tables = []
        if uploads:
            for f in uploads:
                try:
                    tables.append((f.name, read_table(f.name, f.getvalue())))
                except Exception as e:
                    st.warning(f"{f.name}: {e}")
    else:
        folder = st.text_input(
            "데이터 루트 폴더",
            value="/Users/kimsuyoung/Desktop/대학/26-2/인공지능프로젝트/data",
        )
        try:
            tables = read_folder(folder)
        except Exception as e:
            tables = []
            st.info(str(e))

st.markdown(
    "**개발자를 위한 실험 결과 비교 대시보드와, 의사결정자를 위한 리뷰 분석 페이지를 분리했습니다.**"
)
page = st.radio("페이지", ["실험 비교", "리뷰 분석 대시보드"], horizontal=True)
if not tables:
    st.info("왼쪽에서 평가 파일과 리뷰 데이터 파일을 불러오세요.")
    st.stop()
(
    normalized,
    predictions,
    metric_files,
    comparison,
    metadata,
    safety_summary,
    safety_examples,
    common_gold,
    gold_by_experiment,
    experiment_order,
) = get_experiment_tables(tables)
experiment_colors = {
    name: px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)]
    for i, name in enumerate(experiment_order)
}

if page == "실험 비교":
    st.header("실험 결과 비교")
    st.caption(
        "분류기 개발자를 위한 실험 비교 화면입니다. 실험 폴더를 추가하면 해당 결과가 자동으로 표와 그래프에 포함됩니다."
    )
    if comparison is not None:
        st.subheader("전체 평가 지표")
        exp_col = next(
            (c for c in ["experiment", "실험"] if c in comparison.columns), None
        )
        if exp_col:
            review_rows = []
            for experiment in experiment_order:
                if experiment not in comparison[exp_col].astype(str).values:
                    continue
                pred = predictions.get(experiment)
                summary_match = comparison[
                    comparison[exp_col].astype(str) == experiment
                ]
                summary_values = (
                    summary_match.iloc[0] if not summary_match.empty else {}
                )
                sentiment_count = None
                if pred is not None and {
                    "review_id",
                    "sub_reason",
                    "sentiment",
                }.issubset(pred.columns):
                    sentiment_count = (
                        pred.drop_duplicates(["review_id", "sub_reason"])["sentiment"]
                        .isin(["긍정", "부정", "중립"])
                        .sum()
                    )
                review_rows.append(
                    {
                        "실험": experiment,
                        "Total 리뷰 수": (
                            pred["review_id"].nunique()
                            if pred is not None and "review_id" in pred
                            else None
                        ),
                        "전체 감성 분류 수": sentiment_count,
                        "TP": summary_values.get("sub_tp"),
                        "FP": summary_values.get("sub_fp"),
                        "FN": summary_values.get("sub_fn"),
                        "Precision": summary_values.get("sub_precision"),
                        "Recall": summary_values.get("sub_recall"),
                        "F1": summary_values.get("sub_f1"),
                    }
                )
            st.markdown("**리뷰·감성 분류 및 전체 평가 지표**")
            review_summary = pd.DataFrame(review_rows)
            st.dataframe(
                review_summary.style.format(
                    {
                        c: "{:.1%}"
                        for c in ["Precision", "Recall", "F1"]
                        if c in review_summary.columns
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            metric_cols = [
                c for c in ["Precision", "Recall", "F1"] if c in review_summary.columns
            ]
            if metric_cols:
                metric_plot_data = review_summary[["실험", *metric_cols]].melt(
                    id_vars="실험",
                    value_vars=metric_cols,
                    var_name="지표",
                    value_name="점수",
                )
                metric_plot_data["점수"] = pd.to_numeric(
                    metric_plot_data["점수"], errors="coerce"
                )
                metric_plot_data = metric_plot_data.dropna(subset=["점수"])
                if not metric_plot_data.empty:
                    fig = px.bar(
                        metric_plot_data,
                        x="지표",
                        y="점수",
                        color="실험",
                        barmode="group",
                        text_auto=".1%",
                        title="실험별 전체 Precision · Recall · F1",
                        category_orders={
                            "지표": ["Precision", "Recall", "F1"],
                            "실험": experiment_order,
                        },
                        color_discrete_map=experiment_colors,
                    )
                    fig.update_yaxes(tickformat=".0%", range=[0, 1])
                    fig.update_layout(xaxis_title="평가 지표", yaxis_title="점수")
                    st.plotly_chart(fig, use_container_width=True)

            metric_order = [
                "실험",
                "Aspect_category(대분류)",
                "Sub_reason(소분류)",
                "Precision",
                "Recall",
                "F1",
                "TP",
                "FP",
                "FN",
            ]
            category_frames = []
            for experiment in experiment_order:
                pred = predictions.get(experiment)
                gold = gold_by_experiment.get(experiment, common_gold)
                if pred is None or gold is None:
                    continue
                class_metrics = class_metrics_for_level(pred, gold, "aspect_category")
                if class_metrics.empty:
                    continue
                class_metrics = class_metrics.rename(
                    columns={
                        "aspect_category": "Aspect_category(대분류)",
                        "tp": "TP",
                        "fp": "FP",
                        "fn": "FN",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1",
                    }
                )
                class_metrics["실험"] = experiment
                category_frames.append(class_metrics)
            st.markdown("**Aspect_category(대분류)별 지표**")
            if category_frames:
                category_table = pd.concat(category_frames, ignore_index=True)
                category_table = category_table[
                    [c for c in metric_order if c in category_table.columns]
                ]
                st.dataframe(
                    category_table.style.format(
                        {
                            c: "{:.1%}"
                            for c in ["Precision", "Recall", "F1"]
                            if c in category_table.columns
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                category_exp_order = [
                    x
                    for x in experiment_order
                    if x in category_table["실험"].astype(str).unique()
                ]
                category_table["실험"] = pd.Categorical(
                    category_table["실험"], categories=category_exp_order, ordered=True
                )
                category_table = category_table.sort_values(
                    ["Aspect_category(대분류)", "실험"], kind="stable"
                )
                fig = px.bar(
                    category_table,
                    x="Aspect_category(대분류)",
                    y="F1",
                    color="실험",
                    barmode="group",
                    text_auto=".0%",
                    title="Aspect_category(대분류)별 F1",
                    category_orders={"실험": category_exp_order},
                    color_discrete_map=experiment_colors,
                )
                fig.update_yaxes(tickformat=".0%", range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(
                    "Aspect_category(대분류) 기준 예측·정답 데이터가 있어야 계산할 수 있습니다."
                )

            sub_frames = []
            for experiment in experiment_order:
                raw = metric_files.get(experiment)
                if raw is None:
                    continue
                sub_frame = normalize_metric_columns(raw)
                if "세부 원인" in sub_frame:
                    sub_frame = sub_frame.rename(
                        columns={"세부 원인": "Sub_reason(소분류)"}
                    )
                    sub_frame["실험"] = experiment
                    sub_frames.append(sub_frame)
            st.markdown("**Sub_reason(소분류)별 지표**")
            if sub_frames:
                sub_table = pd.concat(sub_frames, ignore_index=True)
                sub_table = sub_table[
                    [c for c in metric_order if c in sub_table.columns]
                ]
                st.dataframe(
                    sub_table.style.format(
                        {
                            c: "{:.1%}"
                            for c in ["Precision", "Recall", "F1"]
                            if c in sub_table.columns
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("Sub_reason(소분류)별 지표 파일을 찾을 수 없습니다.")

    else:
        st.warning(
            "comparison_summary.csv를 불러오면 전체 지표 비교 차트가 표시됩니다."
        )

    st.subheader("Sub_reason(소분류)별 성능 비교")
    if metric_files:
        selected_reasons = st.multiselect(
            "확인할 Sub_reason(소분류)",
            sorted(
                set().union(
                    *(
                        set(
                            normalize_metric_columns(x)
                            .get("세부 원인", pd.Series(dtype=str))
                            .astype(str)
                        )
                        for x in metric_files.values()
                    )
                )
            ),
            default=[],
        )
        frames = []
        for exp, raw in metric_files.items():
            frame = normalize_metric_columns(raw)
            if "세부 원인" not in frame:
                continue
            frame["실험"] = exp
            if selected_reasons:
                frame = frame[frame["세부 원인"].isin(selected_reasons)]
            frames.append(frame)
        if frames:
            detail = pd.concat(frames, ignore_index=True)
            detail = detail.rename(columns={"세부 원인": "Sub_reason(소분류)"})
            detail_columns = [
                "실험",
                "Sub_reason(소분류)",
                "Precision",
                "Recall",
                "F1",
                "TP",
                "FP",
                "FN",
            ]
            detail = detail[[c for c in detail_columns if c in detail.columns]]
            if "F1" in detail:
                exp_order = [
                    x
                    for x in experiment_order
                    if x in detail["실험"].astype(str).unique()
                ]
                detail["실험"] = pd.Categorical(
                    detail["실험"], categories=exp_order, ordered=True
                )
                detail = detail.sort_values(
                    ["Sub_reason(소분류)", "실험"], kind="stable"
                )
                fig = px.bar(
                    detail,
                    x="Sub_reason(소분류)",
                    y="F1",
                    color="실험",
                    barmode="group",
                    text_auto=".0%",
                    title="Sub_reason(소분류)별 F1",
                    category_orders={"실험": exp_order},
                    color_discrete_map=experiment_colors,
                )
                fig.update_yaxes(tickformat=".0%", range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            # 소분류별 과잉(FP)·누락(FN) 건수 비교
            if {"TP", "FP", "FN"}.issubset(detail.columns):
                conditional = detail.copy()
                for col in ["TP", "FP", "FN"]:
                    conditional[col] = pd.to_numeric(
                        conditional[col], errors="coerce"
                    ).fillna(0)
                prob_order = [
                    x
                    for x in experiment_order
                    if x in conditional["실험"].astype(str).unique()
                ]
                st.markdown(
                    "**오답이 났을 때 과잉(FP)과 누락(FN) 중 어느 쪽이 많은가?**"
                )
                selected_error_exp = st.selectbox(
                    "오답 구성 확인 실험", prob_order, key="error_mix_exp"
                )
                selected_mix = conditional[
                    conditional["실험"].astype(str) == selected_error_exp
                ].copy()
                selected_mix["오답 전체"] = selected_mix["FP"] + selected_mix["FN"]
                selected_mix["소분류 표시"] = selected_mix["Sub_reason(소분류)"].astype(
                    str
                )
                mix = selected_mix.melt(
                    id_vars=["Sub_reason(소분류)", "소분류 표시", "오답 전체"],
                    value_vars=["FP", "FN"],
                    var_name="오답 유형",
                    value_name="오답 건수",
                )
                mix["오답 유형"] = mix["오답 유형"].map(
                    {"FP": "과잉(FP)", "FN": "누락(FN)"}
                )
                fig = px.bar(
                    mix,
                    x="오답 건수",
                    y="소분류 표시",
                    color="오답 유형",
                    barmode="group",
                    orientation="h",
                    text_auto=True,
                    hover_data={
                        "오답 전체": True,
                        "오답 건수": True,
                        "소분류 표시": False,
                    },
                    category_orders={"오답 유형": ["과잉(FP)", "누락(FN)"]},
                    color_discrete_map={
                        "과잉(FP)": "#F58518",
                        "누락(FN)": "#4C78A8",
                    },
                    title="Sub_reason(소분류)별 FP·FN 건수",
                )
                fig.update_layout(
                    xaxis_title="오답 건수", yaxis_title="Sub_reason(소분류)"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "막대 길이는 Sub_reason(소분류)별 FP·FN 건수입니다. 막대에 마우스를 올리면 오답 전체(FP+FN) 건수를 확인할 수 있습니다."
                )
                ratio_source = selected_mix.copy()
                ratio_source["과잉(FP) 비율"] = (
                    ratio_source["FP"]
                    .div(
                        ratio_source["오답 전체"].where(ratio_source["오답 전체"].ne(0))
                    )
                    .fillna(0)
                )
                ratio_source["누락(FN) 비율"] = (
                    ratio_source["FN"]
                    .div(
                        ratio_source["오답 전체"].where(ratio_source["오답 전체"].ne(0))
                    )
                    .fillna(0)
                )
                ratio_source["소분류 표시"] = ratio_source["Sub_reason(소분류)"].astype(
                    str
                )
                ratio = ratio_source.melt(
                    id_vars=["소분류 표시", "오답 전체"],
                    value_vars=["과잉(FP) 비율", "누락(FN) 비율"],
                    var_name="오답 유형",
                    value_name="오답 비율",
                )
                ratio["표시 비율"] = ratio["오답 비율"].map(
                    lambda value: f"{value:.0%}"
                )
                ratio_fig = px.bar(
                    ratio,
                    x="오답 비율",
                    y="소분류 표시",
                    color="오답 유형",
                    barmode="stack",
                    orientation="h",
                    text="표시 비율",
                    hover_data={
                        "오답 전체": True,
                        "오답 비율": ":.1%",
                        "표시 비율": False,
                        "소분류 표시": False,
                    },
                    category_orders={"오답 유형": ["과잉(FP) 비율", "누락(FN) 비율"]},
                    color_discrete_map={
                        "과잉(FP) 비율": "#F58518",
                        "누락(FN) 비율": "#4C78A8",
                    },
                    title="Sub_reason(소분류)별 오답 중 FP·FN 비율",
                )
                ratio_fig.update_xaxes(tickformat=".0%", range=[0, 1])
                ratio_fig.update_layout(
                    xaxis_title="해당 소분류 오답 중 비율",
                    yaxis_title="Sub_reason(소분류)",
                )
                st.plotly_chart(ratio_fig, use_container_width=True)
                st.caption(
                    "비율은 각 Sub_reason(소분류)의 FP+FN을 100%로 두고 계산합니다. 오답 건수가 0인 소분류는 비율 0%로 표시됩니다."
                )

    else:
        st.info(
            "data/output/<실험명>/w5_metrics_subreason.parquet을 넣으면 실험별 소분류 지표를 자동 비교합니다."
        )
    if predictions:
        dev_experiment = st.selectbox(
            "Drill-down 실험", list(predictions.keys()), key="dev_drilldown_experiment"
        )
        dev_gold = gold_by_experiment.get(dev_experiment, common_gold)
        render_developer_drilldown(
            predictions[dev_experiment], dev_gold, metadata, dev_experiment
        )
    if safety_summary is not None or safety_examples is not None:
        st.subheader("안전 태그 참고")
        st.caption(
            "안전 gold 라벨이 없는 경우 이 값은 성능 점수가 아니라 모델이 표시한 사례의 수입니다."
        )
        if safety_summary is not None and "n_flagged" in safety_summary.columns:
            st.metric(
                "안전 우려로 표시된 사례",
                int(
                    pd.to_numeric(safety_summary["n_flagged"], errors="coerce")
                    .fillna(0)
                    .sum()
                ),
            )
            if "flagged_rate" in safety_summary.columns:
                st.caption(
                    f"표시 비율: {pd.to_numeric(safety_summary['flagged_rate'], errors='coerce').fillna(0).mean():.1%}"
                )
        if safety_examples is not None:
            st.dataframe(safety_examples, use_container_width=True, hide_index=True)

else:
    st.header("리뷰 분석 대시보드")
    st.caption("기업 의사결정자를 위한 페이지")
    with st.container(border=True):
        st.markdown("#### 대시보드 Flow")
        st.markdown("""
1. **제품과 Aspect_category(대분류)를 선택합니다.** 해당 제품의 긍정·부정·중립 언급과 주요 부정 Sub_reason(소분류), 부정 키워드를 확인합니다.
2. **부정 비율이 높은 소분류를 확인합니다.** 부정 비율 30% 이상은 주의, 50% 이상은 경계로 표시했습니다.
3. **관심 있는 소분류 행을 클릭합니다.** 아래 리뷰 Drill-down이 해당 소분류와 부정 감성으로 자동 필터링됩니다.
4. **고객 원문과 근거 문구를 확인합니다.** 부정 언급의 구체적인 맥락을 살펴 제품 개선이나 고객 대응의 근거로 활용할 수 있습니다.
            """)
    if not predictions:
        st.warning(
            "이 페이지에는 review_id와 sub_reason 열이 포함된 실험별 예측 parquet이 필요합니다. 비교 지표 파일만으로는 실제 리뷰 분포를 계산할 수 없습니다."
        )
        st.stop()
    exp = st.selectbox("분석할 실험", list(predictions.keys()))
    df = merge_metadata(predictions[exp], metadata)
    if df.empty:
        st.warning("선택된 실험의 예측 행이 없습니다.")
        st.stop()

    # 필수 기능 1: 제품 또는 카테고리 선택
    product_options = sorted(
        v for v in df["product"].dropna().astype(str).unique() if v.strip()
    )
    category_options = sorted(
        v for v in df["aspect_category"].dropna().astype(str).unique() if v.strip()
    )
    c1, c2 = st.columns(2)
    with c1:
        product = (
            st.selectbox("제품 선택", ["전체 제품"] + product_options)
            if product_options
            else "전체 제품"
        )
    with c2:
        category = (
            st.selectbox("Aspect 카테고리 선택", ["전체 카테고리"] + category_options)
            if category_options
            else "전체 카테고리"
        )
    if product != "전체 제품":
        df = df[df["product"] == product]
    if category != "전체 카테고리":
        df = df[df["aspect_category"] == category]

    st.caption(
        "예측 결과에 제품 정보가 없으면 Aspect 대분류를 선택해 필터링할 수 있습니다."
    )
    # 필수 기능 2: Aspect별 긍정/부정 현황
    st.subheader("Aspect(대분류) 별 긍정·부정 현황")
    sentiment = (
        df[df["sentiment"].isin(["긍정", "부정", "중립"])]
        .groupby(["aspect_category", "sentiment"])
        .size()
        .reset_index(name="언급 수")
    )
    if not sentiment.empty:
        fig = px.bar(
            sentiment,
            x="aspect_category",
            y="언급 수",
            color="sentiment",
            barmode="group",
            color_discrete_map={
                "긍정": "#3B82F6",
                "부정": "#EF4444",
                "중립": "#9CA3AF",
            },
        )
        fig.update_layout(
            xaxis_title="Aspect 대분류", yaxis_title="언급 수", legend_title="감성"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("감성 라벨이 있는 예측 행이 없습니다.")

    # 필수 기능 3: Sub-reason별 주요 부정 원인
    st.subheader("Sub-reason(세부 원인별) 부정 언급")
    negative = df[df["sentiment"] == "부정"]
    causes = (
        negative.groupby("sub_reason")
        .size()
        .sort_values(ascending=False)
        .head(15)
        .rename("부정 언급 수")
        .reset_index()
    )
    if not causes.empty:
        fig = px.bar(
            causes.sort_values("부정 언급 수"),
            x="부정 언급 수",
            y="sub_reason",
            orientation="h",
            color_discrete_sequence=["#EF4444"],
        )
        fig.update_layout(xaxis_title="부정 언급 수", yaxis_title="세부 원인")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("선택한 범위에서 부정 언급이 없습니다.")

    # 필수 기능 4: 주요 키워드 또는 의견
    st.subheader("주요 부정 키워드와 고객 의견")
    text_col = (
        "evidence_text"
        if negative["evidence_text"].str.strip().any()
        else "RawText_clean"
    )
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(
            top_terms(negative[text_col]), hide_index=True, use_container_width=True
        )
    with right:
        opinions = (
            negative[negative[text_col].str.strip().ne("")][
                ["review_id", "sub_reason", text_col]
            ]
            .drop_duplicates()
            .head(12)
        )
        st.dataframe(
            opinions.rename(
                columns={
                    "review_id": "리뷰 ID",
                    "sub_reason": "세부 원인",
                    text_col: "고객 의견/근거",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("부정 비율이 높은 Sub_reason(소분류)")
    st.caption(
        "선택한 실험과 제품 범위의 예측 결과를 기준으로 계산합니다. 주의는 부정 비율 30% 이상 50% 미만, 경계는 50% 이상입니다."
    )
    sentiment_rows = df[["review_id", "sub_reason", "sentiment"]].copy()
    sentiment_rows = sentiment_rows[
        sentiment_rows["sentiment"].isin(["긍정", "부정", "중립"])
    ]
    sentiment_rows = sentiment_rows.drop_duplicates(["review_id", "sub_reason"])
    sentiment_stats = (
        sentiment_rows.assign(
            부정건수=sentiment_rows["sentiment"].eq("부정").astype(int)
        )
        .groupby("sub_reason", as_index=False)
        .agg(전체건수=("review_id", "size"), 부정건수=("부정건수", "sum"))
    )
    sentiment_stats["부정 비율"] = (
        sentiment_stats["부정건수"] / sentiment_stats["전체건수"]
    )
    sentiment_stats = sentiment_stats.rename(
        columns={"sub_reason": "Sub_reason(소분류)"}
    )
    caution = (
        sentiment_stats[
            (sentiment_stats["부정 비율"] >= 0.30)
            & (sentiment_stats["부정 비율"] < 0.50)
        ]
        .sort_values("부정 비율", ascending=False)
        .reset_index(drop=True)
    )
    alert = (
        sentiment_stats[sentiment_stats["부정 비율"] >= 0.50]
        .sort_values("부정 비율", ascending=False)
        .reset_index(drop=True)
    )
    st.caption(
        "주의 또는 경계 표에서 Sub_reason(소분류) 행을 클릭하면 아래 리뷰 Drill-down이 해당 소분류와 부정 감성으로 자동 필터링됩니다."
    )
    caution_col, alert_col = st.columns(2)
    with caution_col:
        st.markdown("**주의 · 부정 비율 30% 이상**")
        if caution.empty:
            st.caption("해당 Sub_reason(소분류)가 없습니다.")
        else:
            st.session_state["caution_subreason_options"] = caution[
                "Sub_reason(소분류)"
            ].tolist()
            st.dataframe(
                caution.style.format({"부정 비율": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
                on_select=lambda: apply_risk_subreason_selection(
                    "caution_subreason_table", "caution_subreason_options"
                ),
                selection_mode="single-row",
                key="caution_subreason_table",
            )
    with alert_col:
        st.markdown("**경계 · 부정 비율 50% 이상**")
        if alert.empty:
            st.caption("해당 Sub_reason(소분류)가 없습니다.")
        else:
            st.session_state["alert_subreason_options"] = alert[
                "Sub_reason(소분류)"
            ].tolist()
            st.dataframe(
                alert.style.format({"부정 비율": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
                on_select=lambda: apply_risk_subreason_selection(
                    "alert_subreason_table", "alert_subreason_options"
                ),
                selection_mode="single-row",
                key="alert_subreason_table",
            )

    st.subheader("리뷰 Drill-down")
    st.caption(
        "기업 의사결정자가 제품·Aspect_category(대분류)·Sub_reason(소분류)·감성별로 고객의 실제 리뷰와 근거 문구를 살펴보는 화면입니다."
    )
    business = df.copy()
    b1, b2, b3 = st.columns(3)
    with b1:
        business_reasons = sorted(
            x for x in business["sub_reason"].dropna().astype(str).unique() if x.strip()
        )
        business_reason = st.selectbox(
            "Sub_reason(소분류)", ["전체"] + business_reasons, key="biz_reason"
        )
    with b2:
        business_sentiments = [
            x
            for x in ["긍정", "부정", "중립"]
            if x in set(business["sentiment"].astype(str))
        ]
        business_sentiment = st.selectbox(
            "감성", ["전체"] + business_sentiments, key="biz_sentiment"
        )
    with b3:
        max_reviews = st.selectbox(
            "페이지당 리뷰 수", [3, 5, 10], index=0, key="biz_page_size"
        )
    if business_reason != "전체":
        business = business[business["sub_reason"].astype(str) == business_reason]
    if business_sentiment != "전체":
        business = business[business["sentiment"].astype(str) == business_sentiment]
    business = business.drop_duplicates(
        ["review_id", "sub_reason", "sentiment", "evidence_text"]
    )
    business = business.sort_values(["review_id", "sub_reason"], kind="stable")
    review_ids = sorted(business["review_id"].dropna().astype(str).unique())
    if not review_ids:
        st.info("선택한 조건에 해당하는 리뷰가 없습니다.")
    else:
        total_pages = (len(review_ids) + max_reviews - 1) // max_reviews
        business_signature = (
            exp,
            product,
            category,
            business_reason,
            business_sentiment,
            max_reviews,
        )
        if st.session_state.get("business_filter_signature") != business_signature:
            st.session_state["business_page"] = 0
            st.session_state["business_filter_signature"] = business_signature
        page_index = min(st.session_state.get("business_page", 0), total_pages - 1)
        nav_l, nav_mid, nav_r = st.columns([1, 2, 1])
        with nav_l:
            if st.button("← 이전", disabled=page_index == 0, key="biz_prev"):
                st.session_state["business_page"] = page_index - 1
                st.rerun()
        with nav_mid:
            st.markdown(
                f"<div style='text-align:center;padding-top:0.4rem'>페이지 <b>{page_index + 1} / {total_pages}</b> · 전체 {len(review_ids):,}개 리뷰 · 페이지당 {max_reviews}개</div>",
                unsafe_allow_html=True,
            )
        with nav_r:
            if st.button(
                "다음 →", disabled=page_index >= total_pages - 1, key="biz_next"
            ):
                st.session_state["business_page"] = page_index + 1
                st.rerun()
        page_ids = review_ids[page_index * max_reviews : (page_index + 1) * max_reviews]
        for review_no, rid in enumerate(page_ids, start=page_index * max_reviews + 1):
            review_aspects = business[business["review_id"].astype(str) == rid]
            first = review_aspects.iloc[0]
            with st.container(border=True):
                st.markdown(
                    f"### 리뷰 {review_no} · ID {rid} · {first['product'] or '제품 정보 없음'}"
                )
                st.text_area(
                    "실제 리뷰 원문",
                    first["RawText_clean"] or "원문을 찾지 못했습니다.",
                    height=110,
                    disabled=True,
                    key=f"biz_raw_{exp}_{rid}",
                )
                details = review_aspects[
                    [
                        c
                        for c in [
                            "aspect_category",
                            "sub_reason",
                            "sentiment",
                            "evidence_text",
                        ]
                        if c in review_aspects.columns
                    ]
                ].rename(
                    columns={
                        "aspect_category": "Aspect_category(대분류)",
                        "sub_reason": "Sub_reason(소분류)",
                        "sentiment": "감성",
                        "evidence_text": "리뷰 근거 문구",
                    }
                )
                st.dataframe(details, hide_index=True, use_container_width=True)
