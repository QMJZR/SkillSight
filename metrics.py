"""Retrieval metrics and JSON summary helpers for SkillSight."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .data import QueryExample
from .retriever import RetrievalResult
from .utils import json_float

SUMMARY_METRIC_LABELS = {"recall": "R", "hit": "H", "mrr": "MRR"}


def compute_retrieval_metrics(
    result: RetrievalResult,
    examples: Sequence[QueryExample],
    skill_ids: Sequence[str],
    top_ks: Sequence[int],
) -> Dict[str, object]:
    skill_ids_arr = np.asarray(skill_ids)
    skill_id_set = set(skill_ids)
    n = len(examples)
    n_gold_oov = sum(1 for ex in examples if not (set(ex.gold_ids) & skill_id_set))

    topk: Dict[str, Dict[str, float]] = {}
    for k0 in top_ks:
        k = min(k0, result.top_indices.shape[1])
        recall_sum = hit_sum = mrr_sum = 0.0
        for i, ex in enumerate(examples):
            gold = set(ex.gold_ids)
            ids = skill_ids_arr[result.top_indices[i, :k]].tolist()
            hit_positions = [rank + 1 for rank, sid in enumerate(ids) if sid in gold]
            recall_sum += len(set(ids) & gold) / max(1, len(gold))
            hit_sum += 1.0 if hit_positions else 0.0
            mrr_sum += 1.0 / hit_positions[0] if hit_positions else 0.0
        topk[str(k0)] = {
            "recall": json_float(recall_sum / n if n else 0.0),
            "hit": json_float(hit_sum / n if n else 0.0),
            "mrr": json_float(mrr_sum / n if n else 0.0),
        }
    return {
        "n": int(n),
        "n_gold_oov": int(n_gold_oov),
        # Online retrieval latency for this (dataset, variant): total scoring
        # wall time and the derived throughput. See RetrievalResult.search_seconds.
        "search_seconds": json_float(result.search_seconds),
        "queries_per_second": json_float(
            n / result.search_seconds if result.search_seconds > 0 else 0.0
        ),
        "topk": topk,
    }


def compute_overall_metrics(
    per_category: Dict[str, Dict[str, dict]],
    variants: Sequence[str],
    top_ks: Sequence[int],
) -> Dict[str, Dict[str, object]]:
    overall: Dict[str, Dict[str, object]] = {"macro": {}, "micro": {}}
    for variant in variants:
        category_rows = [per_category[dataset][variant] for dataset in per_category if variant in per_category[dataset]]
        if not category_rows:
            continue

        macro_topk: Dict[str, Dict[str, float]] = {}
        micro_topk: Dict[str, Dict[str, float]] = {}
        total_n = sum(int(row["n"]) for row in category_rows)
        total_oov = sum(int(row.get("n_gold_oov", 0)) for row in category_rows)
        # Sum online retrieval latency across datasets for this variant.
        total_search_seconds = sum(float(row.get("search_seconds", 0.0)) for row in category_rows)

        for k0 in top_ks:
            key = str(k0)
            macro_topk[key] = {
                metric: json_float(np.mean([float(row["topk"][key][metric]) for row in category_rows]))
                for metric in ("recall", "hit", "mrr")
            }
            if total_n > 0:
                micro_topk[key] = {
                    metric: json_float(
                        sum(float(row["topk"][key][metric]) * int(row["n"]) for row in category_rows) / total_n
                    )
                    for metric in ("recall", "hit", "mrr")
                }
            else:
                micro_topk[key] = {metric: 0.0 for metric in ("recall", "hit", "mrr")}

        overall["macro"][variant] = {
            "num_categories": int(len(category_rows)),
            "topk": macro_topk,
        }
        overall["micro"][variant] = {
            "n": int(total_n),
            "n_gold_oov": int(total_oov),
            "search_seconds": json_float(total_search_seconds),
            "queries_per_second": json_float(
                total_n / total_search_seconds if total_search_seconds > 0 else 0.0
            ),
            "avg_latency_ms": json_float(
                total_search_seconds / total_n * 1000.0
                if (total_n > 0 and total_search_seconds > 0) else 0.0
            ),
            "topk": micro_topk,
        }
    return overall

def build_topk_summary_tables(
    per_category: Dict[str, Dict[str, dict]],
    overall: Dict[str, Dict[str, object]],
    variants: Sequence[str],
    top_ks: Sequence[int],
    metrics: Sequence[str] = ("recall", "hit"),
    overall_average: str = "micro",
) -> Dict[str, list[dict]]:
    tables: Dict[str, list[dict]] = {}
    for variant in variants:
        rows: list[dict] = []
        for dataset, dataset_rows in per_category.items():
            if variant not in dataset_rows:
                continue
            row_metrics = dataset_rows[variant]
            row = {"dataset": dataset, "n": int(row_metrics.get("n", 0))}
            for metric in metrics:
                label = SUMMARY_METRIC_LABELS[metric]
                for k0 in top_ks:
                    row[f"{label}@{k0}"] = json_float(row_metrics["topk"][str(k0)][metric])
            rows.append(row)

        overall_row = overall.get(overall_average, {}).get(variant)
        if overall_row is not None:
            row = {"dataset": "overall", "n": int(overall_row.get("n", sum(r["n"] for r in rows)))}
            for metric in metrics:
                label = SUMMARY_METRIC_LABELS[metric]
                for k0 in top_ks:
                    row[f"{label}@{k0}"] = json_float(overall_row["topk"][str(k0)][metric])
            rows.append(row)
        tables[variant] = rows
    return tables


def print_overall_summary(overall: Dict[str, Dict[str, object]], top_ks: Sequence[int]) -> None:
    """Print the variant comparison at the primary k for both macro and micro."""
    primary = 10 if 10 in top_ks else max(top_ks)
    key = str(primary)
    for average_name in ("macro", "micro"):
        block = overall.get(average_name, {})
        if not block:
            continue
        print("\n" + "=" * 90)
        print(f"[overall {average_name}] k={primary}")
        print("=" * 90)
        print(f"{'variant':28s} {'Recall':>10s} {'Hit':>10s} {'MRR':>10s}")
        print("-" * 90)
        for variant in sorted(block):
            m = block[variant]["topk"][key]
            print(f"{variant:28s} {m['recall']:10.4f} {m['hit']:10.4f} {m['mrr']:10.4f}")


def print_latency_summary(overall: Dict[str, Dict[str, object]]) -> None:
    """Print average online retrieval latency (ms/query) and QPS per variant."""
    micro = overall.get("micro", {})
    if not micro:
        return
    print("\n" + "=" * 90)
    print("[retrieval latency micro] (online scoring only; excludes corpus embedding)")
    print("=" * 90)
    print(f"{'variant':28s} {'avg_latency_ms':>16s} {'QPS':>10s}")
    print("-" * 90)
    for variant in sorted(micro):
        row = micro[variant]
        print(
            f"{variant:28s} {row.get('avg_latency_ms', 0.0):16.2f} "
            f"{row.get('queries_per_second', 0.0):10.2f}"
        )


def print_full_topk_tables(
    summary_tables: Dict[str, list[dict]],
    top_ks: Sequence[int],
    metrics: Sequence[str] = ("recall", "hit", "mrr"),
    overall_average: str = "micro",
) -> None:
    columns = ["dataset", "n"]
    for metric in metrics:
        label = SUMMARY_METRIC_LABELS[metric]
        columns.extend([f"{label}@{k0}" for k0 in top_ks])

    for variant, rows in summary_tables.items():
        if not rows:
            continue
        widths = {col: len(col) for col in columns}
        for row in rows:
            for col in columns:
                widths[col] = max(widths[col], len(str(row.get(col, ""))))
        for col in columns:
            if col not in {"dataset", "n"}:
                widths[col] = max(widths[col], 7)

        print("\n" + "=" * 120)
        print(f"[final top-k summary] variant={variant} | overall={overall_average}")
        print("=" * 120)
        header = " ".join(f"{col:>{widths[col]}}" if col != "dataset" else f"{col:<{widths[col]}}" for col in columns)
        print(header)
        print("-" * len(header))
        for row in rows:
            cells = []
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                if col == "dataset":
                    cells.append(f"{text:<{widths[col]}}")
                else:
                    cells.append(f"{text:>{widths[col]}}")
            print(" ".join(cells))
