"""Analyze background energy and gold ranks on ArguAna, FiQA, and SciFact."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..cache import candidate_embedding_cache_paths, load_or_encode_embeddings
from ..constants import ENCODER_PRESETS, DEFAULT_ENCODER_PRESET, MAX_TOKENS_PER_DOC
from ..encoder import DenseEncoder
from ..text_model import load_or_build_text_model, shared_text_model_cache_tag
from ..utils import l2_normalize
from .background_energy import background_energy, distribution
from .recall_depth import rank_of_best_gold


DEFAULT_DATASETS = ("arguana", "fiqa", "scifact")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _read_qrels(path: Path):
    qrels = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score = int(row.get("score", row.get("relevance", 0)))
            if score > 0:
                qrels[str(row["query-id"])][str(row["corpus-id"])] = score
    return qrels


def _format_document(item) -> str:
    return "\n".join(part for part in (item.get("title", ""), item.get("text", "")) if part)


def _load_dataset(root: Path, name: str):
    dataset_root = root / name
    corpus = list(_read_jsonl(dataset_root / "corpus.jsonl"))
    query_by_id = {str(row["_id"]): row for row in _read_jsonl(dataset_root / "queries.jsonl")}
    qrels = _read_qrels(dataset_root / "qrels" / "test.tsv")

    # ArguAna incorporates the query text in the corpus; BEIR excludes those
    # self-documents during retrieval.
    if name == "arguana":
        query_ids = set(query_by_id)
        corpus = [doc for doc in corpus if str(doc["_id"]) not in query_ids]

    query_ids = [qid for qid in qrels if qid in query_by_id]
    corpus_ids = [str(doc["_id"]) for doc in corpus]
    corpus_texts = [_format_document(doc) for doc in corpus]
    query_texts = [_format_document(query_by_id[qid]) for qid in query_ids]
    return qrels, query_ids, corpus_ids, corpus_texts, query_texts


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    results,
    sra_recall_summary: dict | None,
    sra_energy_summary: dict | None,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for dataset, result in results.items():
        points = result["recall"]
        axis.plot(
            [point["depth"] for point in points],
            [point["recall"] for point in points],
            marker="o",
            label=dataset,
        )
    if sra_recall_summary:
        points = sra_recall_summary["overall"]["dense"]["recall"]
        axis.plot(
            [int(depth) for depth in points],
            list(points.values()),
            marker="s",
            label="SRA-Bench",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Top-k depth (log scale)")
    axis.set_ylabel("Gold recall")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Gold recall over depth")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ordinary_text_recall_depth.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    points = []
    for dataset, result in results.items():
        ed = result["document_energy"]["mean"]
        eq = result["query_energy"]["mean"]
        points.append((dataset, ed, eq, "ordinary"))
    if sra_energy_summary:
        sra_query_rows = list(sra_energy_summary["query_background_energy"].values())
        sra_query_count = sum(row["count"] for row in sra_query_rows)
        points.append(
            (
                "SRA-Bench",
                sra_energy_summary["document_background_energy"]["mean"],
                float(
                    sum(row["mean"] * row["count"] for row in sra_query_rows)
                    / max(1, sra_query_count)
                ),
                "skill",
            )
        )
    for dataset, ed, eq, group in points:
        axis.scatter(ed, eq, marker="o" if group == "ordinary" else "s", s=55)
        axis.annotate(dataset, (ed, eq), xytext=(4, 4), textcoords="offset points", fontsize=8)
    limit = max((ed for _, ed, _, _ in points), default=1.0)
    axis.plot([0.0, limit * 1.1], [0.0, limit * 1.1], "--", color="black", linewidth=1)
    axis.set_xlabel("Mean document background energy $E_D$")
    axis.set_ylabel("Mean query background energy $E_Q$")
    axis.set_title("Background energy: ordinary text vs SRA-Bench")
    fig.tight_layout()
    fig.savefig(output_dir / "ordinary_text_energy_gap.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    labels = list(results)
    values = [results[name]["mean_documents_ranked_above_gold_pct"] for name in labels]
    if sra_recall_summary:
        labels.append("SRA-Bench")
        values.append(
            sra_recall_summary["overall"]["dense"]["mean_documents_ranked_above_gold_pct"]
        )
    axis.bar(labels, values)
    axis.set_ylabel("Docs ranked above best gold (%)")
    axis.set_title("Expected-gold ranking difficulty")
    fig.tight_layout()
    fig.savefig(output_dir / "ordinary_text_gold_rank_pct.png", dpi=200)
    plt.close(fig)


def run(args) -> dict:
    package_root = Path(__file__).resolve().parents[1]
    preset = ENCODER_PRESETS[args.encoder_preset]
    model_path = args.model_path or str((package_root / preset["model_path"]).resolve())
    max_seq_length = args.max_seq_length or 4096
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    encoder = None
    results = {}
    rank_rows = []

    for name in args.datasets:
        print(f"[dataset] {name}", flush=True)
        qrels, query_ids, corpus_ids, corpus_texts, query_texts = _load_dataset(
            args.beir_root.resolve(), name
        )
        doc_paths = candidate_embedding_cache_paths(
            cache_dir=cache_dir,
            kind="doc_all",
            dataset=f"beir_{name}",
            model_path=model_path,
            max_seq_length=max_seq_length,
            ids_or_texts=corpus_ids,
        )
        docs, encoder = load_or_encode_embeddings(
            encoder=encoder,
            texts=corpus_texts,
            cache_paths=doc_paths,
            batch_size=args.doc_batch,
            prompt_name=None,
            label=f"{name} corpus",
            model_path=model_path,
            device=args.device,
            max_seq_length=max_seq_length,
            allow_encode=True,
        )
        query_paths = candidate_embedding_cache_paths(
            cache_dir=cache_dir,
            kind="query",
            dataset=f"beir_{name}",
            model_path=model_path,
            max_seq_length=max_seq_length,
            ids_or_texts=query_texts,
        )
        queries, encoder = load_or_encode_embeddings(
            encoder=encoder,
            texts=query_texts,
            cache_paths=query_paths,
            batch_size=args.query_batch,
            prompt_name="query",
            label=f"{name} queries",
            model_path=model_path,
            device=args.device,
            max_seq_length=max_seq_length,
            allow_encode=True,
        )
        docs = l2_normalize(docs)
        queries = l2_normalize(queries)
        stats, basis = load_or_build_text_model(
            doc_texts=corpus_texts,
            doc_embeddings=docs,
            cache_dir=output_dir / "text_models",
            cache_tag=shared_text_model_cache_tag(name, corpus_ids),
            max_tokens_per_doc=args.max_tokens_per_doc,
        )
        doc_energy = background_energy(docs, basis)
        query_energy = background_energy(queries, basis)

        id_to_index = {doc_id: index for index, doc_id in enumerate(corpus_ids)}
        ranks = []
        for start in tqdm(range(0, len(query_ids), args.score_batch), desc=f"  {name}", unit="batch"):
            end = min(start + args.score_batch, len(query_ids))
            scores = queries[start:end] @ docs.T
            for offset, qid in enumerate(query_ids[start:end]):
                gold_indices = np.asarray(
                    [id_to_index[doc_id] for doc_id in qrels[qid] if doc_id in id_to_index],
                    dtype=np.int64,
                )
                rank = rank_of_best_gold(scores[offset], gold_indices)
                ranks.append(rank)
                rank_rows.append({"dataset": name, "query_id": qid, "dense_rank": rank})

        ranks_array = np.asarray(ranks, dtype=np.int64)
        recall = [
            {
                "depth": min(depth, len(corpus_ids)),
                "recall": float(np.mean(ranks_array <= min(depth, len(corpus_ids)))),
            }
            for depth in args.depths
        ]
        results[name] = {
            "corpus_size": len(corpus_ids),
            "query_count": len(query_ids),
            "background_rank": int(stats.background_rank),
            "background_explained_variance": float(stats.background_explained_variance),
            "generic_token_count": len(stats.generic_tokens),
            "document_energy": distribution(doc_energy),
            "query_energy": distribution(query_energy),
            "mean_documents_ranked_above_gold_pct": float(
                np.mean((ranks_array - 1) / len(corpus_ids) * 100.0)
            ),
            "recall": recall,
        }

    sra_recall_summary = None
    if args.sra_recall_summary and args.sra_recall_summary.exists():
        sra_recall_summary = json.loads(args.sra_recall_summary.read_text(encoding="utf-8"))
    sra_energy_summary = None
    if args.sra_energy_summary and args.sra_energy_summary.exists():
        sra_energy_summary = json.loads(args.sra_energy_summary.read_text(encoding="utf-8"))
    output = {
        "model_path": model_path,
        "max_seq_length": max_seq_length,
        "datasets": results,
        "sra_recall_summary": (
            str(args.sra_recall_summary) if args.sra_recall_summary else None
        ),
        "sra_energy_summary": (
            str(args.sra_energy_summary) if args.sra_energy_summary else None
        ),
    }
    _write_csv(output_dir / "dense_gold_ranks.csv", rank_rows)
    _plot(results, sra_recall_summary, sra_energy_summary, output_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def parse_args(argv=None):
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beir-root", type=Path, default=package_root / "BEIR")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--encoder-preset", choices=tuple(ENCODER_PRESETS), default=DEFAULT_ENCODER_PRESET)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path, default=package_root / "cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sra-recall-summary",
        type=Path,
        default=package_root / "analysis_output" / "recall_depth_qwen3_sra" / "summary.json",
    )
    parser.add_argument(
        "--sra-energy-summary",
        type=Path,
        default=package_root
        / "analysis_output"
        / "background_energy_qwen3_sra"
        / "summary.json",
    )
    parser.add_argument("--doc-batch", type=int, default=8)
    parser.add_argument("--query-batch", type=int, default=32)
    parser.add_argument("--score-batch", type=int, default=64)
    parser.add_argument("--depths", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--max-tokens-per-doc", type=int, default=MAX_TOKENS_PER_DOC)
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(args.datasets))
    args.depths = sorted(set(args.depths))
    return args


def main(argv=None) -> None:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
