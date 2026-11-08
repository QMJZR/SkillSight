"""Evaluate SRA-Bench gold recall over retrieval depth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..cache import candidate_embedding_cache_paths, load_or_encode_embeddings
from ..constants import DEFAULT_DATASETS, DEFAULT_ENCODER_PRESET, ENCODER_PRESETS, MAX_TOKENS_PER_DOC
from ..data import collect_queries, load_instances, skill_text
from ..text_model import load_or_build_text_model, project_out_background, shared_text_model_cache_tag
from ..utils import l2_normalize


def rank_of_best_gold(scores: np.ndarray, gold_indices: np.ndarray) -> int:
    """Return the 1-based rank of the highest-scoring gold item."""
    if gold_indices.size == 0:
        return int(scores.shape[0]) + 1
    best_score = float(scores[gold_indices].max())
    return 1 + int(np.count_nonzero(scores > best_score))


def _load_vectors(texts, paths, prompt_name, label, model_path, max_seq_length):
    vectors, _ = load_or_encode_embeddings(
        encoder=None,
        texts=texts,
        cache_paths=paths,
        batch_size=1,
        prompt_name=prompt_name,
        label=label,
        model_path=model_path,
        device="cpu",
        max_seq_length=max_seq_length,
        allow_encode=False,
    )
    return vectors


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(overall, depths, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for variant in overall:
        values = [overall[variant]["recall"][str(depth)] for depth in depths]
        axis.plot(depths, values, marker="o", label=variant)
    axis.set_xscale("log")
    axis.set_xlabel("Top-k depth (log scale)")
    axis.set_ylabel("Gold recall")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("SRA-Bench gold recall over retrieval depth")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gold_recall_depth.png", dpi=200)
    plt.close(fig)


def run(args) -> dict:
    package_root = Path(__file__).resolve().parents[1]
    preset = ENCODER_PRESETS[args.encoder_preset]
    model_path = args.model_path or str((package_root / preset["model_path"]).resolve())
    max_seq_length = args.max_seq_length or int(preset["max_seq_length"])
    repo_root = args.repo_root.resolve()
    cache_dir = args.cache_dir.resolve()
    text_cache_dir = args.text_cache_dir.resolve()
    output_dir = args.output_dir.resolve()

    corpus = json.loads(
        (repo_root / "data" / "bench" / "corpus" / "corpus.json").read_text(encoding="utf-8")
    )
    doc_texts = [skill_text(skill) for skill in corpus]
    skill_ids = [str(skill["skill_id"]) for skill in corpus]
    id_to_index = {skill_id: index for index, skill_id in enumerate(skill_ids)}

    doc_paths = candidate_embedding_cache_paths(
        cache_dir=cache_dir,
        kind="doc_all",
        dataset=None,
        model_path=model_path,
        max_seq_length=max_seq_length,
        ids_or_texts=skill_ids,
    )
    docs = l2_normalize(
        _load_vectors(doc_texts, doc_paths, None, "SRA documents", model_path, max_seq_length)
    )
    stats, basis = load_or_build_text_model(
        doc_texts=doc_texts,
        doc_embeddings=docs,
        cache_dir=text_cache_dir,
        cache_tag=shared_text_model_cache_tag("all", skill_ids),
        max_tokens_per_doc=args.max_tokens_per_doc,
    )
    docs_perp = project_out_background(docs, basis)

    rows = []
    per_dataset = {}
    max_depth = min(max(args.depths), len(skill_ids))
    for dataset in args.datasets:
        examples = collect_queries(
            dataset,
            load_instances(repo_root / "data" / "bench" / "instances" / f"{dataset}.json"),
        )
        query_paths = candidate_embedding_cache_paths(
            cache_dir=cache_dir,
            kind="query",
            dataset=dataset,
            model_path=model_path,
            max_seq_length=max_seq_length,
            ids_or_texts=[example.query for example in examples],
        )
        queries = l2_normalize(
            _load_vectors(
                [example.query for example in examples],
                query_paths,
                "query",
                f"{dataset} queries",
                model_path,
                max_seq_length,
            )
        )
        queries_perp = project_out_background(queries, basis)
        dataset_rows = []
        for start in tqdm(range(0, len(examples), args.score_batch), desc=f"  {dataset}", unit="batch"):
            end = min(start + args.score_batch, len(examples))
            dense_scores = queries[start:end] @ docs.T
            sbc_scores = queries_perp[start:end] @ docs_perp.T
            for offset, example in enumerate(examples[start:end]):
                gold_indices = np.asarray(
                    [id_to_index[gid] for gid in example.gold_ids if gid in id_to_index],
                    dtype=np.int64,
                )
                dense_rank = rank_of_best_gold(dense_scores[offset], gold_indices)
                sbc_rank = rank_of_best_gold(sbc_scores[offset], gold_indices)
                row = {
                    "dataset": dataset,
                    "instance_index": example.instance_index,
                    "dense_rank": dense_rank,
                    "sbc_rank": sbc_rank,
                }
                rows.append(row)
                dataset_rows.append(row)
        per_dataset[dataset] = dataset_rows

    def aggregate(dataset_rows):
        return {
            variant: {
                "recall": {
                    str(depth): float(
                        np.mean([row[f"{variant}_rank"] <= min(depth, len(skill_ids)) for row in dataset_rows])
                    )
                    for depth in args.depths
                },
                "mean_documents_ranked_above_gold_pct": float(
                    np.mean([(row[f"{variant}_rank"] - 1) / len(skill_ids) * 100.0 for row in dataset_rows])
                ),
            }
            for variant in ("dense", "sbc")
        }

    category_summary = {dataset: aggregate(dataset_rows) for dataset, dataset_rows in per_dataset.items()}
    overall = aggregate(rows)
    output = {
        "model_path": model_path,
        "max_seq_length": max_seq_length,
        "corpus_size": len(skill_ids),
        "depths": [min(depth, len(skill_ids)) for depth in args.depths],
        "background_rank": int(stats.background_rank),
        "overall": overall,
        "per_dataset": category_summary,
    }
    _write_csv(output_dir / "gold_ranks.csv", rows)
    _plot(overall, output["depths"], output_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def parse_args(argv=None):
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=package_root / "third_party" / "SR-Agents-main")
    parser.add_argument("--encoder-preset", choices=tuple(ENCODER_PRESETS), default=DEFAULT_ENCODER_PRESET)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=package_root / "cache")
    parser.add_argument("--text-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--depths", nargs="+", type=int, default=[100, 300, 1000, 3000, 10000])
    parser.add_argument("--score-batch", type=int, default=64)
    parser.add_argument("--max-tokens-per-doc", type=int, default=MAX_TOKENS_PER_DOC)
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(args.datasets))
    args.depths = sorted(set(args.depths))
    return args


def main(argv=None) -> None:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
