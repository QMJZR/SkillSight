"""SRA-Bench sensitivity analysis for SkillSight hyperparameters."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..cache import candidate_embedding_cache_paths, load_or_encode_embeddings
from ..constants import (
    DEFAULT_DATASETS,
    DEFAULT_ENCODER_PRESET,
    ENCODER_PRESETS,
    MAX_TOKENS_PER_DOC,
    VARIANT_SKILLSIGHT,
)
from ..data import collect_queries, load_instances, skill_text
from ..metrics import compute_overall_metrics, compute_retrieval_metrics
from ..retriever import SkillSightRetriever
from ..text_model import load_or_build_text_model, shared_text_model_cache_tag
from ..utils import l2_normalize


def ranked_background_basis(stats, rank: int) -> np.ndarray:
    """Rebuild the leading-`rank` background basis from token prototypes."""
    if stats.generic_token_matrix.size == 0:
        return np.zeros((stats.corpus_mean.shape[0], 0), dtype=np.float32)
    matrix = l2_normalize(
        np.vstack([stats.generic_token_matrix, stats.corpus_mean[None, :]]).astype(np.float32)
    )
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    effective_rank = max(1, min(rank, vt.shape[0], stats.corpus_mean.shape[0]))
    basis, _ = np.linalg.qr(vt[:effective_rank].T)
    return basis[:, :effective_rank].astype(np.float32)


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


def _evaluate(retriever, examples_by_dataset, query_embeddings_by_dataset, top_ks):
    per_dataset = {}
    for dataset, examples in examples_by_dataset.items():
        results = retriever.score(
            query_texts=[example.query for example in examples],
            query_embeddings=query_embeddings_by_dataset[dataset],
            final_k=max(top_ks),
            variants=[VARIANT_SKILLSIGHT],
        )
        per_dataset[dataset] = {
            VARIANT_SKILLSIGHT: compute_retrieval_metrics(
                results[VARIANT_SKILLSIGHT],
                examples,
                retriever.skill_ids,
                top_ks,
            )
        }
    overall = compute_overall_metrics(per_dataset, [VARIANT_SKILLSIGHT], top_ks)["micro"][VARIANT_SKILLSIGHT]
    return per_dataset, overall


def _metric_row(parameter: str, value, scope: str, metrics, top_ks):
    row = {parameter: value, "scope": scope}
    for depth in top_ks:
        values = metrics["topk"][str(depth)]
        row[f"recall@{depth}"] = values["recall"]
        row[f"hit@{depth}"] = values["hit"]
        row[f"mrr@{depth}"] = values["mrr"]
    return row


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary, output_dir: Path, top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    specs = (
        ("candidate_k", "Candidate K"),
        ("beta", "Lexical beta"),
        ("background_rank", "Background rank"),
    )
    for axis, (name, label) in zip(axes, specs):
        points = summary[name]
        axis.plot(
            [point["value"] for point in points],
            [point["overall"][f"recall@{top_k}"] for point in points],
            marker="o",
        )
        axis.set_xlabel(label)
        axis.set_ylabel(f"Recall@{top_k}")
        axis.grid(alpha=0.25)
    fig.suptitle("SkillSight SRA-Bench sensitivity")
    fig.tight_layout()
    fig.savefig(output_dir / "hyperparameter_sensitivity.png", dpi=200)
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
    stats, default_basis = load_or_build_text_model(
        doc_texts=doc_texts,
        doc_embeddings=docs,
        cache_dir=text_cache_dir,
        cache_tag=shared_text_model_cache_tag("all", skill_ids),
        max_tokens_per_doc=args.max_tokens_per_doc,
    )

    examples_by_dataset = {}
    query_embeddings_by_dataset = {}
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
        query_embeddings = _load_vectors(
            [example.query for example in examples],
            query_paths,
            "query",
            f"{dataset} queries",
            model_path,
            max_seq_length,
        )
        examples_by_dataset[dataset] = examples
        query_embeddings_by_dataset[dataset] = query_embeddings

    def build_retriever(candidate_k: int, beta: float, basis: np.ndarray):
        return SkillSightRetriever(
            doc_texts=doc_texts,
            doc_embeddings=docs,
            skill_ids=skill_ids,
            candidate_k=candidate_k,
            score_batch=args.score_batch,
            prebuilt_text_model=(stats, basis),
            lexical_beta=beta,
        )

    output = {"candidate_k": [], "beta": [], "background_rank": []}
    detail_rows = []

    for candidate_k in args.candidate_ks:
        retriever = build_retriever(candidate_k, 1.0, default_basis)
        per_dataset, overall = _evaluate(
            retriever, examples_by_dataset, query_embeddings_by_dataset, args.top_ks
        )
        output["candidate_k"].append(
            {"value": candidate_k, "overall": {f"{m}@{k}": overall["topk"][str(k)][m] for m in ("recall", "hit", "mrr") for k in args.top_ks}}
        )
        detail_rows.append(_metric_row("candidate_k", candidate_k, "overall", overall, args.top_ks))
        for dataset, rows in per_dataset.items():
            detail_rows.append(
                _metric_row(
                    "candidate_k",
                    candidate_k,
                    dataset,
                    rows[VARIANT_SKILLSIGHT],
                    args.top_ks,
                )
            )

    for beta in args.betas:
        retriever = build_retriever(300, beta, default_basis)
        per_dataset, overall = _evaluate(
            retriever, examples_by_dataset, query_embeddings_by_dataset, args.top_ks
        )
        output["beta"].append(
            {"value": beta, "overall": {f"{m}@{k}": overall["topk"][str(k)][m] for m in ("recall", "hit", "mrr") for k in args.top_ks}}
        )
        detail_rows.append(_metric_row("beta", beta, "overall", overall, args.top_ks))
        for dataset, rows in per_dataset.items():
            detail_rows.append(
                _metric_row("beta", beta, dataset, rows[VARIANT_SKILLSIGHT], args.top_ks)
            )

    for rank in args.background_ranks:
        basis = ranked_background_basis(stats, rank)
        retriever = build_retriever(300, 1.0, basis)
        per_dataset, overall = _evaluate(
            retriever, examples_by_dataset, query_embeddings_by_dataset, args.top_ks
        )
        output["background_rank"].append(
            {"value": rank, "effective_rank": int(basis.shape[1]), "overall": {f"{m}@{k}": overall["topk"][str(k)][m] for m in ("recall", "hit", "mrr") for k in args.top_ks}}
        )
        detail_rows.append(_metric_row("background_rank", rank, "overall", overall, args.top_ks))
        for dataset, rows in per_dataset.items():
            detail_rows.append(
                _metric_row(
                    "background_rank",
                    rank,
                    dataset,
                    rows[VARIANT_SKILLSIGHT],
                    args.top_ks,
                )
            )

    output["default_background_rank"] = int(default_basis.shape[1])
    output["top_ks"] = args.top_ks
    _write_csv(output_dir / "hyperparameter_details.csv", detail_rows)
    _plot(output, output_dir / "figures", args.primary_k)
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
    parser.add_argument("--candidate-ks", nargs="+", type=int, default=[50, 100, 200, 300, 500])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--background-ranks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--score-batch", type=int, default=64)
    parser.add_argument("--max-tokens-per-doc", type=int, default=MAX_TOKENS_PER_DOC)
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(args.datasets))
    args.candidate_ks = sorted(set(args.candidate_ks))
    args.betas = sorted(set(args.betas))
    args.background_ranks = sorted(set(args.background_ranks))
    args.top_ks = sorted(set(args.top_ks))
    if args.primary_k not in args.top_ks:
        args.top_ks = sorted(set(args.top_ks + [args.primary_k]))
    return args


def main(argv=None) -> None:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
