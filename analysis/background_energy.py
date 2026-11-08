"""Reproduce the query/document background-energy analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from ..cache import candidate_embedding_cache_paths, load_or_encode_embeddings
from ..constants import (
    DEFAULT_DATASETS,
    DEFAULT_ENCODER_PRESET,
    ENCODER_PRESETS,
    MAX_TOKENS_PER_DOC,
)
from ..data import collect_queries, load_instances, skill_text
from ..text_model import load_or_build_text_model, shared_text_model_cache_tag
from ..utils import l2_normalize


def background_energy(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return ||B^T v||_2^2 for every row of a normalized vector matrix."""
    values = l2_normalize(np.asarray(vectors, dtype=np.float32))
    if basis.size == 0:
        return np.zeros(values.shape[0], dtype=np.float64)
    projected = values @ basis
    return np.sum(projected * projected, axis=1, dtype=np.float64)


def distribution(values: Sequence[float]) -> Dict[str, float]:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size == 0:
        return {key: 0.0 for key in ("mean", "median", "p05", "p95", "min", "max")}
    return {
        "mean": float(values_array.mean()),
        "median": float(np.median(values_array)),
        "p05": float(np.percentile(values_array, 5)),
        "p95": float(np.percentile(values_array, 95)),
        "min": float(values_array.min()),
        "max": float(values_array.max()),
    }


def _load_cached_vectors(
    texts: Sequence[str],
    paths,
    prompt_name,
    label: str,
    model_path: str,
    max_seq_length: int,
) -> np.ndarray:
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


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    document_values: np.ndarray,
    per_dataset: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    bins = np.linspace(
        0.0,
        max(
            float(document_values.max(initial=0.0)),
            max((float(values.max(initial=0.0)) for values in per_dataset.values()), default=0.0),
        ),
        50,
    )
    if bins[-1] <= bins[0]:
        bins = np.linspace(0.0, 1.0, 50)
    axis.hist(document_values, bins=bins, alpha=0.65, label="Skill documents", density=True)
    for dataset, values in per_dataset.items():
        axis.hist(values, bins=bins, alpha=0.35, label=f"Queries: {dataset}", density=True)
    axis.set_xlabel("Background energy")
    axis.set_ylabel("Density")
    axis.set_title("SRA-Bench background-energy distributions")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "background_energy_distribution.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    document_mean = float(document_values.mean()) if document_values.size else 0.0
    for dataset, values in per_dataset.items():
        axis.scatter(
            document_mean,
            float(values.mean()) if values.size else 0.0,
            label=dataset,
            s=45,
        )
    limit = max(
        document_mean,
        max((float(values.mean()) for values in per_dataset.values()), default=0.0),
        1e-6,
    )
    axis.plot([0.0, limit * 1.1], [0.0, limit * 1.1], "--", color="black", linewidth=1, label="E_D = E_Q")
    axis.set_xlabel("Mean document energy $E_D$")
    axis.set_ylabel("Mean query energy $E_Q$")
    axis.set_title("Query-document background-energy gap")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "background_energy_gap.png", dpi=200)
    plt.close(fig)


def run(args) -> dict:
    package_root = Path(__file__).resolve().parents[1]
    preset = ENCODER_PRESETS[args.encoder_preset]
    model_path = args.model_path or str((package_root / preset["model_path"]).resolve())
    max_seq_length = args.max_seq_length or int(preset["max_seq_length"])
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    text_cache_dir = args.text_cache_dir.resolve()
    repo_root = args.repo_root.resolve()

    corpus_path = repo_root / "data" / "bench" / "corpus" / "corpus.json"
    instances_dir = repo_root / "data" / "bench" / "instances"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
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
    docs = _load_cached_vectors(
        doc_texts,
        doc_paths,
        None,
        "SRA corpus documents",
        model_path,
        max_seq_length,
    )

    text_tag = shared_text_model_cache_tag("all", skill_ids)
    stats, basis = load_or_build_text_model(
        doc_texts=doc_texts,
        doc_embeddings=docs,
        cache_dir=text_cache_dir,
        cache_tag=text_tag,
        max_tokens_per_doc=args.max_tokens_per_doc,
    )
    doc_energy = background_energy(docs, basis)

    query_rows = []
    dataset_energy: Dict[str, np.ndarray] = {}
    dataset_counts: Dict[str, int] = {}
    for dataset in args.datasets:
        instance_path = instances_dir / f"{dataset}.json"
        if not instance_path.exists():
            raise FileNotFoundError(instance_path)
        examples = collect_queries(dataset, load_instances(instance_path))
        query_texts = [example.query for example in examples]
        query_paths = candidate_embedding_cache_paths(
            cache_dir=cache_dir,
            kind="query",
            dataset=dataset,
            model_path=model_path,
            max_seq_length=max_seq_length,
            ids_or_texts=query_texts,
        )
        queries = _load_cached_vectors(
            query_texts,
            query_paths,
            "query",
            f"{dataset} queries",
            model_path,
            max_seq_length,
        )
        energies = background_energy(queries, basis)
        dataset_energy[dataset] = energies
        dataset_counts[dataset] = len(examples)
        for example, energy in zip(examples, energies):
            query_rows.append(
                {
                    "dataset": dataset,
                    "instance_index": example.instance_index,
                    "query_preview": example.query[:160].replace("\n", " "),
                    "background_energy": float(energy),
                }
            )

    document_rows = [
        {
            "skill_id": skill_id,
            "background_energy": float(energy),
            "document_preview": text[:160].replace("\n", " "),
        }
        for skill_id, text, energy in zip(skill_ids, doc_texts, doc_energy)
    ]
    _write_csv(output_dir / "document_energy.csv", document_rows)
    _write_csv(output_dir / "query_energy.csv", query_rows)
    _plot(doc_energy, dataset_energy, output_dir / "figures")

    summary = {
        "model_path": model_path,
        "max_seq_length": max_seq_length,
        "corpus_size": len(skill_ids),
        "idf_threshold": float(stats.idf_threshold),
        "generic_token_count": len(stats.generic_tokens),
        "background_rank": int(stats.background_rank),
        "background_explained_variance": float(stats.background_explained_variance),
        "background_rank_rule": stats.background_rank_rule,
        "document_background_energy": distribution(doc_energy),
        "query_background_energy": {
            dataset: {
                "count": dataset_counts[dataset],
                **distribution(dataset_energy[dataset]),
                "document_to_query_mean_ratio": (
                    float(doc_energy.mean() / dataset_energy[dataset].mean())
                    if dataset_energy[dataset].size and dataset_energy[dataset].mean() > 0
                    else 0.0
                ),
            }
            for dataset in args.datasets
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "SR-Agents-main",
    )
    parser.add_argument("--encoder-preset", choices=tuple(ENCODER_PRESETS), default=DEFAULT_ENCODER_PRESET)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[1] / "cache")
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "ablations" / "cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "analysis_output" / "background_energy",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-tokens-per-doc", type=int, default=MAX_TOKENS_PER_DOC)
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(args.datasets))
    return args


def main(argv=None) -> None:
    summary = run(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
