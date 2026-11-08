"""Reproduce the generic-token and gold-skill-rate analysis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..cache import candidate_embedding_cache_paths, load_or_encode_embeddings
from ..constants import (
    DEFAULT_DATASETS,
    DEFAULT_ENCODER_PRESET,
    ENCODER_PRESETS,
    MAX_TOKENS_PER_DOC,
    SKILLROUTER_TIERS,
)
from ..data import collect_queries, load_instances, load_skillrouter_corpus, load_skillrouter_examples, skill_text
from ..text_model import load_or_build_text_model, shared_text_model_cache_tag


def _load_vectors(texts, paths, model_path: str, max_seq_length: int, label: str | None = None):
    vectors, _ = load_or_encode_embeddings(
        encoder=None,
        texts=texts,
        cache_paths=paths,
        batch_size=1,
        prompt_name=None,
        label=label or "corpus",
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


def _resolve(args):
    package_root = Path(__file__).resolve().parents[1]
    preset = ENCODER_PRESETS[args.encoder_preset]
    model_path = args.model_path or str((package_root / preset["model_path"]).resolve())
    max_seq_length = args.max_seq_length or int(preset["max_seq_length"])
    return package_root, model_path, max_seq_length


def _load_benchmark(args, package_root: Path):
    if args.benchmark == "sra":
        root = args.repo_root.resolve()
        corpus = json.loads(
            (root / "data" / "bench" / "corpus" / "corpus.json").read_text(encoding="utf-8")
        )
        texts = [skill_text(skill) for skill in corpus]
        ids = [str(skill["skill_id"]) for skill in corpus]
        gold_ids = set()
        for dataset in args.datasets:
            examples = collect_queries(
                dataset,
                load_instances(root / "data" / "bench" / "instances" / f"{dataset}.json"),
            )
            gold_ids.update(gid for example in examples for gid in example.gold_ids)
        tier_tag = None
    else:
        tiers = tuple(args.skillrouter_tiers)
        texts, ids = load_skillrouter_corpus(args.skillrouter_root.resolve(), tiers=tiers)
        examples = load_skillrouter_examples(
            args.skillrouter_root.resolve(),
            exclude_generic_only=not args.include_generic_only,
            include_excluded=args.include_excluded,
        )
        gold_ids = {gid for example in examples for gid in example.gold_ids}
        tier_tag = "all" if set(tiers) == {"easy", "hard"} else "_".join(tiers)
    return texts, ids, gold_ids, tier_tag


def _plot(rows, output_dir: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0.0, 1.0, 21)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for group, label in (("generic", "Generic tokens"), ("other", "Other tokens")):
        values = [row["gold_skill_rate"] for row in rows if row["group"] == group]
        counts, _ = np.histogram(values, bins=edges)
        axis.plot(centers[counts > 0], counts[counts > 0], marker="o", label=label)
    axis.set_yscale("log")
    axis.set_xlabel("Gold-skill rate")
    axis.set_ylabel("Token count (log scale)")
    axis.set_title(title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "token_gold_rate.png", dpi=200)
    plt.close(fig)


def run(args) -> dict:
    package_root, model_path, max_seq_length = _resolve(args)
    texts, ids, gold_ids, tier_tag = _load_benchmark(args, package_root)
    cache_dir = args.cache_dir.resolve()
    text_cache_dir = args.text_cache_dir.resolve()
    output_dir = args.output_dir.resolve()

    dataset_name = (
        "sra"
        if args.benchmark == "sra"
        else f"skillrouter_{tier_tag}"
    )
    doc_paths = candidate_embedding_cache_paths(
        cache_dir=cache_dir,
        kind="doc_all",
        dataset=None if args.benchmark == "sra" else dataset_name,
        model_path=model_path,
        max_seq_length=max_seq_length,
        ids_or_texts=ids,
    )
    docs = _load_vectors(texts, doc_paths, model_path, max_seq_length, label=f"{dataset_name} corpus")
    stats, _ = load_or_build_text_model(
        doc_texts=texts,
        doc_embeddings=docs,
        cache_dir=text_cache_dir,
        cache_tag=shared_text_model_cache_tag(dataset_name, ids),
        max_tokens_per_doc=args.max_tokens_per_doc,
    )

    id_to_index = {skill_id: index for index, skill_id in enumerate(ids)}
    gold_token_counts = Counter()
    matched_gold_ids = set(gold_ids) & set(ids)
    for skill_id in matched_gold_ids:
        gold_token_counts.update(stats.doc_tokens[id_to_index[skill_id]])

    rows = []
    for token, df in stats.df.items():
        gold_count = int(gold_token_counts.get(token, 0))
        rows.append(
            {
                "token": token,
                "df": int(df),
                "idf": float(stats.idf[token]),
                "gold_skill_count": gold_count,
                "gold_skill_rate": float(gold_count / df) if df else 0.0,
                "group": "generic" if token in stats.generic_tokens else "other",
            }
        )
    rows.sort(key=lambda row: (-row["df"], row["token"]))
    _write_csv(output_dir / "token_evidence.csv", rows)
    _plot(rows, output_dir / "figures", f"{dataset_name} token evidence")

    def group_values(group: str):
        return [row["gold_skill_rate"] for row in rows if row["group"] == group]

    def summarize(values):
        if not values:
            return {"count": 0, "mean": 0.0, "median": 0.0, "max": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "max": float(array.max()),
        }

    summary = {
        "benchmark": args.benchmark,
        "dataset": dataset_name,
        "model_path": model_path,
        "max_seq_length": max_seq_length,
        "corpus_size": len(ids),
        "gold_skill_count": len(matched_gold_ids),
        "gold_skill_oov": len(set(gold_ids) - set(ids)),
        "idf_threshold": float(stats.idf_threshold),
        "token_count": len(rows),
        "generic": summarize(group_values("generic")),
        "other": summarize(group_values("other")),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args(argv=None):
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["sra", "skillrouter"], default="sra")
    parser.add_argument("--repo-root", type=Path, default=package_root / "third_party" / "SR-Agents-main")
    parser.add_argument("--skillrouter-root", type=Path, default=package_root / "SkillRouter-Eval-Core")
    parser.add_argument("--skillrouter-tiers", nargs="+", choices=SKILLROUTER_TIERS, default=list(SKILLROUTER_TIERS))
    parser.add_argument("--include-generic-only", action="store_true")
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument("--encoder-preset", choices=tuple(ENCODER_PRESETS), default=DEFAULT_ENCODER_PRESET)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=package_root / "cache")
    parser.add_argument("--text-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-tokens-per-doc", type=int, default=MAX_TOKENS_PER_DOC)
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(args.datasets))
    args.skillrouter_tiers = list(dict.fromkeys(args.skillrouter_tiers))
    return args


def main(argv=None) -> None:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
