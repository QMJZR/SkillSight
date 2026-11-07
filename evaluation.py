"""End-to-end SkillSight evaluation orchestration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .cache import candidate_embedding_cache_paths, load_or_encode_embeddings
from .constants import DEFAULT_VARIANTS, LEXICAL_BETA, MIN_GENERIC_DF
from .data import (
    collect_queries,
    load_instances,
    load_skillrouter_corpus,
    load_skillrouter_examples,
    skill_text,
)
from .encoder import DenseEncoder
from .metrics import (
    build_topk_summary_tables,
    compute_overall_metrics,
    compute_retrieval_metrics,
    print_full_topk_tables,
    print_latency_summary,
    print_overall_summary,
)
from .retriever import SkillSightRetriever
from .text_model import (
    TextStats,
    load_or_build_text_model,
    shared_text_model_cache_key,
    shared_text_model_cache_tag,
)
from .utils import l2_normalize, model_tag


def evaluate_dataset(
    dataset: str,
    examples,
    query_embeddings: np.ndarray,
    doc_texts: Sequence[str],
    doc_embeddings: np.ndarray,
    skill_ids: Sequence[str],
    candidate_k: int,
    top_ks: Sequence[int],
    score_batch: int,
    cache_dir: Path,
    max_tokens_per_doc: int,
    variants: Sequence[str],
    prebuilt_text_model: Optional[Tuple[TextStats, np.ndarray]] = None,
    lexical_beta: float = LEXICAL_BETA,
) -> Dict[str, dict]:
    print(f"[dataset] evaluating {dataset}", flush=True)
    cache_tag = shared_text_model_cache_tag(dataset=dataset, skill_ids=skill_ids)
    retriever = SkillSightRetriever(
        doc_texts=doc_texts,
        doc_embeddings=doc_embeddings,
        skill_ids=skill_ids,
        candidate_k=candidate_k,
        score_batch=score_batch,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        prebuilt_text_model=prebuilt_text_model,
        lexical_beta=lexical_beta,
    )
    query_texts = [ex.query for ex in examples]
    results = retriever.score(
        query_texts=query_texts,
        query_embeddings=query_embeddings,
        final_k=max(top_ks),
        variants=variants,
    )
    dataset_metrics: Dict[str, dict] = {}
    for variant, result in results.items():
        dataset_metrics[variant] = compute_retrieval_metrics(result, examples, skill_ids, top_ks)
    return dataset_metrics


def get_or_build_shared_text_model(
    dataset: str,
    payload: dict,
    cache_dir: Path,
    max_tokens_per_doc: int,
    text_model_cache: Dict[Tuple[object, ...], Tuple[TextStats, np.ndarray]],
) -> Tuple[TextStats, np.ndarray]:
    doc_texts = payload["doc_texts"]
    doc_embeddings = l2_normalize(payload["doc_embeddings"])
    skill_ids = payload["skill_ids"]
    key = shared_text_model_cache_key(doc_texts, doc_embeddings)

    cache_tag = shared_text_model_cache_tag(dataset=dataset, skill_ids=skill_ids)
    if key not in text_model_cache:
        print(f"[skillsight] prepare shared text model: {cache_tag}", flush=True)
        text_model_cache[key] = load_or_build_text_model(
            doc_texts=doc_texts,
            doc_embeddings=doc_embeddings,
            cache_dir=cache_dir,
            cache_tag=cache_tag,
            max_tokens_per_doc=max_tokens_per_doc,
        )
    else:
        print(f"[skillsight] shared text model already prepared: {cache_tag}", flush=True)

    return text_model_cache[key]


def serializable_config(args) -> dict:
    out = vars(args).copy()
    for key in ("repo_root", "cache_dir", "output_dir", "output_json", "skillrouter_root"):
        if out.get(key) is not None:
            out[key] = str(out[key])
    return out


def run_evaluation(args) -> dict:
    """Run the full SkillSight evaluation from parsed CLI args."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"[warn] {args.device} requested but CUDA unavailable; using cpu", flush=True)
        args.device = "cpu"

    if args.primary_k not in args.top_k:
        args.top_k = sorted(set(args.top_k + [args.primary_k]))
    args.top_k = sorted(set(args.top_k))
    args.variants = list(dict.fromkeys(getattr(args, "variants", None) or DEFAULT_VARIANTS))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.output_dir / "performance_metrics.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    encoder: Optional[DenseEncoder] = None
    dataset_payloads: Dict[str, dict] = {}

    if args.benchmark == "skillrouter":
        encoder = _load_skillrouter_payloads(args, dataset_payloads)
    else:
        encoder = _load_sra_payloads(args, dataset_payloads)

    text_model_cache: Dict[Tuple[object, ...], Tuple[TextStats, np.ndarray]] = {}
    text_model_cache_dir = args.output_dir / "cache"

    per_category: Dict[str, Dict[str, dict]] = {}
    for dataset, payload in dataset_payloads.items():
        prebuilt_text_model = get_or_build_shared_text_model(
            dataset=dataset,
            payload=payload,
            cache_dir=text_model_cache_dir,
            max_tokens_per_doc=args.max_tokens_per_doc,
            text_model_cache=text_model_cache,
        )
        per_category[dataset] = evaluate_dataset(
            dataset=dataset,
            examples=payload["examples"],
            query_embeddings=payload["query_embeddings"],
            doc_texts=payload["doc_texts"],
            doc_embeddings=payload["doc_embeddings"],
            skill_ids=payload["skill_ids"],
            candidate_k=args.candidate_k,
            top_ks=args.top_k,
            score_batch=args.score_batch,
            cache_dir=text_model_cache_dir,
            max_tokens_per_doc=args.max_tokens_per_doc,
            variants=args.variants,
            prebuilt_text_model=prebuilt_text_model,
            lexical_beta=getattr(args, "lexical_beta", LEXICAL_BETA),
        )

    overall = compute_overall_metrics(per_category, args.variants, args.top_k)

    summary_tables = build_topk_summary_tables(
        per_category=per_category,
        overall=overall,
        variants=args.variants,
        top_ks=args.top_k,
        metrics=("recall", "hit", "mrr"),
        overall_average="micro",
    )

    output = {
        "config": serializable_config(args),
        "method": {
            "name": "SkillSight",
            "lexical_beta": getattr(args, "lexical_beta", LEXICAL_BETA),
            "min_generic_df": MIN_GENERIC_DF,
            "max_tokens_per_doc": args.max_tokens_per_doc,
        },
        "variants": list(args.variants),
        "top_k": [int(k) for k in args.top_k],
        "primary_k": int(args.primary_k),
        "per_category": per_category,
        "overall": overall,
        "summary_tables": summary_tables,
    }

    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print_overall_summary(overall, args.top_k)
    print_latency_summary(overall)
    print_full_topk_tables(
        summary_tables=summary_tables,
        top_ks=args.top_k,
        metrics=("recall", "hit", "mrr"),
        overall_average="micro",
    )
    print(f"\n[done] wrote JSON performance metrics: {output_json}", flush=True)
    return output


def _load_sra_payloads(args, dataset_payloads: Dict[str, dict]) -> Optional[DenseEncoder]:
    """Load the SR-Agents corpus + per-dataset queries (full corpus pool)."""
    corpus_path = args.repo_root / "data" / "bench" / "corpus" / "corpus.json"
    instances_dir = args.repo_root / "data" / "bench" / "instances"
    if not corpus_path.exists():
        sys.exit(f"corpus not found: {corpus_path}")

    print(f"[load] corpus: {corpus_path}", flush=True)
    corpus_all = json.loads(corpus_path.read_text())
    full_doc_texts = [skill_text(skill) for skill in corpus_all]
    full_doc_ids = [str(skill["skill_id"]) for skill in corpus_all]
    print(f"[load] corpus size: {len(full_doc_ids)}", flush=True)
    print(f"[config] benchmark=sra datasets={args.datasets} variants={args.variants}", flush=True)
    print(f"[config] top_k={args.top_k} candidate_k={args.candidate_k}", flush=True)

    encoder: Optional[DenseEncoder] = None
    full_doc_cache_paths = candidate_embedding_cache_paths(
        cache_dir=args.cache_dir,
        kind="doc_all",
        dataset=None,
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        ids_or_texts=full_doc_ids,
    )
    full_doc_embeddings, encoder = load_or_encode_embeddings(
        encoder=encoder,
        texts=full_doc_texts,
        cache_paths=full_doc_cache_paths,
        batch_size=args.doc_batch,
        prompt_name=None,
        label="full corpus docs",
        model_path=args.model_path,
        device=args.device,
        max_seq_length=args.max_seq_length,
        allow_encode=not args.cache_only,
    )

    for dataset in args.datasets:
        instance_path = instances_dir / f"{dataset}.json"
        if not instance_path.exists():
            print(f"[warn] missing instances: {instance_path}, skip", flush=True)
            continue
        instances = load_instances(instance_path)
        examples = collect_queries(dataset, instances)
        if not examples:
            print(f"[warn] no evaluable examples for {dataset}, skip", flush=True)
            continue

        query_texts = [ex.query for ex in examples]
        q_cache_paths = candidate_embedding_cache_paths(
            cache_dir=args.cache_dir,
            kind="query",
            dataset=dataset,
            model_path=args.model_path,
            max_seq_length=args.max_seq_length,
            ids_or_texts=query_texts,
        )
        query_embeddings, encoder = load_or_encode_embeddings(
            encoder=encoder,
            texts=query_texts,
            cache_paths=q_cache_paths,
            batch_size=args.query_batch,
            prompt_name="query",
            label=f"queries {dataset}",
            model_path=args.model_path,
            device=args.device,
            max_seq_length=args.max_seq_length,
            allow_encode=not args.cache_only,
        )
        dataset_payloads[dataset] = {
            "examples": examples,
            "query_embeddings": query_embeddings,
            "doc_texts": full_doc_texts,
            "doc_embeddings": full_doc_embeddings,
            "skill_ids": full_doc_ids,
        }
    return encoder


def _load_skillrouter_payloads(args, dataset_payloads: Dict[str, dict]) -> Optional[DenseEncoder]:
    """Load the SkillRouter-Eval-Core 'all' pool (deduped easy+hard) + tasks."""
    root = args.skillrouter_root
    if not root.exists():
        sys.exit(f"SkillRouter-Eval-Core root not found: {root}")

    print(f"[load] SkillRouter-Eval-Core: {root} (pool=all)", flush=True)
    doc_texts, skill_ids = load_skillrouter_corpus(root)
    examples = load_skillrouter_examples(
        root,
        exclude_generic_only=not args.include_generic_only,
        include_excluded=getattr(args, "include_excluded", False),
    )
    print(f"[load] corpus={len(skill_ids)} queries={len(examples)}", flush=True)
    print(f"[config] benchmark=skillrouter variants={args.variants}", flush=True)
    print(f"[config] top_k={args.top_k} candidate_k={args.candidate_k}", flush=True)

    tag = model_tag(args.model_path)
    encoder: Optional[DenseEncoder] = None

    doc_cache_paths = [args.cache_dir / f"skillrouter_all_{tag}_{args.max_seq_length}.npy"]
    doc_embeddings, encoder = load_or_encode_embeddings(
        encoder=encoder,
        texts=doc_texts,
        cache_paths=doc_cache_paths,
        batch_size=args.doc_batch,
        prompt_name=None,
        label="skillrouter all docs",
        model_path=args.model_path,
        device=args.device,
        max_seq_length=args.max_seq_length,
        allow_encode=not args.cache_only,
    )

    query_texts = [ex.query for ex in examples]
    q_cache_paths = [args.cache_dir / f"skillrouter_query_all_{tag}_{args.max_seq_length}.npy"]
    query_embeddings, encoder = load_or_encode_embeddings(
        encoder=encoder,
        texts=query_texts,
        cache_paths=q_cache_paths,
        batch_size=args.query_batch,
        prompt_name="query",
        label="skillrouter all queries",
        model_path=args.model_path,
        device=args.device,
        max_seq_length=args.max_seq_length,
        allow_encode=not args.cache_only,
    )
    dataset_payloads["skillrouter_core"] = {
        "examples": examples,
        "query_embeddings": query_embeddings,
        "doc_texts": doc_texts,
        "doc_embeddings": doc_embeddings,
        "skill_ids": skill_ids,
    }
    return encoder
