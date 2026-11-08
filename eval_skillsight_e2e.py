#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate SkillSight retrieval END-TO-END on SRA-Bench (in-repo edition).

This is the project-local adaptation of the original SRA-bench script. All
paths are resolved *relative to this file* so the pipeline runs from wherever
the ``SkillSight`` package lives, with no hard-coded machine paths.

Unlike the recall/hit/mrr studies (which only score the ranking), this script
measures the *task* accuracy of an agent that is handed the top-K skills
SkillSight retrieves for each instance.

Pipeline (three stages, matching the official SR-Agents CLI):

    Stage 1 — RETRIEVE : run the SkillSight retriever over the full skill corpus
                          and save the top-K skills per instance as a retrieval
                          results JSON (``sragents`` schema).
    Stage 2 — INFER    : sragents infer --provider topk --provider-arg source=<json>
                          --provider-arg k=K   (engine=react for ToolQA, else direct)
    Stage 3 — EVALUATE : sragents evaluate

Layout (all relative to this file):
    <pkg>/                              this package (imported as ``SkillSight``)
    <pkg>/third_party/SR-Agents-main    the SR-Agents checkout (src + data/bench)
    <pkg>/results/skillsight_e2e/       outputs
    <pkg>/cache/                        shared embedding cache

The corpus ships zipped (``corpus.json.zip``); stage 1 auto-extracts it on
first use.

task_latency = retrieval_latency (amortised) + inference_latency

Output:
  results/skillsight_e2e/retrieval/skillsight-{dataset}.json
  results/skillsight_e2e/inference/{dataset}-{model}-skillsight_top{K}.jsonl
  results/skillsight_e2e/eval/{dataset}-{model}-skillsight_top{K}.json
  results/skillsight_e2e/summary_skillsight_e2e.csv
  results/skillsight_e2e/per_sample_skillsight_e2e.csv

Example:
  python -m SkillSight.eval_skillsight_e2e \\
    --model Qwen3-4B-Instruct-2507 \\
    --api-base http://127.0.0.1:8001/v1 \\
    --top-ks 3 \\
    --no-start-vllm \\
    --overwrite
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- resolve repo layout RELATIVE to this file ------------------------------
HERE = Path(__file__).resolve().parent            # the SkillSight package dir
PACKAGE_PARENT = HERE.parent                       # so ``import SkillSight`` works
SR_AGENTS_ROOT = HERE / "third_party" / "SR-Agents-main"
SR_AGENTS_SRC = SR_AGENTS_ROOT / "src"
for p in (PACKAGE_PARENT, SR_AGENTS_SRC):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from .constants import LEXICAL_BETA
except ImportError:  # Direct script execution in the upstream checkout.
    from SkillSight.constants import LEXICAL_BETA


DEFAULT_DATASETS = [
    "theoremqa",
    "logicbench",
    "toolqa",
    "medcalcbench",
    "champ",
    "bigcodebench",
]

DEFAULT_TOP_KS = [3]

# Inference concurrency (parallel infer workers) per engine.
WORKERS_BY_ENGINE = {"direct": 32, "react": 8}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def cell(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_corpus_extracted() -> Path:
    """Return the path to corpus.json, extracting it from the shipped zip once."""
    corpus_dir = SR_AGENTS_ROOT / "data" / "bench" / "corpus"
    corpus_path = corpus_dir / "corpus.json"
    if corpus_path.exists():
        return corpus_path
    zip_path = corpus_dir / "corpus.json.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"neither {corpus_path} nor {zip_path} exists; cannot load skill corpus."
        )
    print(f"[corpus] extracting {zip_path.name} -> {corpus_path} ...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"extracted {zip_path} but {corpus_path} still missing (unexpected archive layout)."
        )
    return corpus_path


def load_retrieval_latency(retrieval_path: Path) -> float:
    """Return the amortised per-query retrieval latency (seconds) for a dataset."""
    try:
        obj = load_json(retrieval_path)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not read retrieval latency from {retrieval_path}: {e}",
              flush=True)
        return 0.0
    extra = obj.get("extra", {}) or obj.get("metadata", {}).get("extra", {}) or {}
    per_query = extra.get("retrieval_latency_per_query_s")
    if isinstance(per_query, (int, float)):
        return float(per_query)
    print(f"[warn] no retrieval_latency_per_query_s in {retrieval_path}; "
          f"re-run retrieval (drop --reuse-retrieval) to record it. "
          f"Task latency will equal inference latency only.", flush=True)
    return 0.0


def load_inference_stats(infer_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Read per-instance latency & input-token stats from an inference JSONL."""
    per_instance: Dict[str, Dict[str, Any]] = {}
    latencies: List[float] = []
    input_tokens: List[int] = []

    if not infer_path.exists():
        print(f"[warn] inference file missing, no latency/token stats: {infer_path}",
              flush=True)
        return per_instance, {}

    with infer_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = str(rec.get("instance_id", ""))
            meta = rec.get("meta", {}) or {}
            per_instance[iid] = meta
            if isinstance(meta.get("latency_s"), (int, float)):
                latencies.append(float(meta["latency_s"]))
            if isinstance(meta.get("input_tokens"), (int, float)):
                input_tokens.append(int(meta["input_tokens"]))

    averages: Dict[str, float] = {}
    if latencies:
        averages["avg_latency_s"] = round(sum(latencies) / len(latencies), 4)
        averages["total_latency_s"] = round(sum(latencies), 4)
        averages["n_latency_samples"] = len(latencies)
    if input_tokens:
        averages["avg_input_tokens"] = round(sum(input_tokens) / len(input_tokens), 2)
        averages["total_input_tokens"] = sum(input_tokens)
        averages["n_token_samples"] = len(input_tokens)
    return per_instance, averages


def get_instance_id(inst: Dict[str, Any]) -> str:
    for key in ("instance_id", "id", "qid", "question_id"):
        if key in inst:
            return str(inst[key])
    return ""


def get_question(inst: Dict[str, Any]) -> str:
    for key in ("question", "query", "problem", "prompt", "instruction"):
        if inst.get(key):
            return cell(inst[key])
    return ""


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print("\n[run]", shell_join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def wait_for_openai_server(api_base: str, timeout: int = 900) -> None:
    """Block until the OpenAI-compatible /models endpoint answers 2xx."""
    api_base = api_base.rstrip("/")
    url = api_base + "/models"
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    deadline = time.time() + timeout
    print(f"[check] waiting for vLLM server: {url}", flush=True)

    while time.time() < deadline:
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {api_key}"}, method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    print("[check] vLLM server is ready.", flush=True)
                    return
        except Exception as e:  # noqa: BLE001
            print(f"[check] not ready yet: {type(e).__name__}: {e}", flush=True)
            time.sleep(5)

    raise TimeoutError(f"vLLM server not ready after {timeout}s: {url}")


def maybe_launch_vllm(project_root: Path, api_base: str) -> None:
    """Start vLLM via vllm.sh if present, else leave the server to the user."""
    vllm_sh = project_root / "vllm.sh"
    if not vllm_sh.exists():
        print(f"[vllm] {vllm_sh} not found; assuming server is managed externally.",
              flush=True)
        return
    print("[vllm] launching server via vllm.sh ...", flush=True)
    subprocess.run(["bash", str(vllm_sh)], cwd=str(project_root), check=True)


# ---------------------------------------------------------------------------
# Stage 1: SkillSight retrieval -> sragents-schema retrieval results JSON
# ---------------------------------------------------------------------------

def build_subprocess_env() -> Dict[str, str]:
    """Env for the sragents infer/evaluate subprocesses."""
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "EMPTY")

    # Ensure the *in-repo* sragents wins over any globally-installed copy.
    src_path = str(SR_AGENTS_SRC)
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path + (":" + old if old else "")

    # ToolQA's react engine builds a local sentence-transformers index; point it
    # at a cached model and force offline so it never hangs on huggingface.co.
    local_embed = env.get("TOOLQA_EMBED_MODEL", "./models/all-mpnet-base-v2")
    if Path(local_embed).exists():
        env["TOOLQA_EMBED_MODEL"] = local_embed
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Keep per-encode thread pools bounded under many infer workers.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def skillsight_retrieve_dataset(
    dataset: str,
    max_k: int,
    args: argparse.Namespace,
    encoder_holder: List[Any],
) -> Tuple[Path, Dict[str, float]]:
    """Run SkillSight over the full corpus for one dataset and dump top-``max_k``."""
    import numpy as np

    from SkillSight.cache import (
        candidate_embedding_cache_paths,
        load_or_encode_embeddings,
    )
    from SkillSight.constants import VARIANT_SKILLSIGHT
    from SkillSight.data import get_gold, load_instances, skill_text
    from SkillSight.retriever import SkillSightRetriever
    from SkillSight.text_model import (
        load_or_build_text_model,
        shared_text_model_cache_tag,
    )
    from SkillSight.utils import l2_normalize

    from sragents.retrieve.schema import RetrievalRecord, RetrievalResults

    corpus_path = ensure_corpus_extracted()
    instances_path = SR_AGENTS_ROOT / "data" / "bench" / "instances" / f"{dataset}.json"
    if not instances_path.exists():
        raise FileNotFoundError(f"instances not found: {instances_path}")

    # ---- corpus (documents) ----
    corpus = json.loads(corpus_path.read_text())
    doc_texts = [skill_text(s) for s in corpus]
    skill_ids = [str(s["skill_id"]) for s in corpus]

    # ---- queries: ALL instances with a question ----
    instances = load_instances(instances_path)
    query_meta: List[Tuple[str, List[str]]] = []  # (instance_id, gold_ids)
    query_texts: List[str] = []
    for inst in instances:
        q = get_question(inst)
        if not q:
            continue
        query_meta.append((get_instance_id(inst), get_gold(inst)))
        query_texts.append(q)

    print(f"[skillsight] {dataset}: corpus={len(skill_ids)} queries={len(query_texts)}",
          flush=True)

    encoder = encoder_holder[0] if encoder_holder else None

    retrieve_t0 = time.perf_counter()

    # ---- encode docs (shared full-corpus cache) + queries ----
    doc_cache = candidate_embedding_cache_paths(
        cache_dir=args.cache_dir, kind="doc_all", dataset=None,
        model_path=args.embed_model_path, max_seq_length=args.max_seq_length,
        ids_or_texts=skill_ids,
    )
    doc_embeddings, encoder = load_or_encode_embeddings(
        encoder=encoder, texts=doc_texts, cache_paths=doc_cache,
        batch_size=args.doc_batch, prompt_name=None, label="full corpus docs",
        model_path=args.embed_model_path, device=args.device,
        max_seq_length=args.max_seq_length, allow_encode=True,
    )

    q_cache = candidate_embedding_cache_paths(
        cache_dir=args.cache_dir, kind="query", dataset=f"{dataset}_e2e",
        model_path=args.embed_model_path, max_seq_length=args.max_seq_length,
        ids_or_texts=query_texts,
    )
    query_embeddings, encoder = load_or_encode_embeddings(
        encoder=encoder, texts=query_texts, cache_paths=q_cache,
        batch_size=args.query_batch, prompt_name="query",
        label=f"queries {dataset} (e2e)", model_path=args.embed_model_path,
        device=args.device, max_seq_length=args.max_seq_length, allow_encode=True,
    )
    if encoder is not None:
        encoder_holder[:] = [encoder]

    # ---- SkillSight generic text model (shared, honours --max-tokens-per-doc) ----
    text_model_cache_dir = args.output_root / "retrieval" / "cache"
    text_model_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = shared_text_model_cache_tag(dataset=dataset, skill_ids=skill_ids)
    prebuilt = load_or_build_text_model(
        doc_texts=doc_texts,
        doc_embeddings=l2_normalize(doc_embeddings),
        cache_dir=text_model_cache_dir,
        cache_tag=cache_tag,
        max_tokens_per_doc=args.max_tokens_per_doc,
    )

    # ---- score ----
    retriever = SkillSightRetriever(
        doc_texts=doc_texts,
        doc_embeddings=doc_embeddings,
        skill_ids=skill_ids,
        candidate_k=args.candidate_k,
        score_batch=args.score_batch,
        cache_dir=text_model_cache_dir,
        cache_tag=cache_tag,
        prebuilt_text_model=prebuilt,
        lexical_beta=args.lexical_beta,
    )
    results = retriever.score(
        query_texts=query_texts,
        query_embeddings=query_embeddings,
        final_k=max_k,
    )
    result = results[VARIANT_SKILLSIGHT]
    skill_ids_arr = np.asarray(skill_ids)

    retrieval_wall_s = time.perf_counter() - retrieve_t0
    n_q = len(query_texts)
    retrieval_latency_per_query = (retrieval_wall_s / n_q) if n_q else 0.0
    print(f"[skillsight] {dataset}: retrieval wall={retrieval_wall_s:.2f}s over "
          f"{n_q} queries -> {retrieval_latency_per_query:.4f}s/query (amortised)",
          flush=True)

    # ---- assemble sragents-schema records ----
    records: List[RetrievalRecord] = []
    recall_hits = 0
    n_gold = 0
    for i, (iid, gold_ids) in enumerate(query_meta):
        idxs = result.top_indices[i]
        scores = result.top_scores[i]
        retrieved = [
            {"skill_id": str(skill_ids_arr[idx]), "score": float(scores[j])}
            for j, idx in enumerate(idxs)
        ]
        records.append(RetrievalRecord(
            instance_id=iid, gold_skill_ids=gold_ids, retrieved=retrieved,
        ))
        gold = set(gold_ids)
        if gold:
            n_gold += 1
            if gold & {r["skill_id"] for r in retrieved}:
                recall_hits += 1

    out_path = args.output_root / "retrieval" / f"skillsight-{dataset}.json"
    RetrievalResults(
        retriever="skillsight",
        top_k=max_k,
        corpus_size=len(skill_ids),
        records=records,
        metrics={},
        dataset=dataset,
        extra={
            "embed_model": args.embed_model_path,
            "candidate_k": args.candidate_k,
            "max_tokens_per_doc": args.max_tokens_per_doc,
            "n_queries": len(records),
            "retrieval_wall_s": round(retrieval_wall_s, 4),
            "retrieval_latency_per_query_s": round(retrieval_latency_per_query, 6),
        },
    ).dump(out_path)

    hit_at_maxk = (recall_hits / n_gold) if n_gold else 0.0
    metrics = {f"hit@{max_k}(gold)": round(hit_at_maxk, 4), "n_gold": n_gold}
    print(f"[skillsight] {dataset}: wrote {out_path}  "
          f"hit@{max_k}(gold)={hit_at_maxk:.4f} over {n_gold} gold instances",
          flush=True)
    return out_path, metrics


# ---------------------------------------------------------------------------
# Stage 2/3: infer + evaluate for one (dataset, K)
# ---------------------------------------------------------------------------

def run_infer_and_eval(
    dataset: str,
    top_k: int,
    retrieval_path: Path,
    args: argparse.Namespace,
    env: Dict[str, str],
) -> Dict[str, Any]:
    instances_path = SR_AGENTS_ROOT / "data" / "bench" / "instances" / f"{dataset}.json"

    inference_dir = args.output_root / "inference"
    eval_dir = args.output_root / "eval"
    inference_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    safe_model = args.model.replace("/", "_").replace(":", "_")
    tag = f"skillsight_top{top_k}"
    infer_out = inference_dir / f"{dataset}-{safe_model}-{tag}.jsonl"
    eval_out = eval_dir / f"{dataset}-{safe_model}-{tag}.json"

    engine = "react" if dataset == "toolqa" else "direct"

    if args.overwrite or not infer_out.exists():
        workers = args.workers or WORKERS_BY_ENGINE.get(engine, 32)
        infer_cmd = [
            sys.executable, "-m", "sragents.cli.main", "infer",
            "--instances", str(instances_path),
            "--output", str(infer_out),
            "--model", args.model,
            "--api-base", args.api_base,
            "--provider", "topk",
            "--provider-arg", f"source={retrieval_path}",
            "--provider-arg", f"k={top_k}",
            "--engine", engine,
            "--label", tag,
            "--workers", str(workers),
        ]
        if args.overwrite:
            infer_cmd.append("--force")
        run_cmd(infer_cmd, cwd=SR_AGENTS_ROOT, env=env)
    else:
        print(f"[skip] inference exists: {infer_out}", flush=True)

    if args.overwrite or not eval_out.exists():
        eval_cmd = [
            sys.executable, "-m", "sragents.cli.main", "evaluate",
            "--input", str(infer_out),
            "--instances", str(instances_path),
            "--output", str(eval_out),
        ]
        if args.overwrite:
            eval_cmd.append("--force")
        run_cmd(eval_cmd, cwd=SR_AGENTS_ROOT, env=env)
    else:
        print(f"[skip] eval exists: {eval_out}", flush=True)

    result = load_json(eval_out)
    result["_dataset"] = dataset
    result["_top_k"] = top_k
    result["_engine"] = engine
    result["_instances_path"] = str(instances_path)
    result["_retrieval_path"] = str(retrieval_path)
    result["_inference_path"] = str(infer_out)
    result["_eval_path"] = str(eval_out)

    per_instance_stats, stat_averages = load_inference_stats(infer_out)

    retrieval_latency = load_retrieval_latency(retrieval_path)
    task_latencies: List[float] = []
    for meta in per_instance_stats.values():
        infer_lat = meta.get("latency_s")
        meta["retrieval_latency_s"] = round(retrieval_latency, 6)
        if isinstance(infer_lat, (int, float)):
            task_lat = float(infer_lat) + retrieval_latency
            meta["task_latency_s"] = round(task_lat, 4)
            task_latencies.append(task_lat)

    stat_averages["retrieval_latency_per_query_s"] = round(retrieval_latency, 6)
    if task_latencies:
        stat_averages["avg_task_latency_s"] = round(
            sum(task_latencies) / len(task_latencies), 4)
        stat_averages["total_task_latency_s"] = round(sum(task_latencies), 4)

    result["_instance_stats"] = per_instance_stats
    result["_stat_averages"] = stat_averages
    return result


# ---------------------------------------------------------------------------
# Output CSVs
# ---------------------------------------------------------------------------

def write_summary_csv(results: List[Dict[str, Any]], output_path: Path, model: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "top_k", "method", "model", "engine",
        "accuracy", "correct", "total",
        "avg_task_latency_s", "avg_infer_latency_s",
        "retrieval_latency_per_query_s",
        "total_task_latency_s", "total_infer_latency_s",
        "avg_input_tokens", "total_input_tokens", "n_samples",
        "retrieval_path", "inference_path", "eval_path",
    ]
    rows = []
    for obj in results:
        metrics = obj.get("metrics", {})
        averages = obj.get("_stat_averages", {})
        rows.append({
            "dataset": obj.get("_dataset", ""),
            "top_k": obj.get("_top_k", ""),
            "method": obj.get("method", f"skillsight_top{obj.get('_top_k', '')}"),
            "model": obj.get("model", model),
            "engine": obj.get("_engine", ""),
            "accuracy": metrics.get("accuracy", ""),
            "correct": metrics.get("correct", ""),
            "total": metrics.get("total", ""),
            "avg_task_latency_s": averages.get("avg_task_latency_s", ""),
            "avg_infer_latency_s": averages.get("avg_latency_s", ""),
            "retrieval_latency_per_query_s":
                averages.get("retrieval_latency_per_query_s", ""),
            "total_task_latency_s": averages.get("total_task_latency_s", ""),
            "total_infer_latency_s": averages.get("total_latency_s", ""),
            "avg_input_tokens": averages.get("avg_input_tokens", ""),
            "total_input_tokens": averages.get("total_input_tokens", ""),
            "n_samples": averages.get("n_latency_samples",
                                      averages.get("n_token_samples", "")),
            "retrieval_path": obj.get("_retrieval_path", ""),
            "inference_path": obj.get("_inference_path", ""),
            "eval_path": obj.get("_eval_path", ""),
        })
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] summary csv: {output_path}", flush=True)


def write_per_sample_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "top_k", "instance_id", "correct",
        "task_latency_s", "infer_latency_s", "retrieval_latency_s",
        "input_tokens", "output_tokens", "n_llm_calls",
        "extracted_answer", "ground_truth", "question", "engine", "eval_path",
    ]
    rows = []
    for obj in results:
        dataset = obj.get("_dataset", "")
        top_k = obj.get("_top_k", "")
        engine = obj.get("_engine", "")
        eval_path = obj.get("_eval_path", "")
        instance_stats = obj.get("_instance_stats", {})
        try:
            instances = json.loads(Path(obj["_instances_path"]).read_text())
            if isinstance(instances, dict):
                for key in ("instances", "data", "examples", "items"):
                    if isinstance(instances.get(key), list):
                        instances = instances[key]
                        break
            id2inst = {get_instance_id(i): i for i in instances}
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to load instances for {dataset}: {e}", flush=True)
            id2inst = {}

        for d in obj.get("details", []):
            iid = str(d.get("instance_id", d.get("id", "")))
            meta = instance_stats.get(iid, {})
            rows.append({
                "dataset": dataset,
                "top_k": top_k,
                "instance_id": iid,
                "correct": d.get("correct", ""),
                "task_latency_s": meta.get("task_latency_s", ""),
                "infer_latency_s": meta.get("latency_s", ""),
                "retrieval_latency_s": meta.get("retrieval_latency_s", ""),
                "input_tokens": meta.get("input_tokens", ""),
                "output_tokens": meta.get("output_tokens", ""),
                "n_llm_calls": meta.get("n_llm_calls", ""),
                "extracted_answer": cell(d.get("extracted_answer", "")),
                "ground_truth": cell(d.get("ground_truth", "")),
                "question": get_question(id2inst.get(iid, {})),
                "engine": engine,
                "eval_path": eval_path,
            })
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] per-sample csv: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end SkillSight-retrieval task accuracy on SRA-Bench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # LLM (task model, served by vLLM)
    p.add_argument("--model", default="Qwen3-4B-Instruct-2507",
                   help="Served LLM name. Must match vllm.sh --served-model-name.")
    p.add_argument("--api-base", default="http://127.0.0.1:8001/v1",
                   help="OpenAI-compatible endpoint (vLLM).")

    # Retrieval (SkillSight)
    p.add_argument("--embed-model-path", default=str(HERE / "models" / "Qwen3-Embedding-0.6B"),
                   help="Dense encoder used by SkillSight retrieval.")
    p.add_argument("--device", default="cuda:0",
                   help="Device for the SkillSight encoder.")
    p.add_argument("--candidate-k", type=int, default=300,
                   help="Candidate depth for SkillSight's dense+invariant rerank union.")
    p.add_argument("--max-tokens-per-doc", type=int, default=200,
                   help="Unique-token cap per doc for SkillSight's generic text model.")
    p.add_argument("--lexical-beta", type=float, default=LEXICAL_BETA,
                   help="Lexical discriminativeness-suppression strength beta "
                        "(0 recovers plain-IDF lexical evidence).")
    p.add_argument("--max-seq-length", type=int, default=32768)
    p.add_argument("--doc-batch", type=int, default=4)
    p.add_argument("--query-batch", type=int, default=64)
    p.add_argument("--score-batch", type=int, default=64)
    p.add_argument("--cache-dir", type=Path, default=HERE / "cache",
                   help="Embedding cache dir (shared with other retrieval evals).")

    # Experiment
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--top-ks", nargs="+", type=int, default=DEFAULT_TOP_KS,
                   help="Top-K skill counts to inject and evaluate.")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel inference workers. Defaults: direct=32, react=8.")
    p.add_argument("--output-root", type=Path,
                   default=HERE / "results" / "skillsight_e2e")

    # Control
    p.add_argument("--overwrite", action="store_true",
                   help="Redo inference/eval even if outputs exist.")
    p.add_argument("--reuse-retrieval", action="store_true",
                   help="Reuse an existing retrieval JSON instead of re-retrieving.")
    p.add_argument("--retrieval-only", action="store_true",
                   help="Only generate retrieval JSON, then exit without starting or calling vLLM. "
                        "Use this first on a single-GPU host.")
    p.add_argument("--no-start-vllm", action="store_true",
                   help="Do not launch vllm.sh; assume the server is already up.")
    p.add_argument("--skip-server-check", action="store_true",
                   help="Skip waiting for the vLLM /models endpoint.")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    top_ks = sorted(set(int(k) for k in args.top_ks))
    max_k = max(top_ks)

    # ---- Stage 0: vLLM ----
    if not args.retrieval_only:
        if not args.no_start_vllm:
            maybe_launch_vllm(HERE, args.api_base)
        if not args.skip_server_check:
            wait_for_openai_server(args.api_base)

    # ---- Stage 1: SkillSight retrieval (once per dataset, at K=max) ----
    retrieval_paths: Dict[str, Path] = {}
    encoder_holder: List[Any] = []
    for dataset in args.datasets:
        rjson = args.output_root / "retrieval" / f"skillsight-{dataset}.json"
        if args.reuse_retrieval and rjson.exists():
            print(f"[skip] reuse retrieval: {rjson}", flush=True)
            retrieval_paths[dataset] = rjson
            continue
        print("\n" + "=" * 80)
        print(f"[retrieve] {dataset}")
        print("=" * 80)
        rpath, _ = skillsight_retrieve_dataset(dataset, max_k, args, encoder_holder)
        retrieval_paths[dataset] = rpath

    # Free the encoder / GPU before the LLM-bound inference stage.
    encoder_holder.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    if args.retrieval_only:
        print("[done] retrieval-only stage complete.", flush=True)
        for dataset, path in retrieval_paths.items():
            print(f"[done] {dataset}: {path}", flush=True)
        return

    subprocess_env = build_subprocess_env()

    # ---- Stage 2/3: infer + evaluate per (dataset, K) ----
    results: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        for top_k in top_ks:
            print("\n" + "=" * 80)
            print(f"[eval] {dataset}  top_k={top_k}")
            print("=" * 80)
            result = run_infer_and_eval(
                dataset=dataset,
                top_k=top_k,
                retrieval_path=retrieval_paths[dataset],
                args=args,
                env=subprocess_env,
            )
            acc = result.get("metrics", {}).get("accuracy", "?")
            averages = result.get("_stat_averages", {})
            avg_task = averages.get("avg_task_latency_s", "?")
            avg_infer = averages.get("avg_latency_s", "?")
            retr_lat = averages.get("retrieval_latency_per_query_s", "?")
            avg_tok = averages.get("avg_input_tokens", "?")
            n = averages.get("n_latency_samples", averages.get("n_token_samples", "?"))
            print(f"[eval] {dataset} top{top_k}: accuracy={acc}  "
                  f"avg_task_latency={avg_task}s "
                  f"(retrieval={retr_lat}s + infer={avg_infer}s)  "
                  f"avg_input_tokens={avg_tok}  (n={n})",
                  flush=True)
            results.append(result)

    write_summary_csv(results, args.output_root / "summary_skillsight_e2e.csv", args.model)
    write_per_sample_csv(results, args.output_root / "per_sample_skillsight_e2e.csv")

    print("\n[done] SkillSight end-to-end evaluation finished.", flush=True)
    print(f"[done] output root: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
