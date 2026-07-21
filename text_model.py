"""SkillSight text model: generic-token discovery and latent nuisance subspace."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm

from .constants import MAX_TOKENS_PER_DOC, MIN_GENERIC_DF, STOPWORDS, TEXT_MODEL_VERSION
from .utils import TOKEN_RE, hash_strings, l2_normalize, otsu_threshold


def tokenize(text: str, min_len: int = 2) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    out = []
    for tok in toks:
        if len(tok) < min_len:
            continue
        if tok in STOPWORDS:
            continue
        if len(tok) > 40:
            continue
        out.append(tok)
    return out


@dataclass
class TextStats:
    doc_tokens: List[Set[str]]
    df: Dict[str, int]
    idf: Dict[str, float]
    generic_tokens: Set[str]                 # T_g: low-IDF generic token set
    token_to_generic_index: Dict[str, int]
    generic_token_matrix: np.ndarray         # rows are token-induced means g_t
    corpus_mean: np.ndarray                  # corpus mean embedding d_bar
    idf_threshold: float
    background_rank: int = 0                  # r: rank of background subspace B
    background_explained_variance: float = 0.0
    background_rank_rule: str = "entropy-effective-rank"


def build_text_stats_and_prototypes(
    texts: Sequence[str],
    doc_embeddings: np.ndarray,
    max_tokens_per_doc: int = MAX_TOKENS_PER_DOC,
) -> TextStats:
    """Build IDF statistics and low-IDF token-induced mean embeddings g_t."""
    print("[skillsight] building token statistics and generic-token means ...", flush=True)
    doc_tokens: List[Set[str]] = []
    df_counter: Counter = Counter()

    for text in tqdm(texts, desc="  tokenize-docs", unit="doc"):
        toks = tokenize(text)
        seen: Set[str] = set()
        unique_toks: List[str] = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                unique_toks.append(t)
                if max_tokens_per_doc > 0 and len(unique_toks) >= max_tokens_per_doc:
                    break
        st = set(unique_toks)
        doc_tokens.append(st)
        df_counter.update(st)

    n = len(texts)
    idf = {token: float(math.log((n + 1.0) / (df + 1.0)) + 1.0) for token, df in df_counter.items()}

    eligible = [
        (token, df)
        for token, df in df_counter.items()
        if df >= MIN_GENERIC_DF and token not in STOPWORDS
    ]
    if eligible:
        idf_vals = np.asarray([idf[token] for token, _ in eligible], dtype=np.float64)
        dfs = np.asarray([df for _, df in eligible], dtype=np.int64)
        idf_values = np.repeat(idf_vals, dfs)
        idf_threshold = otsu_threshold(idf_values)
    else:
        idf_threshold = float("inf")

    print(f"[skillsight] Otsu IDF threshold: {idf_threshold:.4f}", flush=True)
    generic_tokens = {token for token, _ in eligible if idf[token] <= idf_threshold}
    print(f"[skillsight] generic token count: {len(generic_tokens)}", flush=True)

    dim = doc_embeddings.shape[1]
    # g_t accumulator: mean document embedding over docs containing generic token t (Eq. 3).
    gt_sum: Dict[str, np.ndarray] = {token: np.zeros(dim, dtype=np.float32) for token in generic_tokens}
    gt_cnt: Dict[str, int] = {token: 0 for token in generic_tokens}

    for toks, emb in tqdm(
        zip(doc_tokens, doc_embeddings),
        total=len(doc_tokens),
        desc="  generic-token-means",
        unit="doc",
    ):
        for token in toks & generic_tokens:
            gt_sum[token] += emb
            gt_cnt[token] += 1

    gt_tokens: List[str] = []
    gt_vecs: List[np.ndarray] = []
    for token in sorted(generic_tokens):
        cnt = gt_cnt.get(token, 0)
        if cnt > 0:
            gt_tokens.append(token)
            gt_vecs.append(gt_sum[token] / float(cnt))

    if gt_vecs:
        generic_token_matrix = l2_normalize(np.vstack(gt_vecs).astype(np.float32))
    else:
        generic_token_matrix = np.zeros((0, dim), dtype=np.float32)

    return TextStats(
        doc_tokens=doc_tokens,
        df=dict(df_counter),
        idf=idf,
        generic_tokens=set(gt_tokens),
        token_to_generic_index={token: idx for idx, token in enumerate(gt_tokens)},
        generic_token_matrix=generic_token_matrix,
        corpus_mean=l2_normalize(doc_embeddings.mean(axis=0).astype(np.float32)),
        idf_threshold=float(idf_threshold),
    )


def build_background_subspace(stats: TextStats, n_docs: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate the corpus-level background subspace basis B via SVD (Eq. 4)."""
    dim = int(stats.corpus_mean.shape[0])
    if stats.generic_token_matrix.size == 0:
        print("[skillsight] no generic-token means; background basis is empty", flush=True)
        return np.zeros((dim, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.0

    # G stacks the token-induced means g_t together with the corpus mean d_bar as a
    # background anchor; do not center G, because the mean direction itself is a
    # background component that should be removable.
    G = np.vstack([stats.generic_token_matrix, stats.corpus_mean[None, :]]).astype(np.float32)
    G = l2_normalize(G)

    try:
        _, singular_values, vt = np.linalg.svd(G, full_matrices=False)
    except np.linalg.LinAlgError:
        q_basis, _ = np.linalg.qr(G.T)
        rank = min(q_basis.shape[1], max(1, int(math.ceil(math.sqrt(max(1, n_docs))))))
        background_basis = q_basis[:, :rank].astype(np.float32)
        return background_basis, np.zeros((rank,), dtype=np.float32), 0.0

    power = np.asarray(singular_values ** 2, dtype=np.float64)
    total_power = float(power.sum())
    if total_power <= 1e-12:
        return np.zeros((dim, 0), dtype=np.float32), singular_values.astype(np.float32), 0.0

    probs = power / total_power
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    rank = int(math.ceil(math.exp(entropy)))
    rank = max(1, min(rank, vt.shape[0], dim))

    # B = [u_1, ..., u_r]: leading r right-singular directions of G (Eq. 4).
    background_basis = vt[:rank].T.astype(np.float32)
    background_basis, _ = np.linalg.qr(background_basis)
    background_basis = background_basis.astype(np.float32)
    explained = float(power[:rank].sum() / total_power)

    stats.background_rank = int(rank)
    stats.background_explained_variance = explained
    stats.background_rank_rule = "entropy-effective-rank"
    print(
        f"[skillsight] background subspace rank={rank} explained={explained:.4f} "
        f"rule={stats.background_rank_rule}",
        flush=True,
    )
    return background_basis, singular_values.astype(np.float32), explained


def project_out_background(x: np.ndarray, background_basis: np.ndarray) -> np.ndarray:
    """Project rows of x onto the orthogonal complement of the background subspace.

    Applies (I - P_B) x with P_B = B B^T (Eq. 8), yielding the residual
    representation used by the background-calibrated dense score s_perp (Eq. 9).
    """
    x = np.asarray(x, dtype=np.float32)
    if background_basis.size == 0 or background_basis.shape[1] == 0:
        return x.astype(np.float32)
    return (x - (x @ background_basis) @ background_basis.T).astype(np.float32)


def build_query_tokens(query_texts: Sequence[str]) -> List[Set[str]]:
    return [set(tokenize(text)) for text in query_texts]


def text_model_cache_paths(cache_dir: Path, cache_tag: str) -> Tuple[Path, Path]:
    base = f"clear_textmodel_{cache_tag}"
    return cache_dir / f"{base}.npz", cache_dir / f"{base}.meta.json"


def load_or_build_text_model(
    doc_texts: Sequence[str],
    doc_embeddings: np.ndarray,
    cache_dir: Optional[Path],
    cache_tag: str,
    max_tokens_per_doc: int = MAX_TOKENS_PER_DOC,
) -> Tuple[TextStats, np.ndarray]:
    """Load or build token statistics and the background subspace basis B."""
    if cache_dir is not None:
        npz_path, meta_path = text_model_cache_paths(cache_dir, cache_tag)
        if npz_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if (
                    meta.get("text_sha1") == hash_strings(doc_texts)
                    and int(meta.get("embedding_dim")) == int(doc_embeddings.shape[1])
                    and meta.get("text_model_version") == TEXT_MODEL_VERSION
                    and int(meta.get("max_tokens_per_doc", MAX_TOKENS_PER_DOC)) == int(max_tokens_per_doc)
                ):
                    arr = np.load(npz_path, allow_pickle=True)
                    # On-disk npz keys are kept for cache/backward compatibility;
                    # they map onto the paper-aligned TextStats field names.
                    stats = TextStats(
                        doc_tokens=[set(x) for x in arr["doc_tokens"].tolist()],
                        df={str(k): int(v) for k, v in arr["df"].tolist()},
                        idf={str(k): float(v) for k, v in arr["idf"].tolist()},
                        generic_tokens=set(arr["generic_tokens"].tolist()),
                        token_to_generic_index={str(k): int(v) for k, v in arr["token_to_idx"].tolist()},
                        generic_token_matrix=arr["prototype_matrix"].astype(np.float32),
                        corpus_mean=arr["generic_mean"].astype(np.float32),
                        idf_threshold=float(arr["idf_threshold"]),
                        background_rank=int(arr["generic_rank"]),
                        background_explained_variance=float(arr["generic_explained_variance"]),
                        background_rank_rule=str(arr["generic_rank_rule"]),
                    )
                    background_basis = arr["generic_basis"].astype(np.float32)
                    print(f"[cache] load CLEAR text model: {npz_path}", flush=True)
                    return stats, background_basis
            except Exception as exc:
                print(f"[cache] failed to load CLEAR text model ({exc}); rebuild", flush=True)

    stats = build_text_stats_and_prototypes(doc_texts, doc_embeddings, max_tokens_per_doc=max_tokens_per_doc)
    background_basis, singular_values, explained = build_background_subspace(stats, n_docs=len(doc_texts))
    stats.background_rank = int(background_basis.shape[1])
    stats.background_explained_variance = float(explained)

    if cache_dir is not None:
        npz_path, meta_path = text_model_cache_paths(cache_dir, cache_tag)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            generic_tokens=np.asarray(sorted(stats.generic_tokens), dtype=object),
            token_to_idx=np.asarray(list(stats.token_to_generic_index.items()), dtype=object),
            doc_tokens=np.asarray([list(s) for s in stats.doc_tokens], dtype=object),
            df=np.asarray(list(stats.df.items()), dtype=object),
            idf=np.asarray(list(stats.idf.items()), dtype=object),
            prototype_matrix=stats.generic_token_matrix,
            generic_mean=stats.corpus_mean,
            idf_threshold=np.asarray(stats.idf_threshold, dtype=np.float64),
            generic_basis=background_basis,
            singular_values=singular_values,
            generic_rank=np.asarray(stats.background_rank, dtype=np.int64),
            generic_explained_variance=np.asarray(stats.background_explained_variance, dtype=np.float64),
            generic_rank_rule=np.asarray(stats.background_rank_rule, dtype=object),
        )
        meta = {
            "text_sha1": hash_strings(doc_texts),
            "embedding_dim": int(doc_embeddings.shape[1]),
            "text_model_version": TEXT_MODEL_VERSION,
            "method": "CLEAR: Calibrated Latent Evidence-Aware Retrieval",
            "idf_threshold": float(stats.idf_threshold),
            "generic_rank": int(stats.background_rank),
            "generic_explained_variance": float(stats.background_explained_variance),
            "generic_rank_rule": stats.background_rank_rule,
            "min_generic_df": int(MIN_GENERIC_DF),
            "max_tokens_per_doc": int(max_tokens_per_doc),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"[cache] save CLEAR text model: {npz_path}", flush=True)

    return stats, background_basis


def shared_text_model_cache_tag(dataset: str, skill_ids: Sequence[str]) -> str:
    from .utils import stable_hash

    return f"all_{stable_hash(skill_ids)}"


def shared_text_model_cache_key(doc_texts: Sequence[str], doc_embeddings: np.ndarray) -> tuple[object, ...]:
    return (hash_strings(doc_texts), int(doc_embeddings.shape[1]))
