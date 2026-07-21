"""Embedding and text-model cache helpers for SkillSight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .encoder import DenseEncoder
from .utils import hash_strings, l2_normalize, model_tag, stable_hash


def load_cached_embeddings_if_valid(
    cache_path: Path,
    texts: Sequence[str],
    prompt_name: Optional[str],
    label: str,
    expected_dim: Optional[int] = None,
    max_seq_length: Optional[int] = None,
    model_path: Optional[str] = None,
    strict_text_hash: bool = True,
) -> Optional[np.ndarray]:
    if not cache_path.exists():
        return None

    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            print(f"[cache] cannot read metadata for {label}: {meta_path} ({exc}); skip", flush=True)
            return None

    try:
        arr = np.load(cache_path).astype(np.float32)
    except Exception as exc:
        print(f"[cache] cannot load {label}: {cache_path} ({exc}); skip", flush=True)
        return None

    if arr.ndim != 2 or arr.shape[0] != len(texts):
        print(f"[cache] shape mismatch for {label}: got {arr.shape}, expected n={len(texts)}; skip", flush=True)
        return None
    if expected_dim is not None and arr.shape[1] != expected_dim:
        print(f"[cache] dim mismatch for {label}: got {arr.shape[1]}, expected {expected_dim}; skip", flush=True)
        return None

    if meta:
        n_meta = meta.get("n_texts", meta.get("n"))
        if n_meta is not None and int(n_meta) != len(texts):
            return None
        if meta.get("prompt_name") != prompt_name:
            return None
        if strict_text_hash:
            expected_full_hash = hash_strings(texts)
            expected_short_hash = stable_hash(texts)
            meta_hash = meta.get("text_sha1", meta.get("hash"))
            if meta_hash is not None and meta_hash not in {expected_full_hash, expected_short_hash}:
                return None
        dim_meta = meta.get("embedding_dim", meta.get("dim"))
        if dim_meta is not None and int(dim_meta) != arr.shape[1]:
            return None
        if max_seq_length is not None and meta.get("max_seq_length") is not None:
            if int(meta.get("max_seq_length")) != int(max_seq_length):
                return None
        if model_path is not None and meta.get("model_path") is not None:
            if str(meta.get("model_path")) != str(model_path):
                return None
    elif strict_text_hash:
        return None

    print(f"[cache] load {label}: {cache_path} {arr.shape}", flush=True)
    return arr.astype(np.float32)


def candidate_embedding_cache_paths(
    cache_dir: Path,
    kind: str,
    dataset: Optional[str],
    model_path: str,
    max_seq_length: int,
    ids_or_texts: Sequence[str],
) -> list[Path]:
    tag = model_tag(model_path)
    paths: list[Path] = []
    seen = set()

    if kind == "doc_all":
        ds = dataset or "all"
        names = [
            f"doc_full_{stable_hash(ids_or_texts)}.npy",
            f"doc_all_{tag}_max{max_seq_length}.npy",
            f"{ds}_{tag}_{max_seq_length}.npy",
        ]
    elif kind == "query":
        assert dataset is not None
        names = [
            f"query_{dataset}_{stable_hash(ids_or_texts)}.npy",
            f"query_{dataset}_{tag}_max{max_seq_length}.npy",
            f"{dataset}_query_{tag}_{max_seq_length}.npy",
        ]
    else:
        raise ValueError(f"Unknown cache kind: {kind}")

    for name in names:
        path = cache_dir / name
        if str(path) not in seen:
            paths.append(path)
            seen.add(str(path))
    return paths


def load_or_encode_embeddings(
    encoder: Optional[DenseEncoder],
    texts: Sequence[str],
    cache_paths: Sequence[Path],
    batch_size: int,
    prompt_name: Optional[str],
    label: str,
    model_path: str,
    device: str,
    max_seq_length: int,
    allow_encode: bool,
) -> Tuple[np.ndarray, Optional[DenseEncoder]]:
    for path in cache_paths:
        arr = load_cached_embeddings_if_valid(
            path,
            texts=texts,
            prompt_name=prompt_name,
            label=label,
            max_seq_length=max_seq_length,
            model_path=model_path,
            strict_text_hash=True,
        )
        if arr is not None:
            return l2_normalize(arr), encoder

    if not allow_encode:
        checked = "\n  ".join(str(p) for p in cache_paths)
        raise FileNotFoundError(f"No valid cache for {label}; --cache-only set. Checked:\n  {checked}")

    if encoder is None:
        print("[encoder] cache miss; loading encoder ...", flush=True)
        encoder = DenseEncoder(model_path, device=device, max_seq_length=max_seq_length)

    save_path = cache_paths[-1]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[encode] {label}: n={len(texts)} -> {save_path}", flush=True)

    arr = encoder.encode(texts, batch_size=batch_size, show_progress=True, prompt_name=prompt_name)
    arr = l2_normalize(arr)
    np.save(save_path, arr)
    meta = {
        "n_texts": len(texts),
        "text_sha1": hash_strings(texts),
        "prompt_name": prompt_name,
        "embedding_dim": int(arr.shape[1]),
        "max_seq_length": int(max_seq_length),
        "model_path": model_path,
        "cache_written_by": "SkillSight",
    }
    save_path.with_suffix(save_path.suffix + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    return arr, encoder
