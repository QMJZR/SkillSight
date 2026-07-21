"""Small numerical and hashing utilities for SkillSight."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]{1,}|\d+(?:\.\d+)?")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        denom = max(float(np.linalg.norm(x)), eps)
        return (x / denom).astype(np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return (x / denom).astype(np.float32)


def zscore_rows(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return ((x - mu) / np.maximum(sd, eps)).astype(np.float32)


def zscore_1d(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    sd = float(x.std())
    if sd <= eps:
        return np.zeros_like(x)
    return ((x - float(x.mean())) / (sd + eps)).astype(np.float32)


def topk_indices_batch(scores: np.ndarray, k: int) -> np.ndarray:
    n = scores.shape[1]
    if k >= n:
        return np.argsort(scores, axis=1)[:, ::-1]
    part = np.argpartition(scores, -k, axis=1)[:, -k:]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(part_scores, axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)


def topk_values_batch(scores: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.take_along_axis(scores, idx, axis=1)


def stable_hash(items: Sequence[str]) -> str:
    h = hashlib.sha1()
    for item in items:
        h.update(str(item).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def hash_strings(strings: Sequence[str]) -> str:
    h = hashlib.sha1()
    for s in strings:
        h.update(str(s).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def model_tag(model_path: str) -> str:
    name = Path(model_path).name or "model"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def json_float(x: float) -> float:
    return float(round(float(x), 10))


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Parameter-free Otsu threshold for a one-dimensional distribution."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return vmin

    hist, edges = np.histogram(v, bins=n_bins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float(np.median(v))

    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    cumsum_val = np.cumsum(hist * centers)
    total_val = cumsum_val[-1]
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return float(np.median(v))

    mean_bg = np.where(weight_bg > 0, cumsum_val / np.maximum(weight_bg, 1e-12), 0.0)
    mean_fg = np.where(
        weight_fg > 0, (total_val - cumsum_val) / np.maximum(weight_fg, 1e-12), 0.0
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between = np.where(valid, between, -np.inf)
    best = float(np.max(between))
    if not np.isfinite(best) or best <= 0:
        return float(np.median(v))

    plateau = np.isfinite(between) & (between >= best - 1e-12)
    return float(np.mean(centers[plateau]))
