"""SkillSight retriever implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm

from .constants import (
    ALL_VARIANTS,
    LEXICAL_BETA,
    VARIANT_DENSE,
    VARIANT_DENSE_LEC,
    VARIANT_LEC,
    VARIANT_SBC,
    VARIANT_SKILLSIGHT,
    VARIANT_SKILLSIGHT_EQ15,
)
from .fusion import (
    lexical_evidence_scores,
    nongeneric_query_tokens,
    skillsight_fusion,
)
from .text_model import TextStats, build_query_tokens, load_or_build_text_model, project_out_background
from .utils import l2_normalize, topk_indices_batch, topk_values_batch, zscore_1d


@dataclass
class RetrievalResult:
    variant: str
    top_indices: np.ndarray
    top_scores: np.ndarray
    # Online retrieval wall time (seconds): query-side transform + scoring + top-k.
    search_seconds: float = 0.0


class SkillSightRetriever:
    """SkillSight retrieval.

    Estimates a corpus-level background subspace B, removes it via the projection
    (I - B B^T) (Semantic Background Calibration), and reranks candidates using the
    background-calibrated dense score s_perp fused with the calibrated lexical
    evidence score s_lex (Lexical Evidence Calibration).
    """

    def __init__(
        self,
        doc_texts: Sequence[str],
        doc_embeddings: np.ndarray,
        skill_ids: Sequence[str],
        candidate_k: int = 300,
        score_batch: int = 64,
        cache_dir: Optional[Path] = None,
        cache_tag: str = "all",
        prebuilt_text_model: Optional[Tuple[TextStats, np.ndarray]] = None,
        lexical_beta: float = LEXICAL_BETA,
    ):
        self.doc_texts = list(doc_texts)
        self.docs = l2_normalize(doc_embeddings)
        self.skill_ids = list(skill_ids)
        self.candidate_k = candidate_k
        self.score_batch = score_batch
        self.cache_dir = cache_dir
        self.cache_tag = cache_tag
        self.lexical_beta = float(lexical_beta)

        if prebuilt_text_model is not None:
            self.stats, self.background_basis = prebuilt_text_model
            self.background_basis = self.background_basis.astype(np.float32)
            print(f"[skillsight] reuse token statistics and background subspace: {self.cache_tag}", flush=True)
        else:
            self.stats, self.background_basis = load_or_build_text_model(
                doc_texts=self.doc_texts,
                doc_embeddings=self.docs,
                cache_dir=self.cache_dir,
                cache_tag=self.cache_tag,
            )

        # Background-calibrated document representations d_perp = (I - B B^T) d.
        self.docs_perp = project_out_background(self.docs, self.background_basis)

    def score(
        self,
        query_texts: Sequence[str],
        query_embeddings: np.ndarray,
        final_k: int,
        variants: Optional[Sequence[str]] = None,
    ) -> Dict[str, RetrievalResult]:
        selected_variants = list(dict.fromkeys(variants or [VARIANT_SKILLSIGHT]))
        unknown = set(selected_variants) - set(ALL_VARIANTS)
        if unknown:
            raise ValueError(f"Unknown retrieval variants: {sorted(unknown)}")

        # ---- online query-side transforms (timed) ----
        t0 = time.perf_counter()
        q = l2_normalize(query_embeddings)
        shared_prep_s = time.perf_counter() - t0

        lexical_variants = {
            VARIANT_LEC,
            VARIANT_DENSE_LEC,
            VARIANT_SKILLSIGHT,
            VARIANT_SKILLSIGHT_EQ15,
        }
        needs_dense = any(v != VARIANT_SBC for v in selected_variants)
        needs_perp = any(
            v in {VARIANT_SBC, VARIANT_SKILLSIGHT, VARIANT_SKILLSIGHT_EQ15}
            for v in selected_variants
        )
        needs_tokens = any(v in lexical_variants for v in selected_variants)

        proj_s = 0.0
        q_perp = None
        if needs_perp:
            t0 = time.perf_counter()
            q_perp = project_out_background(q, self.background_basis)
            proj_s = time.perf_counter() - t0

        tok_s = 0.0
        query_tokens = []
        if needs_tokens:
            t0 = time.perf_counter()
            query_tokens = build_query_tokens(query_texts)
            tok_s = time.perf_counter() - t0

        cand_k = min(max(final_k, self.candidate_k), self.docs.shape[0])

        idx_blocks: Dict[str, list[np.ndarray]] = {v: [] for v in selected_variants}
        val_blocks: Dict[str, list[np.ndarray]] = {v: [] for v in selected_variants}
        rank_seconds: Dict[str, float] = {v: 0.0 for v in selected_variants}
        raw_product_s = 0.0
        inv_product_s = 0.0

        for start in tqdm(range(0, q.shape[0], self.score_batch), desc="  score", unit="batch"):
            end = min(start + self.score_batch, q.shape[0])

            s_dense = None
            if needs_dense:
                t0 = time.perf_counter()
                # s(q, d) = q^T d : raw dense score (Eq. 1).
                s_dense = q[start:end] @ self.docs.T
                raw_product_s += time.perf_counter() - t0

            s_perp = None
            if needs_perp:
                t0 = time.perf_counter()
                # s_perp(q, d) = q_perp^T d_perp : calibrated dense score (Eq. 9).
                s_perp = q_perp[start:end] @ self.docs_perp.T
                inv_product_s += time.perf_counter() - t0

            for variant in selected_variants:
                t0 = time.perf_counter()
                if variant == VARIANT_DENSE:
                    var_idx = topk_indices_batch(s_dense, final_k)
                    var_vals = topk_values_batch(s_dense, var_idx)
                elif variant == VARIANT_SBC:
                    var_idx = topk_indices_batch(s_perp, final_k)
                    var_vals = topk_values_batch(s_perp, var_idx)
                else:
                    var_idx, var_vals = self._score_lexical_variant(
                        start=start,
                        end=end,
                        variant=variant,
                        s_dense=s_dense,
                        s_perp=s_perp,
                        query_tokens=query_tokens,
                        final_k=final_k,
                        cand_k=cand_k,
                    )
                rank_seconds[variant] += time.perf_counter() - t0
                idx_blocks[variant].append(var_idx)
                val_blocks[variant].append(var_vals)

        results: Dict[str, RetrievalResult] = {}
        for variant in selected_variants:
            variant_seconds = shared_prep_s + rank_seconds[variant]
            if variant != VARIANT_SBC:
                variant_seconds += raw_product_s
            if variant in {VARIANT_SBC, VARIANT_SKILLSIGHT, VARIANT_SKILLSIGHT_EQ15}:
                variant_seconds += proj_s + inv_product_s
            if variant in lexical_variants:
                variant_seconds += tok_s
            results[variant] = RetrievalResult(
                variant=variant,
                top_indices=np.vstack(idx_blocks[variant]),
                top_scores=np.vstack(val_blocks[variant]),
                search_seconds=float(variant_seconds),
            )
        return results

    def _score_lexical_variant(
        self,
        start: int,
        end: int,
        variant: str,
        s_dense: np.ndarray,
        s_perp: np.ndarray,
        query_tokens: Sequence[Set[str]],
        final_k: int,
        cand_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Score one lexical, hybrid, or full SkillSight variant."""
        dense_idx = topk_indices_batch(s_dense, cand_k)
        uses_union = variant in {VARIANT_SKILLSIGHT, VARIANT_SKILLSIGHT_EQ15}
        perp_idx = topk_indices_batch(s_perp, cand_k) if uses_union else None

        n_rows = end - start
        out_idx = np.zeros((n_rows, final_k), dtype=np.int64)
        out_vals = np.zeros((n_rows, final_k), dtype=np.float32)

        for row in range(n_rows):
            if uses_union:
                cand = np.unique(np.concatenate([dense_idx[row], perp_idx[row]])).astype(np.int64)
            else:
                cand = np.unique(dense_idx[row]).astype(np.int64)
            cand_s_dense = s_dense[row, cand].astype(np.float32)
            cand_s_perp = (
                s_perp[row, cand].astype(np.float32)
                if variant in {VARIANT_SKILLSIGHT, VARIANT_SKILLSIGHT_EQ15}
                else None
            )

            qtok = query_tokens[start + row]
            q_nongeneric = nongeneric_query_tokens(qtok, self.stats)
            cand_s_lex = (
                lexical_evidence_scores(q_nongeneric, cand, self.stats, beta=self.lexical_beta)
                if q_nongeneric else np.zeros_like(cand_s_dense)
            )
            if variant == VARIANT_LEC:
                # Lexical-only diagnostic: lexical score is primary and dense
                # score only breaks exact ties.
                final_scores = (
                    zscore_1d(cand_s_lex) + np.float32(1e-6) * zscore_1d(cand_s_dense)
                ).astype(np.float32)
            elif variant == VARIANT_DENSE_LEC:
                final_scores = skillsight_fusion(cand_s_dense, cand_s_lex, lam=None)
            elif variant == VARIANT_SKILLSIGHT:
                final_scores = skillsight_fusion(cand_s_perp, cand_s_lex, lam=None)
            elif variant == VARIANT_SKILLSIGHT_EQ15:
                final_scores = skillsight_fusion(cand_s_perp, cand_s_lex, lam=1.0)
            else:
                raise ValueError(f"Unsupported lexical variant: {variant}")

            order = np.argsort(final_scores)[::-1][:final_k]
            out_idx[row, :len(order)] = cand[order]
            out_vals[row, :len(order)] = final_scores[order].astype(np.float32)

        return out_idx, out_vals
