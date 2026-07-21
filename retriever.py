"""SkillSight retriever implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm

from .constants import LEXICAL_BETA, VARIANT_SKILLSIGHT
from .fusion import (
    lexical_evidence_scores,
    nongeneric_query_tokens,
    skillsight_fusion,
)
from .text_model import TextStats, build_query_tokens, load_or_build_text_model, project_out_background
from .utils import l2_normalize, topk_indices_batch, topk_values_batch


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
    ) -> Dict[str, RetrievalResult]:
        # ---- online query-side transforms (timed) ----
        t0 = time.perf_counter()
        q = l2_normalize(query_embeddings)
        shared_prep_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_perp = project_out_background(q, self.background_basis)
        proj_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        query_tokens = build_query_tokens(query_texts)
        tok_s = time.perf_counter() - t0
        prep_s = shared_prep_s + proj_s + tok_s

        cand_k = min(max(final_k, self.candidate_k), self.docs.shape[0])

        idx_blocks: list[np.ndarray] = []
        val_blocks: list[np.ndarray] = []
        raw_product_s = 0.0
        inv_product_s = 0.0
        skillsight_score_s = 0.0

        for start in tqdm(range(0, q.shape[0], self.score_batch), desc="  score", unit="batch"):
            end = min(start + self.score_batch, q.shape[0])

            t0 = time.perf_counter()
            # s(q, d) = q^T d : raw dense score (Eq. 1).
            s_dense = q[start:end] @ self.docs.T
            raw_product_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            # s_perp(q, d) = q_perp^T d_perp : background-calibrated dense score (Eq. 9).
            s_perp = q_perp[start:end] @ self.docs_perp.T
            inv_product_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            skillsight_idx, skillsight_vals = self._score_skillsight(
                start=start,
                end=end,
                s_dense=s_dense,
                s_perp=s_perp,
                query_tokens=query_tokens,
                final_k=final_k,
                cand_k=cand_k,
            )
            skillsight_score_s += time.perf_counter() - t0

            idx_blocks.append(skillsight_idx)
            val_blocks.append(skillsight_vals)

        search_seconds = prep_s + raw_product_s + inv_product_s + skillsight_score_s

        return {
            VARIANT_SKILLSIGHT: RetrievalResult(
                variant=VARIANT_SKILLSIGHT,
                top_indices=np.vstack(idx_blocks),
                top_scores=np.vstack(val_blocks),
                search_seconds=float(search_seconds),
            )
        }

    def _score_skillsight(
        self,
        start: int,
        end: int,
        s_dense: np.ndarray,
        s_perp: np.ndarray,
        query_tokens: Sequence[Set[str]],
        final_k: int,
        cand_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rerank the dense + calibrated-dense candidate union with SkillSight fusion."""
        dense_idx = topk_indices_batch(s_dense, cand_k)
        perp_idx = topk_indices_batch(s_perp, cand_k)

        n_rows = end - start
        out_idx = np.zeros((n_rows, final_k), dtype=np.int64)
        out_vals = np.zeros((n_rows, final_k), dtype=np.float32)

        for row in range(n_rows):
            cand = np.unique(np.concatenate([dense_idx[row], perp_idx[row]])).astype(np.int64)
            cand_s_perp = s_perp[row, cand].astype(np.float32)

            qtok = query_tokens[start + row]
            q_nongeneric = nongeneric_query_tokens(qtok, self.stats)
            cand_s_lex = (
                lexical_evidence_scores(q_nongeneric, cand, self.stats, beta=self.lexical_beta)
                if q_nongeneric else np.zeros_like(cand_s_perp)
            )
            final_scores = skillsight_fusion(cand_s_perp, cand_s_lex, lam=None)

            order = np.argsort(final_scores)[::-1][:final_k]
            out_idx[row, :len(order)] = cand[order]
            out_vals[row, :len(order)] = final_scores[order].astype(np.float32)

        return out_idx, out_vals
