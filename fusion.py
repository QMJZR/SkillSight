"""Lexical evidence and local dense-lexical calibration for SkillSight."""

from __future__ import annotations

from typing import List, Set

import numpy as np

from .constants import LEXICAL_BETA, STOPWORDS
from .text_model import TextStats
from .utils import zscore_1d


def nongeneric_query_tokens(q_tokens: Set[str], stats: TextStats) -> List[str]:
    """Non-generic in-corpus query tokens T_q^sp = T_q \\ T_g (Eq. 10).

    Query tokens present in the corpus, ordered by decreasing IDF.
    """
    out = [t for t in q_tokens if t not in STOPWORDS and stats.df.get(t, 0) > 0]
    out.sort(key=lambda t: stats.idf.get(t, 0.0), reverse=True)
    return out


def calibrated_token_weight(token: str, stats: TextStats, beta: float = LEXICAL_BETA) -> float:
    """Lexical evidence calibrated token weight w_beta(t) (Eq. 12).

        w_beta(t) = IDF(t) * (1 - p(t)) ** beta,   p(t) = DF(t)/N

    ``IDF(t)`` is the token self-information; ``(1 - DF(t)/N)`` is the
    probability that a match on ``t`` separates the gold document from a random
    corpus distractor -- its *discriminative yield*. Their product keeps only
    tokens that are both informative and discriminative, suppressing corpus-level
    common background in the lexical channel, mirroring the dense channel's
    generic-subspace projection. ``beta = 0`` recovers the plain-IDF weight.
    """
    idf = stats.idf.get(token, 0.0)
    if idf <= 0.0:
        return 0.0
    n_docs = len(stats.doc_tokens)
    if n_docs == 0:
        return float(idf)
    # 1 - p(t) with document prevalence p(t) = df(t)/N (Eq. 11).
    one_minus_p = 1.0 - float(stats.df.get(token, 0)) / n_docs
    if one_minus_p <= 0.0:
        return 0.0
    return float(idf * (one_minus_p ** beta))


def lexical_evidence_scores(
    q_nongeneric: List[str],
    cand_ids: np.ndarray,
    stats: TextStats,
    beta: float = LEXICAL_BETA,
) -> np.ndarray:
    """Lexical relevance score s_lex(q, d): weighted coverage of non-generic
    query tokens, normalized to [0, 1] (Eq. 13).

    Each non-generic query token contributes ``calibrated_token_weight`` w_beta(t)
    (IDF suppressed by document prevalence) instead of raw IDF. Uses only cached
    corpus statistics, so it has the same O(|C|*L) matching cost as plain IDF.
    """
    if not q_nongeneric:
        return np.zeros(cand_ids.shape[0], dtype=np.float32)
    weights = {t: calibrated_token_weight(t, stats, beta) for t in q_nongeneric}
    denom = float(sum(weights.values()))
    if denom <= 1e-8:
        return np.zeros(cand_ids.shape[0], dtype=np.float32)
    q_set = set(q_nongeneric)
    scores = np.zeros(cand_ids.shape[0], dtype=np.float32)
    for j, sid in enumerate(cand_ids):
        inter = q_set & stats.doc_tokens[int(sid)]
        if inter:
            scores[j] = float(sum(weights[t] for t in inter) / denom)
    return scores


def query_lexical_reliability(q_tokens: Set[str], stats: TextStats, eps: float = 1e-8) -> float:
    """Query-adaptive lexical reliability gate lambda_q in [0, 1].

    lambda_q = (specific-IDF mass) / (total in-corpus IDF mass) of the query,
    where "specific" tokens are the in-corpus, non-generic query tokens under
    CLEAR's Otsu generic-token partition. Under a Naive-Bayes bag-of-words
    relevance model, a token's contribution to the log relevance odds scales
    with its IDF and generic tokens have likelihood ratio ~ 1 (no discrimination),
    so lambda_q is the expected fraction of the query's lexical channel that is
    actually informative. Generic-heavy queries automatically down-weight the
    lexical evidence; specific queries get the full lexical channel.
    """
    in_corpus = [t for t in q_tokens if t not in STOPWORDS and stats.df.get(t, 0) > 0]
    if not in_corpus:
        return 0.0
    total = float(sum(stats.idf.get(t, 0.0) for t in in_corpus))
    if total <= eps:
        return 0.0
    specific = float(
        sum(stats.idf.get(t, 0.0) for t in in_corpus if t not in stats.generic_tokens)
    )
    return max(0.0, min(1.0, specific / total))


def skillsight_fusion(
    s_perp: np.ndarray,
    s_lex: np.ndarray,
    lam: float | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Fuse the background-calibrated dense score s_perp and lexical score s_lex.

    The paper's final score (Eq. 15) is the query-independent-gate case:

        s_SkillSight = z(s_perp) + [z(s_lex)]_+

    where each channel is standardized within the candidate set C_q (Eq. 14) and
    [x]_+ = max(x, 0). The background-calibrated dense score s_perp provides the
    primary ranking signal, while above-average lexical matches receive a positive
    correction; if s_lex is constant over C_q the correction vanishes and ranking
    relies solely on the dense channel. This branch is taken when ``lam`` is a
    scalar (``lam`` = 1 reproduces Eq. 15; the query-adaptive ``lambda_q`` from
    ``query_lexical_reliability`` scales the rectified lexical correction).

    Passing ``lam=None`` uses the symmetric dispersion-weighted variant, kept for
    backward compatibility with older experiment configs.
    """
    dense_scores = np.asarray(s_perp, dtype=np.float32)
    lex_scores = np.asarray(s_lex, dtype=np.float32)

    if lam is None:
        # Legacy: local-dispersion z-score fusion.
        dense_std = float(dense_scores.std())
        lex_std = float(lex_scores.std())
        if lex_std <= eps:
            return dense_scores.astype(np.float32)
        if dense_std <= eps:
            return zscore_1d(lex_scores)
        w_dense = dense_std / (dense_std + lex_std + eps)
        w_lex = lex_std / (dense_std + lex_std + eps)
        return (w_dense * zscore_1d(dense_scores) + w_lex * zscore_1d(lex_scores)).astype(np.float32)

    zd = zscore_1d(dense_scores)
    if lam <= 0.0 or float(lex_scores.std()) <= eps:
        return zd.astype(np.float32)
    return (zd + float(lam) * np.maximum(0.0, zscore_1d(lex_scores))).astype(np.float32)
