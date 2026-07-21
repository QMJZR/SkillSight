"""Command-line interface for SkillSight."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .constants import (
    DEFAULT_DATASETS,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TOP_K,
    LEXICAL_BETA,
    MAX_TOKENS_PER_DOC,
    REPO_ROOT_DEFAULT,
    VARIANT_SKILLSIGHT,
)
from .evaluation import run_evaluation

SKILLROUTER_ROOT_DEFAULT = Path("./SkillRouter-Eval-Core")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "SkillSight: skill retrieval via generic-subspace debiasing and "
            "discriminativeness-weighted lexical-dense fusion."
        )
    )
    ap.add_argument(
        "--benchmark",
        choices=["sra", "skillrouter"],
        default="sra",
        help="sra: SR-Agents corpus.json. skillrouter: SkillRouter-Eval-Core 'all' pool (deduped easy+hard).",
    )
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    ap.add_argument(
        "--skillrouter-root",
        type=Path,
        default=SKILLROUTER_ROOT_DEFAULT,
        help="SkillRouter-Eval-Core dataset root (used when --benchmark skillrouter).",
    )
    ap.add_argument(
        "--include-generic-only",
        action="store_true",
        help="SkillRouter only: keep generic_only tasks (default: drop them).",
    )
    ap.add_argument(
        "--include-excluded",
        action="store_true",
        help=(
            "SkillRouter only: keep tasks flagged excluded=True (e.g. timeout / "
            "requires-GPU tasks). These are valid retrieval queries; the flag "
            "recovers the official 75-task set (73 + 2 excluded)."
        ),
    )
    ap.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--top-k", nargs="+", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--primary-k", type=int, default=10)
    ap.add_argument("--doc-batch", type=int, default=4)
    ap.add_argument("--query-batch", type=int, default=64)
    ap.add_argument("--score-batch", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    ap.add_argument(
        "--max-tokens-per-doc",
        type=int,
        default=MAX_TOKENS_PER_DOC,
        help="Cap on unique tokens kept per document when building the generic text model (<=0 disables).",
    )
    ap.add_argument(
        "--lexical-beta",
        type=float,
        default=LEXICAL_BETA,
        help=(
            "Lexical discriminativeness-suppression strength beta in "
            "w(t)=IDF(t)*(1-DF(t)/N)^beta. 0 recovers plain-IDF lexical evidence."
        ),
    )
    ap.add_argument("--cache-dir", type=Path, default=Path("./cache"))
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("./results"))
    ap.add_argument("--output-json", type=Path, default=None, help="Optional explicit JSON output path.")
    ap.add_argument(
        "--candidate-k",
        type=int,
        default=300,
        help="Candidate depth for the dense + invariant rerank union.",
    )

    args = ap.parse_args(argv)
    args.variants = [VARIANT_SKILLSIGHT]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_evaluation(args)


if __name__ == "__main__":
    main()
