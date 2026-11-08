from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
PACKAGE = importlib.import_module(ROOT.name)
CLI = importlib.import_module(f"{ROOT.name}.cli")
CONSTANTS = importlib.import_module(f"{ROOT.name}.constants")
CACHE = importlib.import_module(f"{ROOT.name}.cache")
RETRIEVER = importlib.import_module(f"{ROOT.name}.retriever")
TEXT_MODEL = importlib.import_module(f"{ROOT.name}.text_model")
BACKGROUND = importlib.import_module(f"{ROOT.name}.analysis.background_energy")


class ReproductionTests(unittest.TestCase):
    def test_cli_presets_and_deduplication(self) -> None:
        defaults = CLI.parse_args([])
        self.assertEqual(defaults.encoder_preset, "qwen3")
        self.assertEqual(defaults.max_seq_length, 32768)
        self.assertTrue(Path(defaults.model_path).is_absolute())
        self.assertEqual(defaults.variants, ["skillsight"])
        self.assertEqual(defaults.skillrouter_tiers, ["easy", "hard"])

        args = CLI.parse_args(
            [
                "--encoder-preset",
                "skillret",
                "--variants",
                "dense",
                "skillsight",
                "dense",
                "--skillrouter-tiers",
                "easy",
                "easy",
                "--doc-batch",
                "2",
            ]
        )
        self.assertEqual(args.max_seq_length, 8192)
        self.assertEqual(args.doc_batch, 2)
        self.assertEqual(args.variants, ["dense", "skillsight"])
        self.assertEqual(args.skillrouter_tiers, ["easy"])

    def test_skillrouter_tier_changes_cache_name(self) -> None:
        easy_paths = CACHE.candidate_embedding_cache_paths(
            cache_dir=Path("cache"),
            kind="doc_all",
            dataset="skillrouter_easy",
            model_path="model",
            max_seq_length=4096,
            ids_or_texts=["a", "b"],
        )
        hard_paths = CACHE.candidate_embedding_cache_paths(
            cache_dir=Path("cache"),
            kind="doc_all",
            dataset="skillrouter_hard",
            model_path="model",
            max_seq_length=4096,
            ids_or_texts=["a", "b", "c"],
        )
        self.assertIn("skillrouter_easy", easy_paths[-1].name)
        self.assertIn("skillrouter_hard", hard_paths[-1].name)
        self.assertNotEqual(easy_paths, hard_paths)

    def test_all_retrieval_variants_execute(self) -> None:
        doc_texts = ["alpha", "beta", "gamma", "delta"]
        docs = np.eye(4, dtype=np.float32)
        tokens = [{"alpha"}, {"beta"}, {"gamma"}, {"delta"}]
        stats = TEXT_MODEL.TextStats(
            doc_tokens=tokens,
            df={token: 1 for token in ("alpha", "beta", "gamma", "delta")},
            idf={token: 2.0 for token in ("alpha", "beta", "gamma", "delta")},
            generic_tokens=set(),
            token_to_generic_index={},
            generic_token_matrix=np.zeros((0, 4), dtype=np.float32),
            corpus_mean=np.full(4, 0.5, dtype=np.float32),
            idf_threshold=0.0,
            background_rank=0,
        )
        retriever = RETRIEVER.SkillSightRetriever(
            doc_texts=doc_texts,
            doc_embeddings=docs,
            skill_ids=["a", "b", "c", "d"],
            candidate_k=4,
            score_batch=2,
            prebuilt_text_model=(stats, np.zeros((4, 0), dtype=np.float32)),
        )
        results = retriever.score(
            query_texts=["alpha"],
            query_embeddings=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            final_k=2,
            variants=list(CONSTANTS.ALL_VARIANTS),
        )
        self.assertEqual(set(results), set(CONSTANTS.ALL_VARIANTS))
        for variant, result in results.items():
            self.assertEqual(result.variant, variant)
            self.assertEqual(result.top_indices.shape, (1, 2))
            self.assertEqual(result.top_scores.shape, (1, 2))
            self.assertEqual(result.top_indices[0, 0], 0)

    def test_default_variant_remains_backward_compatible(self) -> None:
        stats = TEXT_MODEL.TextStats(
            doc_tokens=[{"alpha"}],
            df={"alpha": 1},
            idf={"alpha": 2.0},
            generic_tokens=set(),
            token_to_generic_index={},
            generic_token_matrix=np.zeros((0, 2), dtype=np.float32),
            corpus_mean=np.asarray([0.5, 0.5], dtype=np.float32),
            idf_threshold=0.0,
        )
        retriever = RETRIEVER.SkillSightRetriever(
            doc_texts=["alpha"],
            doc_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            skill_ids=["a"],
            prebuilt_text_model=(stats, np.zeros((2, 0), dtype=np.float32)),
        )
        results = retriever.score(
            query_texts=["alpha"],
            query_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            final_k=1,
        )
        self.assertEqual(list(results), [CONSTANTS.VARIANT_SKILLSIGHT])

    def test_background_energy_is_squared_projection(self) -> None:
        vectors = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        basis = np.asarray([[0.0], [1.0]], dtype=np.float32)
        values = BACKGROUND.background_energy(vectors, basis)
        np.testing.assert_allclose(values, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
