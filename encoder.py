"""Dense encoder wrapper used by SkillSight."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch


class DenseEncoder:
    def __init__(self, model_path: str, device: str = "cuda:0", max_seq_length: int = 4096):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_path,
            device=device,
            trust_remote_code=True,
            model_kwargs={"attn_implementation": "sdpa"},
        )
        self.model.max_seq_length = max_seq_length
        self.device = device
        self.max_seq_length = max_seq_length
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[encoder] loaded: {model_path}", flush=True)
        print(f"[encoder] dim={self.dim} max_seq_length={max_seq_length} device={device}", flush=True)

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = True,
        prompt_name: Optional[str] = None,
    ) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt_name=prompt_name,
        )
        return emb.astype(np.float32)
