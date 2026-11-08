"""Dense encoder wrapper used by SkillSight."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


SKILLROUTER_QUERY_INSTRUCTION = (
    "Instruct: Given a task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)


class DenseEncoder:
    def __init__(self, model_path: str, device: str = "cuda:0", max_seq_length: int = 4096):
        self.device = device
        self.max_seq_length = max_seq_length
        local_path = Path(model_path)
        self.native_transformers = (
            local_path.is_dir()
            and (local_path / "config.json").exists()
            and not (local_path / "config_sentence_transformers.json").exists()
        )

        if self.native_transformers:
            from transformers import AutoModel, AutoTokenizer

            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left",
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            ).to(device).eval()
            self.dim = int(self.model.config.hidden_size)
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_path,
                device=device,
                trust_remote_code=True,
                model_kwargs={"attn_implementation": "sdpa"},
            )
            self.model.max_seq_length = max_seq_length
            self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[encoder] loaded: {model_path}", flush=True)
        print(
            f"[encoder] dim={self.dim} max_seq_length={max_seq_length} "
            f"device={device} native_transformers={self.native_transformers}",
            flush=True,
        )

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = True,
        prompt_name: Optional[str] = None,
    ) -> np.ndarray:
        if self.native_transformers:
            texts = list(texts)
            if prompt_name == "query":
                texts = [SKILLROUTER_QUERY_INSTRUCTION + text for text in texts]
            # Match SentenceTransformer's length-sorted batching behavior so
            # long and short documents do not share large padded batches.
            order = sorted(range(len(texts)), key=lambda idx: len(texts[idx]))
            ordered_texts = [texts[idx] for idx in order]
            batches = []
            for start in tqdm(
                range(0, len(ordered_texts), batch_size),
                desc="  encode",
                disable=not show_progress,
                unit="batch",
            ):
                encoded = self.tokenizer(
                    ordered_texts[start:start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.no_grad():
                    hidden = self.model(**encoded).last_hidden_state
                    padding_is_left = bool(encoded["attention_mask"][:, -1].all())
                    if padding_is_left:
                        embeddings = hidden[:, -1]
                    else:
                        lengths = encoded["attention_mask"].sum(dim=1) - 1
                        row_ids = torch.arange(hidden.shape[0], device=hidden.device)
                        embeddings = hidden[row_ids, lengths]
                    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
                batches.append(embeddings.cpu().numpy())
            encoded_embeddings = np.vstack(batches).astype(np.float32)
            restored = np.empty_like(encoded_embeddings)
            restored[order] = encoded_embeddings
            return restored

        emb = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt_name=prompt_name,
        )
        return emb.astype(np.float32)
