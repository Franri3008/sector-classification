from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.config import settings
from src.embeddings.base import EmbeddingProvider
from src.logging_setup import get_logger
from src.utils.retry import retryable

logger = get_logger(__name__)


class CustomToolsEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model_id = model or settings.embedding.model
        self.dim = settings.embedding.dim
        self._api_key = api_key or settings.openai_api_key.get_secret_value()

    def _ensure_env(self) -> None:
        if os.getenv("OPENAI_API_KEY"):
            return
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is empty — set it in .env before calling CustomTools."
            )
        os.environ["OPENAI_API_KEY"] = self._api_key

    @retryable()
    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_env()
        from CustomTools.llm.embedding import embed as ct_embed

        logger.debug("CustomTools.embed n=%d model=%s", len(texts), self.model_id)
        df = pd.DataFrame({"text": texts})
        out = ct_embed(df, model=self.model_id)
        dim_cols = [c for c in out.columns if c.startswith("d") and c[1:].isdigit()]
        dim_cols.sort(key=lambda c: int(c[1:]))
        arr = out[dim_cols].to_numpy(dtype=np.float32)
        self.dim = arr.shape[1]
        return arr
