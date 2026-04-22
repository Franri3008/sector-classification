from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    model_id: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        pass

    def embed_normalized(self, texts: list[str]) -> np.ndarray:
        emb = self.embed(texts)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return emb / norms
