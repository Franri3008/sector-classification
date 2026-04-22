from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import settings


def top_candidates(
    tag_embeddings: np.ndarray,
    sector_embeddings: np.ndarray,
    sector_meta: pd.DataFrame,
    top_n: int | None = None,
    min_similarity: float | None = None,
) -> list[list[dict]]:
    if top_n is None:
        top_n = settings.ranking.top_n
    if min_similarity is None:
        min_similarity = settings.ranking.min_similarity
    sims = tag_embeddings @ sector_embeddings.T
    out: list[list[dict]] = []
    for row in sims:
        n = min(top_n, row.size)
        top_idx = np.argpartition(-row, n - 1)[:n]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        picks = []
        for i in top_idx:
            score = float(row[i])
            if score < min_similarity:
                continue
            meta = sector_meta.iloc[int(i)]
            picks.append(
                {
                    "division_code": meta["division_code"],
                    "division_name": meta["division_name"],
                    "section_code": meta["section_code"],
                    "section_name": meta["section_name"],
                    "similarity": score,
                }
            )
        out.append(picks)
    return out
