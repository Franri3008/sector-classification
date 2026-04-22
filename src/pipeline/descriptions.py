from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings
from src.io.local_cache import read_parquet, write_parquet
from src.llm.base import LLMClient
from src.logging_setup import get_logger

logger = get_logger(__name__)


def _cache_path(source: str) -> Path:
    return settings.paths.processed_dir / source / "descriptions" / "descriptions.parquet"


def ensure_descriptions(
    source: str, tags: pd.DataFrame, llm: LLMClient, force: bool = False
) -> pd.DataFrame:
    cache_path = _cache_path(source)
    existing = (
        read_parquet(cache_path)
        if cache_path.exists() and (not force)
        else pd.DataFrame(columns=["tag_id", "tag_name", "description", "llm_model"])
    )
    unique_tags = tags.drop_duplicates("tag_id")[["tag_id", "tag_name"]]
    missing = unique_tags[~unique_tags["tag_id"].isin(existing["tag_id"])]
    logger.info(
        "descriptions: %d total, %d cached, %d to generate",
        len(unique_tags),
        len(existing),
        len(missing),
    )
    new_rows = []
    for row in missing.itertuples(index=False):
        result = llm.describe(row.tag_name, source)
        new_rows.append(
            {
                "tag_id": row.tag_id,
                "tag_name": row.tag_name,
                "description": result.description,
                "llm_model": llm.model_id,
            }
        )
        if len(new_rows) >= 50:
            existing = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            write_parquet(existing, cache_path)
            new_rows = []
    if new_rows:
        existing = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        write_parquet(existing, cache_path)
    return existing.merge(unique_tags, on=["tag_id", "tag_name"], how="right")
