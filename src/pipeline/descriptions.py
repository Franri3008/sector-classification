from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.config import settings
from src.io.local_cache import read_parquet, write_parquet
from src.llm.base import LLMClient
from src.llm.schemas import DescriptionResult
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

    has_provided = "description" in tags.columns
    unique_cols = ["tag_id", "tag_name"] + (["description"] if has_provided else [])
    unique_tags = tags.drop_duplicates("tag_id")[unique_cols]

    missing = unique_tags[~unique_tags["tag_id"].isin(existing["tag_id"])]
    logger.info(
        "descriptions[%s]: %d total, %d cached, %d to %s",
        source,
        len(unique_tags),
        len(existing),
        len(missing),
        "seed from source" if has_provided else "generate via LLM",
    )

    new_rows: list[dict] = []

    def _checkpoint() -> None:
        nonlocal existing, new_rows
        if not new_rows:
            return
        existing = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        write_parquet(existing, cache_path)
        new_rows = []

    if has_provided:
        for row in missing.itertuples(index=False):
            new_rows.append(
                {
                    "tag_id": row.tag_id,
                    "tag_name": row.tag_name,
                    "description": row.description,
                    "llm_model": "<provided>",
                }
            )
    elif len(missing) > 0:
        concurrency = max(1, settings.llm.max_concurrency)
        logger.info(
            "describe[%s]: %d tags to call LLM (concurrency=%d)",
            source,
            len(missing),
            concurrency,
        )
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(llm.describe, row.tag_name, source): row
                for row in missing.itertuples(index=False)
            }
            bar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"describe[{source}]",
                unit="tag",
                leave=False,
            )
            for fut in bar:
                row = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    logger.warning("describe failed for %r: %s", row.tag_name, e)
                    result = DescriptionResult(tag_name=row.tag_name, description="")
                new_rows.append(
                    {
                        "tag_id": row.tag_id,
                        "tag_name": row.tag_name,
                        "description": result.description,
                        "llm_model": llm.model_id,
                    }
                )
                if len(new_rows) >= 50:
                    _checkpoint()

    _checkpoint()

    return existing.merge(
        unique_tags[["tag_id", "tag_name"]], on=["tag_id", "tag_name"], how="right"
    )
