from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.config import settings
from src.io.local_cache import read_parquet, write_parquet
from src.llm.base import LLMClient
from src.logging_setup import get_logger
from src.pipeline.sectors import sectors_hash

logger = get_logger(__name__)


def _cache_path(source: str) -> Path:
    ranking = settings.ranking
    stem = f"{settings.llm.backend}__{_model_id()}__{sectors_hash()}__topN{ranking.top_n}_floor{ranking.min_similarity}"
    return settings.paths.processed_dir / source / "classifications" / f"{stem}.parquet"


def _model_id() -> str:
    return (
        settings.llm.openai_model if settings.llm.backend == "openai" else settings.llm.vllm_model
    )


def classify_tags(
    source: str,
    tags: pd.DataFrame,
    candidates: list[list[dict]],
    llm: LLMClient,
    force: bool = False,
) -> pd.DataFrame:
    if len(tags) != len(candidates):
        raise ValueError("tags and candidates must have equal length")
    cache_path = _cache_path(source)
    cached = (
        read_parquet(cache_path)
        if cache_path.exists() and (not force)
        else pd.DataFrame(columns=["tag_id", "division_code", "reason", "similarity", "raw_json"])
    )
    cached_ids = set(cached["tag_id"].unique())
    new_rows: list[dict] = []
    tags_since_checkpoint = 0
    todo = len(tags) - sum(1 for t in tags["tag_id"] if t in cached_ids)
    bar = tqdm(
        zip(tags.itertuples(index=False), candidates, strict=True),
        total=todo,
        desc=f"classify[{source}]",
        unit="tag",
        leave=False,
    )
    for row, cands in bar:
        if row.tag_id in cached_ids:
            continue
        if not cands:
            new_rows.append(
                {
                    "tag_id": row.tag_id,
                    "division_code": None,
                    "reason": None,
                    "similarity": None,
                    "raw_json": "{}",
                }
            )
            tags_since_checkpoint += 1
            continue
        result = llm.classify(row.tag_name, row.description, cands)
        raw = result.model_dump_json()
        sim_by_code = {c["division_code"]: c["similarity"] for c in cands}
        if not result.picks:
            new_rows.append(
                {
                    "tag_id": row.tag_id,
                    "division_code": None,
                    "reason": None,
                    "similarity": None,
                    "raw_json": raw,
                }
            )
        else:
            for pick in result.picks:
                new_rows.append(
                    {
                        "tag_id": row.tag_id,
                        "division_code": pick.division_code,
                        "reason": pick.reason,
                        "similarity": sim_by_code.get(pick.division_code),
                        "raw_json": raw,
                    }
                )
        tags_since_checkpoint += 1
        if tags_since_checkpoint >= 50:
            cached = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
            write_parquet(cached, cache_path)
            cached_ids.update(r["tag_id"] for r in new_rows)
            new_rows = []
            tags_since_checkpoint = 0
    if new_rows:
        cached = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
        write_parquet(cached, cache_path)
    wanted = set(tags["tag_id"])
    return cached[cached["tag_id"].isin(wanted)].reset_index(drop=True)


def load_classifications(source: str) -> pd.DataFrame:
    path = _cache_path(source)
    if not path.exists():
        raise FileNotFoundError(f"no classification cache at {path}")
    return read_parquet(path)


def classification_cache_path(source: str) -> Path:
    return _cache_path(source)


_ = json
