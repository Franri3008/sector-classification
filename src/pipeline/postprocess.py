from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import settings
from src.logging_setup import get_logger
from src.pipeline.sectors import load_sectors

logger = get_logger(__name__)


def build_output(
    source: str,
    tags_to_keys: pd.DataFrame,
    classifications: pd.DataFrame,
    run_id: str | None = None,
) -> Path:
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sectors = load_sectors()[["division_code", "division_name"]]
    picks = classifications.dropna(subset=["division_code"])[["tag_id", "division_code"]]
    merged = (
        tags_to_keys.merge(picks, on="tag_id", how="inner")
        .merge(sectors, on="division_code", how="left")
    )
    out = (
        merged.rename(columns={"division_name": "sector"})[["sector", "key"]]
        .dropna(subset=["sector"])
        .drop_duplicates()
        .sort_values(["key", "sector"])
        .reset_index(drop=True)
    )
    out_path = settings.paths.outputs_dir / f"{source}__{run_id}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("wrote %d rows to %s", len(out), out_path)
    return out_path
