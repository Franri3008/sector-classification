from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.embeddings.base import EmbeddingProvider
from src.embeddings.customtools_provider import CustomToolsEmbeddingProvider
from src.llm.base import LLMClient
from src.llm.factory import build_llm_client
from src.logging_setup import get_logger
from src.pipeline.classify import classify_tags
from src.pipeline.descriptions import ensure_descriptions
from src.pipeline.embed_tags import embed_tags
from src.pipeline.postprocess import build_output
from src.pipeline.ranking import top_candidates
from src.pipeline.sectors import embed_sectors
from src.pipeline.summary import RunSummary
from src.sources.registry import get_adapter, list_sources

logger = get_logger(__name__)


def _default_providers() -> tuple[EmbeddingProvider, LLMClient]:
    return (CustomToolsEmbeddingProvider(), build_llm_client())


def run_source(
    source: str,
    *,
    embedder: EmbeddingProvider | None = None,
    llm: LLMClient | None = None,
    force: bool = False,
    skip_embed: bool = False,
) -> RunSummary:
    if embedder is None or llm is None:
        default_emb, default_llm = _default_providers()
        embedder = embedder or default_emb
        llm = llm or default_llm
    logger.info("=== run_source: %s ===", source)
    t0 = time.perf_counter()
    adapter = get_adapter(source)
    records = adapter.load_records()
    tags_to_keys = adapter.extract_tags(records)
    logger.info(
        "%s: %d (tag, key) rows, %d unique tags",
        source,
        len(tags_to_keys),
        tags_to_keys["tag_id"].nunique(),
    )
    keep_cols = ["tag_id", "tag_name"] + (
        ["description"] if "description" in tags_to_keys.columns else []
    )
    unique_tags = tags_to_keys.drop_duplicates("tag_id")[keep_cols]
    tags_with_desc = ensure_descriptions(source, unique_tags, llm, force=force)
    sectors = embed_sectors(embedder, force=force)
    sector_emb = np.stack(sectors["embedding"].to_numpy())
    if skip_embed:
        from src.pipeline.embed_tags import load_cached_embeddings

        ids, emb = load_cached_embeddings(source)
        id_to_row = {tid: i for i, tid in enumerate(ids)}
        idx = tags_with_desc["tag_id"].map(id_to_row).to_numpy()
        tags_with_desc = tags_with_desc.copy()
        tags_with_desc["embedding"] = list(emb[idx])
    else:
        tags_with_desc = embed_tags(source, tags_with_desc, embedder, force=force)
    tag_emb = np.stack(tags_with_desc["embedding"].to_numpy())
    candidates = top_candidates(tag_emb, sector_emb, sectors)
    classifications = classify_tags(source, tags_with_desc, candidates, llm, force=force)
    output_path, output_rows = build_output(source, tags_to_keys, classifications)
    elapsed = time.perf_counter() - t0

    n_unique = int(tags_to_keys["tag_id"].nunique())
    n_with_picks = int(
        classifications.dropna(subset=["division_code"])["tag_id"].nunique()
    )
    return RunSummary(
        source=source,
        output_path=output_path,
        input_rows=len(tags_to_keys),
        unique_tags=n_unique,
        tags_with_picks=n_with_picks,
        tags_no_pick=n_unique - n_with_picks,
        output_rows=output_rows,
        elapsed_s=elapsed,
    )


def run_all(
    *, embedder: EmbeddingProvider | None = None, llm: LLMClient | None = None, force: bool = False
) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for src in list_sources():
        summaries.append(run_source(src, embedder=embedder, llm=llm, force=force))
    return summaries


_ = pd
