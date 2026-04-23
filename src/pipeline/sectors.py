from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import settings
from src.embeddings.base import EmbeddingProvider
from src.io.local_cache import read_npz, read_parquet, write_meta, write_npz, write_parquet
from src.llm.base import LLMClient
from src.logging_setup import get_logger
from src.utils.batching import throttled_map
from src.utils.hashing import file_hash, stable_hash

logger = get_logger(__name__)


def sectors_hash() -> str:
    return file_hash(settings.paths.sectors_csv)


def load_sectors() -> pd.DataFrame:
    return pd.read_csv(settings.paths.sectors_csv, dtype={"division_code": str})


def _enrichment_cache_path(llm_model: str) -> Path:
    base = settings.paths.processed_dir / "sectors"
    stem = f"enrichment__{llm_model}__{sectors_hash()}"
    return base / f"{stem}.parquet"


def _join_keywords(keywords: list[str]) -> str:
    return ", ".join(k.strip() for k in keywords if k and k.strip())


def enrich_sectors(llm: LLMClient, force: bool = False) -> pd.DataFrame:
    cache_path = _enrichment_cache_path(llm.model_id)
    sectors = load_sectors()
    if cache_path.exists() and not force:
        cached = read_parquet(cache_path)
        if set(cached["division_code"]) >= set(sectors["division_code"]):
            logger.info("enrich_sectors cache hit: %s", cache_path.name)
            return cached

    logger.info("enriching %d sectors via %s (per-section batches)", len(sectors), llm.model_id)
    groups = list(sectors.groupby("section_code", sort=True))

    concurrency = max(1, settings.llm.max_concurrency)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for section_code, group in groups:
            divisions = [
                {"division_code": d, "division_name": n}
                for d, n in zip(group["division_code"], group["division_name"], strict=True)
            ]
            section_name = str(group["section_name"].iloc[0])
            futures[
                pool.submit(llm.enrich_sectors, str(section_code), section_name, divisions)
            ] = (section_code, group)
        bar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="enrich-sectors",
            unit="section",
            leave=False,
        )
        for fut in bar:
            section_code, group = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                logger.warning("enrich_sectors failed for section %s: %s", section_code, e)
                continue
            by_code = {d.division_code: d for d in result.divisions}
            for d, n in zip(group["division_code"], group["division_name"], strict=True):
                entry = by_code.get(str(d))
                if entry is None:
                    logger.warning(
                        "enrich_sectors missing division %s in section %s response", d, section_code
                    )
                    continue
                rows.append(
                    {
                        "division_code": str(d),
                        "division_name": str(n),
                        "section_code": str(section_code),
                        "broad_keywords": _join_keywords(entry.broad_keywords),
                        "distinctive_keywords": _join_keywords(entry.distinctive_keywords),
                        "llm_model": llm.model_id,
                    }
                )

    enriched = pd.DataFrame(rows)
    write_parquet(enriched, cache_path)
    return enriched


def _build_embed_text(row: pd.Series) -> str:
    parts = [str(row["division_name"]), f"(section {row['section_code']}: {row['section_name']})"]
    broad = row.get("broad_keywords") or ""
    distinctive = row.get("distinctive_keywords") or ""
    if broad:
        parts.append(f"Covers: {broad}.")
    if distinctive:
        parts.append(f"Specifically: {distinctive}.")
    return " ".join(parts)


def sectors_with_embed_text(enriched: pd.DataFrame | None = None) -> pd.DataFrame:
    sectors = load_sectors()
    if enriched is not None and not enriched.empty:
        enrich_cols = ["division_code", "broad_keywords", "distinctive_keywords"]
        sectors = sectors.merge(enriched[enrich_cols], on="division_code", how="left")
    else:
        sectors["broad_keywords"] = ""
        sectors["distinctive_keywords"] = ""
    sectors["text"] = sectors.apply(_build_embed_text, axis=1)
    return sectors


def _text_hash(texts: list[str]) -> str:
    return stable_hash("\n".join(texts))


def _cache_paths(text_hash: str | None = None) -> tuple[Path, Path]:
    base = settings.paths.processed_dir / "sectors"
    if text_hash is None:
        stem = f"sectors__{settings.embedding.model}__{sectors_hash()}"
    else:
        stem = f"sectors__{settings.embedding.model}__{sectors_hash()}__{text_hash}"
    return (base / f"{stem}.npz", base / f"{stem}.meta.json")


def latest_embedding_cache() -> Path | None:
    base = settings.paths.processed_dir / "sectors"
    if not base.exists():
        return None
    prefix = f"sectors__{settings.embedding.model}__{sectors_hash()}"
    matches = sorted(base.glob(f"{prefix}*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def embed_sectors(
    provider: EmbeddingProvider, llm: LLMClient | None = None, force: bool = False
) -> pd.DataFrame:
    enriched = enrich_sectors(llm, force=force) if llm is not None else None
    sectors = sectors_with_embed_text(enriched)
    texts = sectors["text"].tolist()
    text_hash = _text_hash(texts)
    npz_path, meta_path = _cache_paths(text_hash)

    if npz_path.exists() and not force:
        logger.info("embed_sectors cache hit: %s", npz_path.name)
        cached = read_npz(npz_path)
        codes = cached["division_code"].tolist()
        emb = cached["emb"]
        order = pd.Index(codes)
        idx = order.get_indexer(sectors["division_code"].tolist())
        if (idx >= 0).all():
            sectors["embedding"] = list(emb[idx])
            return sectors

    logger.info("embedding %d sectors with %s", len(sectors), provider.model_id)
    emb_batches = throttled_map(
        provider.embed_normalized,
        texts,
        batch_size=settings.embedding.batch_size,
        rpm=settings.embedding.rpm,
    )
    emb = np.asarray(emb_batches, dtype=np.float32)
    write_npz(npz_path, division_code=np.asarray(sectors["division_code"].tolist()), emb=emb)
    write_meta(
        meta_path,
        model_id=provider.model_id,
        dim=int(emb.shape[1]),
        sectors_hash=sectors_hash(),
        text_hash=text_hash,
        n=int(emb.shape[0]),
    )
    sectors["embedding"] = list(emb)
    return sectors
