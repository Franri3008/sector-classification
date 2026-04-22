from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings
from src.embeddings.base import EmbeddingProvider
from src.io.local_cache import read_npz, write_meta, write_npz
from src.logging_setup import get_logger
from src.utils.batching import throttled_map
from src.utils.hashing import file_hash

logger = get_logger(__name__)


def sectors_hash() -> str:
    return file_hash(settings.paths.sectors_csv)


def load_sectors() -> pd.DataFrame:
    df = pd.read_csv(settings.paths.sectors_csv, dtype={"division_code": str})
    df["text"] = (
        df["division_name"].fillna("")
        + " — section "
        + df["section_code"].fillna("")
        + ": "
        + df["section_name"].fillna("")
    )
    return df


def _cache_paths() -> tuple[Path, Path]:
    base = settings.paths.processed_dir / "sectors"
    stem = f"sectors__{settings.embedding.model}__{sectors_hash()}"
    return (base / f"{stem}.npz", base / f"{stem}.meta.json")


def embed_sectors(provider: EmbeddingProvider, force: bool = False) -> pd.DataFrame:
    npz_path, meta_path = _cache_paths()
    sectors = load_sectors()
    if npz_path.exists() and (not force):
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
    texts = sectors["text"].tolist()
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
        n=int(emb.shape[0]),
    )
    sectors["embedding"] = list(emb)
    return sectors
