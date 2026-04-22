from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings
from src.embeddings.base import EmbeddingProvider
from src.io.local_cache import read_npz, write_meta, write_npz
from src.logging_setup import get_logger
from src.utils.batching import throttled_map

logger = get_logger(__name__)


def cache_paths(source: str) -> tuple[Path, Path]:
    base = settings.paths.processed_dir / source / "embeddings"
    stem = f"tags__{settings.embedding.model}"
    return (base / f"{stem}.npz", base / f"{stem}.meta.json")


def load_cached_embeddings(source: str) -> tuple[list[str], np.ndarray]:
    npz_path, _ = cache_paths(source)
    if not npz_path.exists():
        raise FileNotFoundError(f"no tag-embedding cache at {npz_path}")
    cached = read_npz(npz_path)
    return (cached["tag_id"].tolist(), cached["emb"])


def embed_tags(
    source: str, tags_with_desc: pd.DataFrame, provider: EmbeddingProvider, force: bool = False
) -> pd.DataFrame:
    npz_path, meta_path = cache_paths(source)
    required = {"tag_id", "tag_name", "description"}
    missing_cols = required - set(tags_with_desc.columns)
    if missing_cols:
        raise ValueError(f"embed_tags: missing columns {missing_cols}")
    cached_ids: list[str] = []
    cached_emb: np.ndarray | None = None
    if npz_path.exists() and (not force):
        cached = read_npz(npz_path)
        cached_ids = cached["tag_id"].tolist()
        cached_emb = cached["emb"]
    want = tags_with_desc.drop_duplicates("tag_id").reset_index(drop=True)
    cached_set = set(cached_ids)
    missing_mask = ~want["tag_id"].isin(cached_set)
    missing = want[missing_mask]
    logger.info(
        "embed_tags[%s]: %d want, %d cached, %d to embed",
        source,
        len(want),
        len(cached_set),
        len(missing),
    )
    if len(missing):
        texts = (missing["tag_name"] + ". " + missing["description"].fillna("")).tolist()
        new_emb = throttled_map(
            provider.embed_normalized,
            texts,
            batch_size=settings.embedding.batch_size,
            rpm=settings.embedding.rpm,
        )
        new_emb = np.asarray(new_emb, dtype=np.float32)
        all_ids = np.asarray(cached_ids + missing["tag_id"].tolist())
        all_emb = np.vstack([cached_emb, new_emb]) if cached_emb is not None else new_emb
        write_npz(npz_path, tag_id=all_ids, emb=all_emb)
        write_meta(
            meta_path,
            model_id=provider.model_id,
            dim=int(all_emb.shape[1]),
            n=int(all_emb.shape[0]),
        )
        cached_ids = all_ids.tolist()
        cached_emb = all_emb
    assert cached_emb is not None
    id_to_row = {tid: i for i, tid in enumerate(cached_ids)}
    idx = want["tag_id"].map(id_to_row).to_numpy()
    want["embedding"] = list(cached_emb[idx])
    return want
