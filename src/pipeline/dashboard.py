from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings
from src.io.local_cache import read_npz, read_parquet
from src.logging_setup import get_logger
from src.pipeline.classify import classification_cache_path
from src.pipeline.descriptions import _cache_path as descriptions_cache_path
from src.pipeline.embed_tags import cache_paths as tag_embedding_cache_paths
from src.pipeline.sectors import latest_embedding_cache, load_sectors
from src.sources.registry import get_adapter, list_sources

logger = get_logger(__name__)

_SOURCE_COLORS = {
    "crunchbase": "#2756d3",
    "regpat": "#E53229",
    "openalex": "#f59e0b",
}

_TRACE_TOP_CANDIDATES = 15
_TRACE_MAX_KEYS_PER_TAG = 50


def _latest_output(source: str) -> Path | None:
    out_dir = settings.paths.outputs_dir
    if not out_dir.exists():
        return None
    matches = sorted(
        out_dir.glob(f"{source}__*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return matches[0] if matches else None


def _load_sector_matrix(sectors: pd.DataFrame) -> tuple[list[str], np.ndarray] | None:
    path = latest_embedding_cache()
    if path is None or not path.exists():
        return None
    cached = read_npz(path)
    codes = [str(c) for c in cached["division_code"].tolist()]
    emb = cached["emb"]
    order = pd.Index(codes)
    idx = order.get_indexer(sectors["division_code"].tolist())
    if (idx < 0).any():
        return None
    return (sectors["division_code"].tolist(), emb[idx])


def _build_trace_tags(
    source: str,
    sectors: pd.DataFrame,
    sector_codes: list[str],
    sector_emb: np.ndarray,
) -> list[dict]:
    desc_path = descriptions_cache_path(source)
    cls_path = classification_cache_path(source)
    tag_emb_path, _ = tag_embedding_cache_paths(source)
    if not (desc_path.exists() and tag_emb_path.exists()):
        logger.warning("trace[%s]: missing description or tag-embedding cache, skipping", source)
        return []

    descriptions = read_parquet(desc_path)[["tag_id", "tag_name", "description"]]
    classifications = (
        read_parquet(cls_path)[["tag_id", "division_code", "similarity", "reason"]]
        if cls_path.exists()
        else pd.DataFrame(columns=["tag_id", "division_code", "similarity", "reason"])
    )

    keys_by_tag: dict[str, list[str]] = {}
    try:
        adapter = get_adapter(source)
        records = adapter.load_records()
        tag_key_df = adapter.extract_tags(records)[["tag_id", "key"]]
        for tag_id, key in zip(tag_key_df["tag_id"], tag_key_df["key"], strict=True):
            keys_by_tag.setdefault(str(tag_id), []).append(str(key))
    except Exception as e:
        logger.warning(
            "trace[%s]: raw source not available (%s) — keys omitted from trace", source, e
        )

    cached = read_npz(tag_emb_path)
    tag_ids_ordered = [str(t) for t in cached["tag_id"].tolist()]
    tag_emb = cached["emb"]
    tag_index = {tid: i for i, tid in enumerate(tag_ids_ordered)}

    division_name = dict(
        zip(sectors["division_code"], sectors["division_name"], strict=True)
    )

    picks_by_tag: dict[str, list[dict]] = {}
    for row in classifications.dropna(subset=["division_code"]).itertuples(index=False):
        picks_by_tag.setdefault(str(row.tag_id), []).append(
            {
                "code": str(row.division_code),
                "similarity": (
                    float(row.similarity) if pd.notna(row.similarity) else None
                ),
                "reason": str(row.reason) if pd.notna(row.reason) else "",
            }
        )

    sector_codes_np = np.asarray(sector_codes, dtype=object)
    top_k = min(_TRACE_TOP_CANDIDATES, len(sector_codes))

    out: list[dict] = []
    for tag_row in descriptions.itertuples(index=False):
        tag_id = str(tag_row.tag_id)
        emb_idx = tag_index.get(tag_id)
        if emb_idx is None:
            continue
        sims = sector_emb @ tag_emb[emb_idx]
        if top_k < len(sims):
            cand_idx = np.argpartition(-sims, top_k)[:top_k]
            cand_idx = cand_idx[np.argsort(-sims[cand_idx])]
        else:
            cand_idx = np.argsort(-sims)
        picks = picks_by_tag.get(tag_id, [])
        picked_codes = {p["code"] for p in picks}
        candidates = [
            {
                "code": str(sector_codes_np[i]),
                "similarity": float(sims[i]),
                "picked": str(sector_codes_np[i]) in picked_codes,
            }
            for i in cand_idx
        ]
        keys = keys_by_tag.get(tag_id, [])
        truncated_keys = keys[:_TRACE_MAX_KEYS_PER_TAG]
        out.append(
            {
                "tag_id": tag_id,
                "tag_name": str(tag_row.tag_name),
                "description": str(tag_row.description) if pd.notna(tag_row.description) else "",
                "keys": truncated_keys,
                "n_keys": len(keys),
                "picks": [
                    {
                        "code": p["code"],
                        "division_name": division_name.get(p["code"], ""),
                        "similarity": p["similarity"],
                        "reason": p["reason"],
                    }
                    for p in picks
                ],
                "candidates": candidates,
            }
        )
    out.sort(key=lambda t: t["tag_name"].lower())
    logger.info("trace[%s]: built %d tag records", source, len(out))
    return out


def build_dashboard_data() -> Path:
    dashboard_dir = settings.paths.root / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = dashboard_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    sectors = load_sectors()
    name_to_section = dict(
        zip(sectors["division_name"], sectors["section_code"], strict=True)
    )
    sector_matrix = _load_sector_matrix(sectors)
    if sector_matrix is None:
        logger.warning(
            "dashboard: no sector embedding cache — trace mode will show picks only (no candidates)"
        )

    sources_data: list[dict] = []
    trace_sources: list[str] = []
    for source in list_sources():
        csv_path = _latest_output(source)
        if csv_path is None:
            logger.warning("dashboard: no output found for %s", source)
            continue
        df = pd.read_csv(csv_path)
        if not {"sector", "key"}.issubset(df.columns):
            logger.warning("dashboard: %s missing required columns, skipping", csv_path)
            continue

        by_division = df.groupby("sector").size().to_dict()
        by_section: dict[str, int] = {}
        for name, count in by_division.items():
            section = name_to_section.get(name)
            if section is None:
                continue
            by_section[section] = by_section.get(section, 0) + int(count)

        sources_data.append(
            {
                "source": source,
                "color": _SOURCE_COLORS.get(source, "#64748b"),
                "output_path": str(csv_path.relative_to(settings.paths.root)),
                "output_mtime": datetime.fromtimestamp(csv_path.stat().st_mtime, UTC).isoformat(),
                "total_mappings": len(df),
                "unique_keys": int(df["key"].nunique()),
                "sectors_covered": int(df["sector"].nunique()),
                "by_division_name": {str(k): int(v) for k, v in by_division.items()},
                "by_section_code": {str(k): int(v) for k, v in by_section.items()},
            }
        )

        if sector_matrix is not None:
            sector_codes, sector_emb = sector_matrix
            try:
                tags = _build_trace_tags(source, sectors, sector_codes, sector_emb)
            except Exception as e:
                logger.warning("trace[%s]: failed to build tag records: %s", source, e)
                tags = []
            if tags:
                trace_payload = {
                    "source": source,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "tags": tags,
                }
                trace_path = traces_dir / f"{source}.json"
                trace_path.write_text(
                    json.dumps(trace_payload, ensure_ascii=False), encoding="utf-8"
                )
                trace_sources.append(source)
                logger.info(
                    "dashboard: wrote trace for %s (%d tags) → %s",
                    source,
                    len(tags),
                    trace_path,
                )

    sections_seen: set[str] = set()
    sections_list: list[dict] = []
    for _, row in sectors.iterrows():
        code = str(row["section_code"])
        if code in sections_seen:
            continue
        sections_seen.add(code)
        sections_list.append(
            {"section_code": code, "section_name": str(row["section_name"])}
        )

    sectors_list = [
        {
            "division_code": str(row["division_code"]),
            "division_name": str(row["division_name"]),
            "section_code": str(row["section_code"]),
            "section_name": str(row["section_name"]),
        }
        for _, row in sectors.iterrows()
    ]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_divisions": len(sectors_list),
        "n_sections": len(sections_list),
        "sections": sections_list,
        "sectors": sectors_list,
        "sources": sources_data,
        "trace_sources": trace_sources,
    }

    out_path = dashboard_dir / "data.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "dashboard: wrote %d sources (%d with traces) → %s",
        len(sources_data),
        len(trace_sources),
        out_path,
    )
    return out_path
