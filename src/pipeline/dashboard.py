from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import settings
from src.logging_setup import get_logger
from src.pipeline.sectors import load_sectors
from src.sources.registry import list_sources

logger = get_logger(__name__)

_SOURCE_COLORS = {
    "crunchbase": "#2756d3",
    "regpat": "#E53229",
    "openalex": "#f59e0b",
}


def _latest_output(source: str) -> Path | None:
    out_dir = settings.paths.outputs_dir
    if not out_dir.exists():
        return None
    matches = sorted(
        out_dir.glob(f"{source}__*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return matches[0] if matches else None


def build_dashboard_data() -> Path:
    dashboard_dir = settings.paths.root / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    sectors = load_sectors()
    name_to_section = dict(
        zip(sectors["division_name"], sectors["section_code"], strict=True)
    )

    sources_data: list[dict] = []
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
    }

    out_path = dashboard_dir / "data.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "dashboard: wrote %d sources, %d divisions, %d sections → %s",
        len(sources_data),
        len(sectors_list),
        len(sections_list),
        out_path,
    )
    return out_path
