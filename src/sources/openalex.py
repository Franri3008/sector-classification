from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings
from src.sources.base import SourceAdapter


class OpenAlexAdapter(SourceAdapter):
    name = "openalex"
    key_column = "work_id"
    tag_column = "concept_display_name"

    def raw_files(self) -> list[Path]:
        folder = settings.paths.raw_dir / self.name
        return sorted(folder.glob("*.csv")) + sorted(folder.glob("*.parquet"))

    def load_records(self) -> pd.DataFrame:
        frames = []
        for p in self.raw_files():
            if p.suffix == ".parquet":
                frames.append(pd.read_parquet(p))
            else:
                frames.append(pd.read_csv(p))
        if not frames:
            raise FileNotFoundError(f"No raw files in {settings.paths.raw_dir / self.name}.")
        return pd.concat(frames, ignore_index=True)

    def extract_tags(self, records: pd.DataFrame) -> pd.DataFrame:
        sub = records[[self.key_column, self.tag_column]].dropna()
        sub = sub.rename(columns={self.key_column: "key", self.tag_column: "tag_name"})
        sub["tag_name"] = sub["tag_name"].astype(str).str.strip()
        sub = sub[sub["tag_name"] != ""]
        return self.assign_tag_ids(sub)[["tag_id", "tag_name", "key"]]
