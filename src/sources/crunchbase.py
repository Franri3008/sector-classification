from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings
from src.sources.base import SourceAdapter


class CrunchbaseAdapter(SourceAdapter):
    name = "crunchbase"
    key_column = "uuid"
    tag_column = "category_list"
    tag_separator = ","

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
            raise FileNotFoundError(
                f"No raw files in {settings.paths.raw_dir / self.name}. Sync from Dropbox first."
            )
        return pd.concat(frames, ignore_index=True)

    def extract_tags(self, records: pd.DataFrame) -> pd.DataFrame:
        sub = records[[self.key_column, self.tag_column]].dropna()
        sub = sub.rename(columns={self.key_column: "key"})
        sub[self.tag_column] = sub[self.tag_column].astype(str)
        exploded = (
            sub.assign(tag_name=sub[self.tag_column].str.split(self.tag_separator))
            .explode("tag_name")
            .drop(columns=[self.tag_column])
        )
        exploded["tag_name"] = exploded["tag_name"].str.strip()
        exploded = exploded[exploded["tag_name"] != ""]
        return self.assign_tag_ids(exploded)[["tag_id", "tag_name", "key"]]
