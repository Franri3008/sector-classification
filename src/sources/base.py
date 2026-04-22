from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.config import settings
from src.utils.hashing import tag_id as make_tag_id


class SourceAdapter(ABC):
    name: str
    raw_filename: str = "index.csv"

    def raw_files(self) -> list[Path]:
        p = settings.paths.raw_dir / self.name / self.raw_filename
        return [p] if p.exists() else []

    def load_records(self) -> pd.DataFrame:
        files = self.raw_files()
        if not files:
            raise FileNotFoundError(
                f"No {self.raw_filename} in "
                f"{settings.paths.raw_dir / self.name}. Sync from Dropbox first."
            )
        p = files[0]
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    @abstractmethod
    def extract_tags(self, records: pd.DataFrame) -> pd.DataFrame: ...

    @staticmethod
    def assign_tag_ids(df: pd.DataFrame, tag_col: str = "tag_name") -> pd.DataFrame:
        out = df.copy()
        out["tag_id"] = out[tag_col].map(make_tag_id)
        return out
