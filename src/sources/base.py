from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.utils.hashing import tag_id as make_tag_id


class SourceAdapter(ABC):
    name: str
    key_column: str

    @abstractmethod
    def raw_files(self) -> list[Path]:
        pass

    @abstractmethod
    def load_records(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def extract_tags(self, records: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def assign_tag_ids(df: pd.DataFrame, tag_col: str = "tag_name") -> pd.DataFrame:
        out = df.copy()
        out["tag_id"] = out[tag_col].map(make_tag_id)
        return out
