from __future__ import annotations

import pandas as pd

from src.sources.base import SourceAdapter


class CrunchbaseAdapter(SourceAdapter):
    name = "crunchbase"

    def extract_tags(self, records: pd.DataFrame) -> pd.DataFrame:
        sub = records[["domain"]].dropna()
        sub["domain"] = sub["domain"].astype(str).str.strip()
        sub = sub[sub["domain"] != ""].drop_duplicates("domain")
        sub = sub.rename(columns={"domain": "tag_name"})
        sub["key"] = sub["tag_name"]
        return self.assign_tag_ids(sub)[["tag_id", "tag_name", "key"]]
