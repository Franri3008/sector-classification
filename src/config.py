from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Paths(BaseModel):
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    outputs_dir: Path = ROOT / "data" / "outputs"
    sectors_csv: Path = ROOT / "sectors.csv"


class DropboxPaths(BaseModel):
    crunchbase: str = "/datalab (3)/crunchbase"
    regpat: str = "/datalab (3)/regpat"
    openalex: str = "/datalab (3)/openalex"


class EmbeddingCfg(BaseModel):
    provider: Literal["customtools"] = "customtools"
    model: str = "text-embedding-3-large"
    batch_size: int = 1024
    rpm: int = 3000
    dim: int = 3072


class LLMCfg(BaseModel):
    backend: Literal["openai", "vllm"] = "openai"
    openai_model: str = "gpt-4o-mini"
    vllm_model: str = "gemma4"
    vllm_base_url: str = "http://localhost:8001/v1"
    rpm: int = 500
    max_concurrency: int = 8
    temperature: float = 0.0
    seed: int = 42
    schema_repair_retries: int = 1


class RankingCfg(BaseModel):
    top_n: int = 10
    min_similarity: float = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore"
    )
    dropbox_token: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    paths: Paths = Field(default_factory=Paths)
    dropbox_paths: DropboxPaths = Field(default_factory=DropboxPaths)
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    ranking: RankingCfg = Field(default_factory=RankingCfg)
    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
