from __future__ import annotations

from src.config import settings
from src.llm.base import LLMClient


def build_llm_client() -> LLMClient:
    backend = settings.llm.backend
    if backend == "openai":
        from src.llm.openai_client import OpenAIClient

        return OpenAIClient()
    if backend == "vllm":
        from src.llm.vllm_client import VLLMClient

        return VLLMClient()
    raise ValueError(f"unknown LLM backend: {backend}")
