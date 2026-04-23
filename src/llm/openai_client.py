from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.config import settings
from src.llm.base import LLMClient
from src.llm.prompts import (
    CLASSIFY_SYSTEM,
    DESCRIBE_SYSTEM,
    ENRICH_SECTORS_SYSTEM,
    classify_user,
    describe_user,
    enrich_sectors_user,
)
from src.llm.schemas import ClassificationResult, DescriptionResult, SectorEnrichmentResult
from src.logging_setup import get_logger
from src.pipeline.summary import record_usage
from src.utils.retry import retryable

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class OpenAIClient(LLMClient):
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model_id = model or settings.llm.openai_model
        self._api_key = api_key or settings.openai_api_key.get_secret_value()
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY is empty — set it in .env.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    @retryable(exceptions=(ConnectionError, TimeoutError, RuntimeError))
    def _parse(self, system: str, user: str, schema: type[T]) -> T:
        resp = self._get_client().chat.completions.parse(
            model=self.model_id,
            temperature=settings.llm.temperature,
            seed=settings.llm.seed,
            response_format=schema,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            record_usage(
                provider="openai",
                model=self.model_id,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(
                f"OpenAI returned null parsed response; refusal={resp.choices[0].message.refusal!r}"
            )
        return parsed

    def describe(self, tag_name: str, source: str) -> DescriptionResult:
        try:
            return self._parse(DESCRIBE_SYSTEM, describe_user(tag_name, source), DescriptionResult)
        except ValidationError as e:
            logger.warning("describe schema failed for %r (skipping): %s", tag_name, e)
            return DescriptionResult(tag_name=tag_name, description="")

    def classify(
        self, tag_name: str, tag_description: str, candidates: list[dict]
    ) -> ClassificationResult:
        user = classify_user(tag_name, tag_description, candidates)
        try:
            return self._parse(CLASSIFY_SYSTEM, user, ClassificationResult)
        except ValidationError as e:
            logger.warning("classify schema failed for %r (returning empty picks): %s", tag_name, e)
            return ClassificationResult(tag_name=tag_name, picks=[])

    def enrich_sectors(
        self, section_code: str, section_name: str, divisions: list[dict]
    ) -> SectorEnrichmentResult:
        user = enrich_sectors_user(section_code, section_name, divisions)
        try:
            return self._parse(ENRICH_SECTORS_SYSTEM, user, SectorEnrichmentResult)
        except ValidationError as e:
            logger.warning(
                "enrich_sectors schema failed for section %s (returning empty): %s",
                section_code,
                e,
            )
            return SectorEnrichmentResult(section_code=section_code, divisions=[])
