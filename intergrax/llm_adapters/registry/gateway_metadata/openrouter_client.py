# © Artur Czarnecki. All rights reserved.

"""OpenRouter ``/models`` metadata client with TTL cache (M-LLM-X.14.2)."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

GatewayModelsFetcher = Callable[[], Sequence[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class GatewayModelMetadata:
    model_id: str
    context_window_tokens: int | None = None


def _parse_models_payload(payload: Mapping[str, Any]) -> list[GatewayModelMetadata]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    records: list[GatewayModelMetadata] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("id", "")).strip()
        if not model_id:
            continue
        top_provider = item.get("top_provider")
        context_raw: Any = None
        if isinstance(top_provider, Mapping):
            context_raw = top_provider.get("context_length")
        if context_raw is None:
            context_raw = item.get("context_length")
        tokens: int | None = None
        if context_raw is not None:
            try:
                parsed = int(context_raw)
            except (TypeError, ValueError):
                parsed = 0
            if parsed > 0:
                tokens = parsed
        records.append(GatewayModelMetadata(model_id=model_id, context_window_tokens=tokens))
    return records


def _default_http_fetch(url: str = OPENROUTER_MODELS_URL) -> list[GatewayModelMetadata]:
    import httpx

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        return []
    return _parse_models_payload(payload)


class OpenRouterModelMetadataClient:
    """Fetch and cache OpenRouter model metadata for context-window merge."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 3600.0,
        fetcher: GatewayModelsFetcher | None = None,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._fetcher = fetcher
        self._cache: dict[str, GatewayModelMetadata] = {}
        self._loaded_at: float | None = None

    def _ensure_loaded(self) -> None:
        now = time.monotonic()
        if self._loaded_at is not None and (now - self._loaded_at) < self._ttl_seconds:
            return
        if self._fetcher is not None:
            raw_models = self._fetcher()
            records = [
                GatewayModelMetadata(
                    model_id=str(item.get("id", "")).strip(),
                    context_window_tokens=(
                        int(item["context_length"])
                        if item.get("context_length") is not None
                        and int(item["context_length"]) > 0
                        else None
                    ),
                )
                for item in raw_models
                if str(item.get("id", "")).strip()
            ]
        else:
            records = _default_http_fetch()
        self._cache = {record.model_id: record for record in records if record.model_id}
        self._loaded_at = now

    def lookup(self, model_id: str) -> GatewayModelMetadata | None:
        normalized = (model_id or "").strip()
        if not normalized:
            return None
        self._ensure_loaded()
        return self._cache.get(normalized)

    def reset_cache(self) -> None:
        self._cache.clear()
        self._loaded_at = None
