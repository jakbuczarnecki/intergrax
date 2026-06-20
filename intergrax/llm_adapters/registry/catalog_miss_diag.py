# © Artur Czarnecki. All rights reserved.

"""Model catalog miss diagnostic (M-LLM-X.14.2 · ADR-LLM-002 step 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider


@dataclass(frozen=True, slots=True)
class ModelCatalogMissDiagV1:
    """Emitted once per model/run when catalog falls back to conservative default."""

    provider_slug: str
    model_id: str
    resolved_tokens: int
    run_id: str | None = None

    schema_id: ClassVar[str] = "intergrax.diag.engine.core_llm.catalog_miss"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_slug": self.provider_slug,
            "model_id": self.model_id,
            "resolved_tokens": self.resolved_tokens,
            "run_id": self.run_id,
        }


_emitted_keys: set[tuple[str, str, str]] = set()


def _provider_slug(provider: LLMProvider | str) -> str:
    if isinstance(provider, LLMProvider):
        return provider.value
    return str(provider or "").strip().lower()


def reset_catalog_miss_diagnostics() -> None:
    """Clear dedupe state (tests)."""
    _emitted_keys.clear()


def maybe_emit_catalog_miss(
    provider: LLMProvider | str,
    model: str,
    resolved_tokens: int,
    *,
    run_id: str | None = None,
) -> ModelCatalogMissDiagV1 | None:
    """Return diagnostic payload on first miss per model/run; otherwise None."""
    slug = _provider_slug(provider)
    model_id = (model or "").strip()
    key = (run_id or "", slug, model_id)
    if key in _emitted_keys:
        return None
    _emitted_keys.add(key)
    return ModelCatalogMissDiagV1(
        provider_slug=slug,
        model_id=model_id,
        resolved_tokens=int(resolved_tokens),
        run_id=run_id,
    )
