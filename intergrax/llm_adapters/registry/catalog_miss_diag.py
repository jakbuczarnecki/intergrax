# © Artur Czarnecki. All rights reserved.

"""Model catalog miss diagnostic (M-LLM-X.14.2 · M-LLM-X.15.1 · M-LLM-X.16.4 · ADR-LLM-002)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

CatalogMissObserver = Callable[["ModelCatalogMissDiagV1"], None]


class CatalogResolutionTier(str, Enum):
    """Non-exact catalog resolution tier that triggers observability."""

    PREFIX_RULE = "prefix_rule"
    PROVIDER_DEFAULT = "provider_default"
    FALLBACK_DEFAULT = "fallback_default"


@dataclass(frozen=True, slots=True)
class ModelCatalogMissDiagV1:
    """Emitted once per model/run when context window resolves without exact catalog hit."""

    provider_slug: str
    model_id: str
    resolved_tokens: int
    resolution_tier: str
    run_id: str | None = None

    schema_id: ClassVar[str] = "intergrax.diag.engine.core_llm.catalog_miss"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_slug": self.provider_slug,
            "model_id": self.model_id,
            "resolved_tokens": self.resolved_tokens,
            "resolution_tier": self.resolution_tier,
            "run_id": self.run_id,
        }


_run_emitted: dict[str, set[tuple[str, str]]] = defaultdict(set)
_run_pending: dict[str, list[ModelCatalogMissDiagV1]] = defaultdict(list)
_run_observers: dict[str, CatalogMissObserver] = {}
_trace_observer: CatalogMissObserver | None = None


def _run_key(run_id: str | None) -> str:
    return str(run_id or "").strip()


def _provider_slug(provider: LLMProvider | str) -> str:
    if isinstance(provider, LLMProvider):
        return provider.value
    return str(provider or "").strip().lower()


def reset_catalog_miss_diagnostics() -> None:
    """Clear all run-scoped dedupe state (tests)."""
    _run_emitted.clear()
    _run_pending.clear()
    _run_observers.clear()
    global _trace_observer
    _trace_observer = None


def begin_catalog_miss_run(run_id: str) -> None:
    """Reset dedupe for one Nexus run (M-LLM-X.16.4). Pending is flushed by bind, not cleared here."""
    rid = _run_key(run_id)
    _run_emitted[rid].clear()


def bind_catalog_miss_run_observer(
    run_id: str,
    observer: CatalogMissObserver | None,
) -> None:
    """Attach a run-scoped Plane A sink and flush that run's pending misses."""
    rid = _run_key(run_id)
    if observer is None:
        _run_observers.pop(rid, None)
        return
    _run_observers[rid] = observer
    pending = list(_run_pending[rid])
    _run_pending[rid].clear()
    for diag in pending:
        observer(diag)


def register_catalog_miss_trace_observer(observer: CatalogMissObserver | None) -> None:
    """Attach global Plane A trace sink; flush any pending misses without run binding."""
    global _trace_observer
    _trace_observer = observer
    if observer is None:
        return
    for rid in list(_run_pending):
        if rid in _run_observers:
            continue
        pending = list(_run_pending[rid])
        _run_pending[rid].clear()
        for diag in pending:
            observer(diag)


def _resolve_observer(run_id: str | None) -> CatalogMissObserver | None:
    rid = _run_key(run_id)
    return _run_observers.get(rid) or _trace_observer


def maybe_emit_catalog_miss(
    provider: LLMProvider | str,
    model: str,
    resolved_tokens: int,
    *,
    resolution_tier: CatalogResolutionTier | str,
    run_id: str | None = None,
) -> ModelCatalogMissDiagV1 | None:
    """Return diagnostic payload on first miss per model/run; otherwise None."""
    slug = _provider_slug(provider)
    model_id = (model or "").strip()
    tier = (
        resolution_tier.value
        if isinstance(resolution_tier, CatalogResolutionTier)
        else str(resolution_tier or "").strip()
    )
    rid = _run_key(run_id)
    dedupe_key = (slug, model_id)
    if dedupe_key in _run_emitted[rid]:
        return None
    _run_emitted[rid].add(dedupe_key)
    diag = ModelCatalogMissDiagV1(
        provider_slug=slug,
        model_id=model_id,
        resolved_tokens=int(resolved_tokens),
        resolution_tier=tier,
        run_id=run_id,
    )
    observer = _resolve_observer(run_id)
    if observer is not None:
        observer(diag)
    else:
        _run_pending[rid].append(diag)
    return diag
