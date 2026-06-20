# © Artur Czarnecki. All rights reserved.

"""Model catalog miss diagnostics on Plane A trace (M-LLM-X.14.2 · ADR-LLM-002)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict

from intergrax.llm_adapters.registry.catalog_miss_diag import ModelCatalogMissDiagV1
from intergrax.runtime.nexus.tracing.trace_models import (
    DiagnosticPayload,
    TraceComponent,
    TraceLevel,
)

TraceEmitFn = Callable[..., None]


@dataclass(frozen=True)
class ModelCatalogMissTraceDiagV1(DiagnosticPayload):
    """PII-safe record when context window falls back to catalog default."""

    provider_slug: str
    model_id: str
    resolved_tokens: int
    run_id: str | None = None

    def redact(self) -> ModelCatalogMissTraceDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return ModelCatalogMissDiagV1.schema_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_slug": self.provider_slug,
            "model_id": self.model_id,
            "resolved_tokens": self.resolved_tokens,
            "run_id": self.run_id,
        }


def trace_diag_from_catalog_miss(diag: ModelCatalogMissDiagV1) -> ModelCatalogMissTraceDiagV1:
    return ModelCatalogMissTraceDiagV1(
        provider_slug=diag.provider_slug,
        model_id=diag.model_id,
        resolved_tokens=diag.resolved_tokens,
        run_id=diag.run_id,
    )


def emit_model_catalog_miss_diag(
    trace_event: TraceEmitFn,
    diag: ModelCatalogMissDiagV1,
) -> None:
    payload = trace_diag_from_catalog_miss(diag)
    trace_event(
        component=TraceComponent.ENGINE,
        step="llm_catalog_miss",
        message="Model catalog miss — conservative context window default applied.",
        level=TraceLevel.WARNING,
        payload=payload,
    )


def wire_catalog_miss_trace_sink(trace_event: TraceEmitFn) -> None:
    """Register Plane A emission for Tier-0 catalog miss diagnostics."""

    def _emit(diag: ModelCatalogMissDiagV1) -> None:
        emit_model_catalog_miss_diag(trace_event, diag)

    from intergrax.llm_adapters.registry.catalog_miss_diag import (
        register_catalog_miss_trace_observer,
    )

    register_catalog_miss_trace_observer(_emit)
