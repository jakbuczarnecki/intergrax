# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.registry.catalog_miss_diag import (
    CatalogResolutionTier,
    ModelCatalogMissDiagV1,
    maybe_emit_catalog_miss,
    register_catalog_miss_trace_observer,
    reset_catalog_miss_diagnostics,
)
from intergrax.runtime.nexus.tracing.adapters.model_catalog_miss import (
    ModelCatalogMissTraceDiagV1,
    emit_model_catalog_miss_diag,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


@pytest.mark.unit
@pytest.mark.gate
def test_emit_model_catalog_miss_diag_records_plane_a_step() -> None:
    emitted: list[dict[str, object]] = []

    def _trace_event(**kwargs: object) -> None:
        emitted.append(kwargs)

    diag = ModelCatalogMissDiagV1(
        provider_slug="openrouter",
        model_id="vendor/unknown",
        resolved_tokens=8192,
        resolution_tier=CatalogResolutionTier.PROVIDER_DEFAULT.value,
        run_id="run-a",
    )
    emit_model_catalog_miss_diag(_trace_event, diag)
    assert len(emitted) == 1
    event = emitted[0]
    assert event["component"] == TraceComponent.ENGINE
    assert event["step"] == "llm_catalog_miss"
    assert event["level"] == TraceLevel.WARNING
    payload = event["payload"]
    assert isinstance(payload, ModelCatalogMissTraceDiagV1)
    assert payload.schema_id() == "intergrax.diag.engine.core_llm.catalog_miss"
    assert payload.resolution_tier == CatalogResolutionTier.PROVIDER_DEFAULT.value


@pytest.mark.unit
@pytest.mark.gate
def test_register_catalog_miss_trace_observer_flushes_pending() -> None:
    reset_catalog_miss_diagnostics()
    received: list[ModelCatalogMissDiagV1] = []

    maybe_emit_catalog_miss(
        "groq",
        "missing-model",
        4096,
        resolution_tier=CatalogResolutionTier.FALLBACK_DEFAULT,
        run_id="run-a",
    )
    assert len(received) == 0

    register_catalog_miss_trace_observer(received.append)
    assert len(received) == 1
    assert received[0].model_id == "missing-model"

    second = maybe_emit_catalog_miss(
        "groq",
        "missing-model",
        4096,
        resolution_tier=CatalogResolutionTier.FALLBACK_DEFAULT,
        run_id="run-a",
    )
    assert second is None
    assert len(received) == 1
