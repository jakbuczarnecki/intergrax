# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.registry.catalog_miss_diag import CatalogResolutionTier
from intergrax.llm_adapters.registry.context_window import resolve_context_window_tokens
from intergrax.llm_adapters.tracking.metrics import (
    get_llm_metrics_collector,
    record_catalog_miss,
    set_metrics_enabled,
)
from intergrax.runtime.nexus.tracing.adapters.model_catalog_miss import emit_model_catalog_miss_diag
from intergrax.llm_adapters.registry.catalog_miss_diag import (
    ModelCatalogMissDiagV1,
    reset_catalog_miss_diagnostics,
)


@pytest.mark.unit
@pytest.mark.gate
def test_record_catalog_miss_exports_prometheus_counter() -> None:
    get_llm_metrics_collector().reset()
    set_metrics_enabled(True)
    record_catalog_miss(
        provider="openrouter",
        model="vendor/unknown",
        resolution_tier=CatalogResolutionTier.PROVIDER_DEFAULT.value,
        tenant_id="tenant-a",
    )
    lines = get_llm_metrics_collector().prometheus_lines()
    assert any(
        'intergrax_llm_catalog_miss_total{tenant_id="tenant-a",provider="openrouter",'
        'model="vendor/unknown",resolution_tier="provider_default"} 1' in line
        for line in lines
    )


@pytest.mark.unit
@pytest.mark.gate
def test_emit_model_catalog_miss_diag_records_metric_when_enabled() -> None:
    get_llm_metrics_collector().reset()
    set_metrics_enabled(True)
    diag = ModelCatalogMissDiagV1(
        provider_slug="openrouter",
        model_id="vendor/x",
        resolved_tokens=128_000,
        resolution_tier=CatalogResolutionTier.PROVIDER_DEFAULT.value,
        run_id="run-m",
    )
    emit_model_catalog_miss_diag(lambda **_kwargs: None, diag)
    lines = get_llm_metrics_collector().prometheus_lines()
    assert any("intergrax_llm_catalog_miss_total" in line for line in lines)


@pytest.mark.unit
@pytest.mark.gate
def test_prefix_rule_resolution_emits_miss_once() -> None:
    reset_catalog_miss_diagnostics()
    received: list[ModelCatalogMissDiagV1] = []
    from intergrax.llm_adapters.registry.catalog_miss_diag import register_catalog_miss_trace_observer

    register_catalog_miss_trace_observer(received.append)
    tokens = resolve_context_window_tokens(
        "claude",
        "claude-unknown-future-model",
        profile_options={"run_id": "run-prefix"},
    )
    assert tokens == 200_000
    assert len(received) == 1
    assert received[0].resolution_tier == CatalogResolutionTier.PREFIX_RULE.value
