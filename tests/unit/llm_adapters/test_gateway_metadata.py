# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.acp_state import AcpInvocationUsageView, AcpTokenUsage
from intergrax.llm_adapters.registry.catalog_miss_diag import (
    ModelCatalogMissDiagV1,
    maybe_emit_catalog_miss,
    reset_catalog_miss_diagnostics,
)
from intergrax.llm_adapters.registry.context_window import resolve_context_window_tokens
from intergrax.llm_adapters.registry.gateway_metadata.openrouter_client import (
    OpenRouterModelMetadataClient,
)
from intergrax.llm_adapters.registry.gateway_metadata.session import (
    lookup_gateway_context_window,
    reset_gateway_metadata_session,
)
from intergrax.llm_adapters.registry.model_catalog import ModelCatalog


@pytest.mark.unit
@pytest.mark.gate
def test_openrouter_client_uses_injected_fetcher_and_ttl_cache() -> None:
    calls = {"count": 0}

    def _fetch() -> list[dict[str, object]]:
        calls["count"] += 1
        return [
            {"id": "vendor/model-a", "context_length": 120_000},
            {"id": "vendor/model-b", "context_length": 8_192},
        ]

    client = OpenRouterModelMetadataClient(ttl_seconds=3600.0, fetcher=_fetch)
    first = client.lookup("vendor/model-a")
    second = client.lookup("vendor/model-a")
    assert first is not None
    assert first.context_window_tokens == 120_000
    assert second == first
    assert calls["count"] == 1


@pytest.mark.unit
@pytest.mark.gate
def test_gateway_metadata_merge_opt_in_between_catalog_and_legacy() -> None:
    reset_gateway_metadata_session()
    cat = ModelCatalog(models=(), prefix_rules=(), provider_defaults={}, fallback_default=4096)

    def _fetch() -> list[dict[str, object]]:
        return [{"id": "unknown/live-model", "context_length": 256_000}]

    tokens = resolve_context_window_tokens(
        "openrouter",
        "unknown/live-model",
        profile_options={
            "fetch_gateway_metadata": True,
            "gateway_metadata_fetcher": _fetch,
        },
        catalog=cat,
    )
    assert tokens == 256_000


@pytest.mark.unit
@pytest.mark.gate
def test_gateway_lookup_respects_fetch_flag() -> None:
    reset_gateway_metadata_session()
    assert (
        lookup_gateway_context_window(
            "openrouter",
            "any-model",
            {"fetch_gateway_metadata": False, "gateway_metadata_fetcher": lambda: []},
        )
        is None
    )


@pytest.mark.unit
@pytest.mark.gate
def test_catalog_miss_diag_emits_once_per_model_run() -> None:
    reset_catalog_miss_diagnostics()
    first = maybe_emit_catalog_miss("groq", "missing-model", 4096, run_id="run-a")
    second = maybe_emit_catalog_miss("groq", "missing-model", 4096, run_id="run-a")
    third = maybe_emit_catalog_miss("groq", "missing-model", 4096, run_id="run-b")
    assert isinstance(first, ModelCatalogMissDiagV1)
    assert first.schema_id == "intergrax.diag.engine.core_llm.catalog_miss"
    assert second is None
    assert third is not None


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_context_window_emits_miss_on_fallback_default() -> None:
    reset_catalog_miss_diagnostics()
    cat = ModelCatalog(models=(), prefix_rules=(), provider_defaults={}, fallback_default=8192)
    tokens = resolve_context_window_tokens(
        "custom-gateway",
        "totally-unknown-model",
        profile_options={"run_id": "run-miss"},
        catalog=cat,
    )
    assert tokens == 8192
    repeat = maybe_emit_catalog_miss(
        "custom-gateway",
        "totally-unknown-model",
        8192,
        run_id="run-miss",
    )
    assert repeat is None
