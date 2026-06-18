# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.fastapi_core.config import ApiEnvironment
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.wiring import build_legal_manifest, build_legal_registry
from legal_application.manifest import LEGAL_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _dev_settings(**overrides: object) -> LegalBackendSettings:
    base = dict(
        environment=ApiEnvironment.DEV,
        legal_product_profile="strict_legal",
        legal_llm_provider="ollama",
        legal_default_agent_id="legal-default",
        legal_route_prefix="/v1/legal",
        identity_source="body_or_context",
        cors_allow_origins=frozenset(),
        allowed_hosts=frozenset(),
        openapi_enabled_override=None,
        session_sqlite_path=None,
    )
    base.update(overrides)
    return LegalBackendSettings(**base)  # type: ignore[arg-type]


def test_legal_manifest_is_product_profile() -> None:
    assert LEGAL_APPLICATION_MANIFEST.profile.value == "product"
    assert len(LEGAL_APPLICATION_MANIFEST.enabled_agents()) == 1
    assert LEGAL_APPLICATION_MANIFEST.enabled_agents()[0].default is True


def test_build_legal_manifest_overrides_contract_id() -> None:
    settings = _dev_settings(legal_default_agent_id="custom-legal-id")
    manifest = build_legal_manifest(settings)
    assert manifest.agents[0].contract_id == "custom-legal-id"


def test_legal_environment_observability_assembly_valid() -> None:
    from intergrax.applications._shared.observability_assembly_resolver import (
        assert_observability_assembly_valid,
    )
    from intergrax.applications._shared.observability_wiring import wire_application_observability
    from legal_application.host.wiring import build_legal_environment_profile

    env = build_legal_environment_profile(_dev_settings())
    assert env.integration_profile.observability_backend is not None
    wiring = wire_application_observability(env)
    assert_observability_assembly_valid(wiring, env)


@pytest.mark.no_ci
def test_build_legal_registry_materializes_default_agent() -> None:
    registry = build_legal_registry(_dev_settings())
    assert registry.has("legal-default") or registry.has("legal")


@pytest.mark.no_ci
def test_build_legal_registry_passes_tool_context() -> None:
    settings = _dev_settings(enable_rag=True, enable_websearch=False)
    manifest = build_legal_manifest(settings)
    from legal_application.host.tool_wiring import wire_legal_tools

    tool_wiring = wire_legal_tools(settings=settings)
    ctx = ApplicationBuildContext.for_manifest(
        manifest,
        settings=settings,
        tool_profile=tool_wiring.profile,
        tool_wiring_context=tool_wiring.wiring_context,
    )
    registry = build_application_registry(manifest, ctx)
    assert registry.list_agent_ids()
