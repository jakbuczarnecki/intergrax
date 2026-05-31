# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.tool_wiring import wire_legal_tools
from legal_application.host.wiring import build_legal_agent, build_legal_registry

pytestmark = pytest.mark.unit


def test_wire_legal_tools_respects_env_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGAL_ENABLE_RAG", "true")
    monkeypatch.setenv("LEGAL_ENABLE_WEBSEARCH", "true")
    settings = LegalBackendSettings.from_env()
    wiring = wire_legal_tools(settings=settings)
    assert wiring.registry.has("rag.retrieve")
    assert wiring.registry.has("websearch.query")


def test_build_legal_agent_passes_tool_profile_to_config() -> None:
    settings = LegalBackendSettings(
        environment="dev",  # type: ignore[arg-type]
        legal_product_profile="strict_legal",
        legal_llm_provider="ollama",
        legal_default_agent_id="legal-default",
        legal_route_prefix="/v1/legal",
        identity_source="body_or_context",
        cors_allow_origins=frozenset(),
        allowed_hosts=frozenset(),
        openapi_enabled_override=None,
        session_sqlite_path=None,
        enable_rag=True,
    )
    tool_wiring = wire_legal_tools(settings=settings)
    ctx = ApplicationBuildContext.for_manifest(
        object(),
        settings=settings,
        tool_profile=tool_wiring.profile,
        tool_wiring_context=tool_wiring.wiring_context,
    )
    agent = build_legal_agent(settings, ctx=ctx)
    assert agent._config.tool_profile is not None
    assert agent._config.tool_profile.is_tool_enabled("rag.retrieve")
    assert agent._config.enable_rag is True


def test_build_legal_registry_materializes_agent() -> None:
    settings = LegalBackendSettings(
        environment="dev",  # type: ignore[arg-type]
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
    registry = build_legal_registry(settings)
    assert registry.has("legal-default") or registry.has("legal")
