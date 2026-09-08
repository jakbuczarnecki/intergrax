# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from legal.legal_agent import LegalAgent
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.tool_wiring import wire_legal_tools
from legal_application.host.wiring import build_legal_agent
from legal_application.tests.legal_ac3_projection import build_legal_test_registry_projection

pytestmark = pytest.mark.unit


def test_wire_legal_tools_respects_env_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGAL_ENABLE_RAG", "true")
    monkeypatch.setenv("LEGAL_ENABLE_WEBSEARCH", "true")
    settings = LegalBackendSettings.from_env()
    wiring = wire_legal_tools(settings=settings)
    assert wiring.registry.has("rag.retrieve")
    assert wiring.registry.has("websearch.query")


def test_build_legal_agent_returns_canonical_reflex_agent() -> None:
    settings = LegalBackendSettings(
        environment="dev",  # type: ignore[arg-type]
        legal_product_profile="strict_legal",
        legal_llm_provider="ollama",
        legal_default_agent_id="legal-default",
        route_prefix="/v1/legal",
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
    assert isinstance(agent, LegalAgent)
    assert agent.get_contract().id == "legal"


def test_build_legal_registry_materializes_agent() -> None:
    settings = LegalBackendSettings(
        environment="dev",  # type: ignore[arg-type]
        legal_product_profile="strict_legal",
        legal_llm_provider="ollama",
        legal_default_agent_id="legal",
        route_prefix="/v1/legal",
        identity_source="body_or_context",
        cors_allow_origins=frozenset(),
        allowed_hosts=frozenset(),
        openapi_enabled_override=None,
        session_sqlite_path=None,
        enable_rag=True,
        enable_websearch=True,
    )
    projection = build_legal_test_registry_projection(settings)
    assert projection.agent_registry.has("legal")
