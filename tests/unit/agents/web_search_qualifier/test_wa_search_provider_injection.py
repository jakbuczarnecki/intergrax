# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-G (PT-WA): the web-search qualifier requires an injected provider (NT-WA-10..12).

DP-15: ambient credentials never activate a search provider inside ``agents/``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.integration_profile import IntegrationProfile

from web_search_qualifier.steps import web_search_job
from web_search_qualifier.web_search_qualifier_agent import WebSearchQualifierAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CREDENTIAL_VARIABLES = (
    "INTERGRAX_TAVILY_API_KEY",
    "TAVILY_API_KEY",
    "INTERGRAX_BRAVE_API_KEY",
    "BRAVE_API_KEY",
    "INTERGRAX_GOOGLE_CSE_API_KEY",
    "GOOGLE_CSE_API_KEY",
    "INTERGRAX_GOOGLE_CSE_CX",
    "GOOGLE_CSE_CX",
    "INTERGRAX_EXA_API_KEY",
    "EXA_API_KEY",
)


def _set_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CREDENTIAL_VARIABLES:
        monkeypatch.setenv(name, "ambient-test-value")


def test_wa_10_agent_without_provider_refused() -> None:
    with pytest.raises(TypeError, match="search_provider"):
        WebSearchQualifierAgent(llm_adapter=SimpleNamespace())  # type: ignore[call-arg]


def test_wa_11_ambient_credentials_do_not_activate_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_ambient_credentials(monkeypatch)

    def _forbidden_factory(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("an agent must not create a vendor provider from the environment")

    for module_path, name in (
        ("intergrax.integrations._shared.p2.factories", "create_brave_search_provider"),
        ("intergrax.integrations._shared.p3.factories", "create_tavily_search_provider"),
        ("intergrax.integrations._shared.p3.factories", "create_exa_search_provider"),
        ("intergrax.integrations._shared.p3.factories", "create_google_cse_search_provider"),
    ):
        monkeypatch.setattr(f"{module_path}.{name}", _forbidden_factory)

    monkeypatch.setattr(
        web_search_job,
        "exec_ctx_from_step",
        lambda _step: SimpleNamespace(metadata={}),
    )
    monkeypatch.setattr(web_search_job, "request_metadata", lambda *_a, **_k: {})
    monkeypatch.setattr(web_search_job, "_resolve_llm_adapter", lambda _ctx: SimpleNamespace())
    step_ctx = SimpleNamespace(run_id="wa-11", metadata={})

    output = asyncio.run(
        web_search_job.run_web_search_job(step_ctx, resolved_provider=None),  # type: ignore[arg-type]
    )
    summary = output["web_search_summary"]
    assert isinstance(summary, dict)
    assert summary["used"] is False
    assert summary["reason"] == "search_provider_not_configured"


def test_wa_12_lkw_builder_refuses_profile_without_search_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_workspace_application.host import agent_factories
    from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

    _set_ambient_credentials(monkeypatch)
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    environment = LOCAL_WORKSPACE_APPLICATION_MANIFEST.environment.model_copy(deep=True)
    environment.integration_profile = IntegrationProfile()
    environment.execution_mode = ExecutionMode.STRICT
    assert environment.integration_profile.search_provider is None
    ctx = ApplicationBuildContext.for_manifest(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        environment=environment,
    )
    binding = AgentBinding.mount(
        WebSearchQualifierAgent,
        contract_id="web_search_qualifier",
        factory=agent_factories.build_local_workspace_web_search_qualifier_from_context,
    )
    with pytest.raises(IntegrationConfigurationError, match="search_provider"):
        agent_factories.build_local_workspace_web_search_qualifier_from_context(ctx, binding)
