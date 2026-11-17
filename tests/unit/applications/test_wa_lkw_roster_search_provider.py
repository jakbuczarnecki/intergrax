# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-G (PT-WA, Amendment 1): the default LKW roster builds without a search provider slot.

NT-WA-23 (acceptance), NT-WA-24 (companion), NT-WA-25 (enablement equals builder acceptance).
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.integration_profile import IntegrationProfile

from local_workspace_application.host import agent_factories
from local_workspace_application.host.agent_factories import local_workspace_profile_has_search_provider
from local_workspace_application.host.environment_profile import build_local_workspace_environment_profile
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.wiring import build_local_workspace_manifest
from local_workspace_application.manifest import (
    LOCAL_WORKSPACE_APPLICATION_MANIFEST,
    build_local_workspace_agent_bindings,
)
from local_workspace_application.tests.lkw_ac3_projection import (
    build_lkw_test_registry_projection,
    lkw_test_environment,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_QUALIFIER_CAPABILITY = "local.workspace.web_search_qualification"
_APP = Path(__file__).resolve().parents[3] / "applications" / "local_workspace_application"


def _binding(bindings: list[AgentBinding]) -> AgentBinding:
    return next(b for b in bindings if b.contract_id == "web_search_qualifier")


def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )


def test_wa_23_default_lkw_roster_builds_without_web_search_qualifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_llm(monkeypatch)
    assert LOCAL_WORKSPACE_APPLICATION_MANIFEST.environment.integration_profile.search_provider is None
    projection = build_lkw_test_registry_projection()
    registry = projection.agent_registry
    assert "web_search_qualifier" not in registry.list_agent_ids()
    assert registry.find_by_capability(_QUALIFIER_CAPABILITY) == []
    assert "local_search" in registry.list_agent_ids()


def test_wa_24_product_derivation_enables_binding_when_slot_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_profile = IntegrationProfile()
    assert not _binding(build_local_workspace_agent_bindings(default_profile)).enabled

    slot_profile = IntegrationProfile(search_provider="tavily")
    binding = _binding(build_local_workspace_agent_bindings(slot_profile))
    assert binding.enabled
    assert local_workspace_profile_has_search_provider(slot_profile)

    _stub_llm(monkeypatch)
    sentinel = SimpleNamespace(search=lambda *_a, **_k: ())
    monkeypatch.setattr(
        "intergrax.integrations.registry.factory.resolve_from_profile",
        lambda *_a, **_k: sentinel,
    )
    environment = build_local_workspace_environment_profile().model_copy(deep=True)
    environment.integration_profile = slot_profile
    ctx = ApplicationBuildContext.for_manifest(LOCAL_WORKSPACE_APPLICATION_MANIFEST, environment=environment)
    agent = agent_factories.build_local_workspace_web_search_qualifier_from_context(ctx, binding)
    assert agent._search_provider.provider is sentinel  # noqa: SLF001
    with pytest.raises(IntegrationConfigurationError):
        agent_factories.build_local_workspace_web_search_qualifier_from_context(
            ApplicationBuildContext.for_manifest(
                LOCAL_WORKSPACE_APPLICATION_MANIFEST,
                environment=build_local_workspace_environment_profile(),
            ),
            binding,
        )


def test_wa_25_manifest_enablement_equals_builder_acceptance() -> None:
    settings = LocalWorkspaceBackendSettings.from_env()
    cases = {
        "module constant": (
            LOCAL_WORKSPACE_APPLICATION_MANIFEST,
            LOCAL_WORKSPACE_APPLICATION_MANIFEST.environment,
        ),
        "build_local_workspace_manifest": (
            build_local_workspace_manifest(settings),
            LOCAL_WORKSPACE_APPLICATION_MANIFEST.environment,
        ),
        "lkw_ac3_projection environment": (
            LOCAL_WORKSPACE_APPLICATION_MANIFEST,
            lkw_test_environment(settings),
        ),
    }
    for label, (manifest, environment) in cases.items():
        expected = local_workspace_profile_has_search_provider(environment.integration_profile)
        assert _binding(list(manifest.agents)).enabled == expected, label

    for name in ("manifest.py", "host/agent_factories.py"):
        tree = ast.parse((_APP / name).read_text(encoding="utf-8"))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "local_workspace_profile_has_search_provider"
        ]
        assert calls, f"{name} must call the single derivation function"
