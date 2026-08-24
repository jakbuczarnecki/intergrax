# © Artur Czarnecki. All rights reserved.

"""APP-1 production runtime boundary checks for incident investigation scenario."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.prompt_wiring import resolve_prompt_registry
from intergrax.applications._shared.runtime_config_bridge import (
    build_runtime_context_from_environment,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation import (
    investigator_agent,
    runtime_composition,
    tools,
)
from platform_proofs.scenarios.ai_incident_investigation.runtime_composition import (
    build_agent_runtime_context,
    build_scenario_environment_profile,
    build_scenario_runtime_composition,
    ensure_scenario_llm_adapter_resolved,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario import build_runtime_bundle
from testing_support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit

_SCENARIO_PACKAGE = Path(__file__).resolve().parents[5] / "platform_proofs" / "scenarios" / "ai_incident_investigation"
_CANONICAL_MODULES = (
    investigator_agent,
    runtime_composition,
    tools,
    __import__(
        "platform_proofs.scenarios.ai_incident_investigation.scenario",
        fromlist=["scenario"],
    ),
)

_FORBIDDEN_TOKENS = (
    "FakeLLMAdapter",
    "testing_support",
    "unittest.mock",
    "MagicMock",
    "from unittest.mock",
    "import Mock",
    "pytest",
)


def _runtime_request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="scenario-tenant",
        agent_id="incident_investigator",
        user_id="scenario-user",
        session_id="scenario-session",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        message="investigate",
    )


def test_canonical_modules_forbid_test_runtime_imports() -> None:
    for module in _CANONICAL_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        for token in _FORBIDDEN_TOKENS:
            assert token not in source, f"{module.__name__} must not reference {token}"


def test_canonical_modules_forbid_provider_specific_hardcoding() -> None:
    for module in _CANONICAL_MODULES:
        lowered = Path(module.__file__).read_text(encoding="utf-8").lower()
        for token in ("openai", "anthropic", "azure", "groq"):
            assert token not in lowered, f"{module.__name__} hardcodes provider {token}"


def test_tool_contracts_use_production_tool_contract() -> None:
    source = Path(tools.__file__).read_text(encoding="utf-8")
    assert "ToolContract(" in source
    assert "tools_agent_make_contract" not in source


def test_build_runtime_bundle_registers_production_tool_contracts() -> None:
    bundle = build_runtime_bundle()
    contract = bundle.registry.get("production.workload.read").contract
    assert isinstance(contract, ToolContract)
    assert contract.input_schema.__name__ == "LineWindowInput"


def test_build_agent_runtime_context_uses_production_platform_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    sentinel = FakeLLMAdapter(fixed_text="probe")

    def _track_resolve(*_args, **_kwargs):
        calls.append("resolve_llm_adapter")
        return sentinel

    def _track_build_context(request, build_ctx, env, **kwargs):
        calls.append("build_runtime_context_from_environment")
        assert kwargs.get("llm_adapter") is None
        ctx = build_runtime_context_from_environment(
            request,
            build_ctx,
            env,
            llm_adapter=sentinel,
        )
        return ctx

    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.runtime_composition.resolve_llm_adapter",
        _track_resolve,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.runtime_composition.build_runtime_context_from_environment",
        _track_build_context,
    )

    bundle = build_runtime_bundle()
    ctx = build_agent_runtime_context(_runtime_request(), bundle.runtime_composition)

    assert calls == [
        "resolve_llm_adapter",
        "build_runtime_context_from_environment",
    ]
    assert isinstance(ctx.config.llm_adapter, FakeLLMAdapter)
    assert ctx.session_manager is not None
    assert isinstance(ctx.prompt_registry, YamlPromptRegistry)


def test_prompt_registry_resolved_through_platform_wiring() -> None:
    env = build_scenario_environment_profile()
    registry = resolve_prompt_registry(env.prompt_profile)
    assert isinstance(registry, YamlPromptRegistry)


def test_missing_llm_configuration_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    env = build_scenario_environment_profile()

    def _raise(*_args, **_kwargs):
        raise ValueError("provider credentials missing")

    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.runtime_composition.resolve_llm_adapter",
        _raise,
    )

    with pytest.raises(RuntimeError, match="incident_scenario_llm_configuration_missing"):
        ensure_scenario_llm_adapter_resolved(env)


def test_investigator_delegates_build_context_to_composition_boundary() -> None:
    source = Path(investigator_agent.__file__).read_text(encoding="utf-8")
    assert "build_agent_runtime_context(" in source
    assert "FakeLLMAdapter" not in source
    assert "MagicMock" not in source
