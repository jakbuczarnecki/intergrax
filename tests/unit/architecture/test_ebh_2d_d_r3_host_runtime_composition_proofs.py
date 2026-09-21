# © Artur Czarnecki. All rights reserved.

"""EBH-2D-D-R3 — replaceability proofs for host runtime composition."""

from __future__ import annotations

import ast
from dataclasses import fields

import pytest

from dispute_sim_application.host.host_runtime_composition import DisputeSimHostRuntimeComposition
from dispute_sim_application.host.orchestration_decision_requirement_policy import (
    default_dispute_sim_harness_orchestration_decision_requirement_policy,
    resolve_dispute_sim_harness_orchestration_decision_requirement_policy,
)
from dispute_sim_application.host.settings import DisputeSimBackendSettings
from legal_application.host.host_runtime_composition import LegalHostRuntimeComposition
from legal_application.host.orchestration_decision_requirement_policy import (
    default_legal_harness_orchestration_decision_requirement_policy,
    resolve_legal_harness_orchestration_decision_requirement_policy,
)
from legal_application.host.settings import LegalBackendSettings
from local_workspace_application.host.host_runtime_composition import (
    LocalWorkspaceHostRuntimeComposition,
)
from local_workspace_application.host.orchestration_decision_requirement_policy import (
    default_local_workspace_harness_orchestration_decision_requirement_policy,
    resolve_local_workspace_harness_orchestration_decision_requirement_policy,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from research_application.host.host_runtime_composition import ResearchHostRuntimeComposition
from research_application.host.orchestration_decision_requirement_policy import (
    default_research_harness_orchestration_decision_requirement_policy,
    resolve_research_harness_orchestration_decision_requirement_policy,
)
from research_application.host.settings import ResearchBackendSettings
from research_application.host.tool_wiring import wire_research_tools
from intergrax.contracts.decision_requirement_policy import DecisionRequirement

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _CustomDecisionRequirementPolicy:
    def evaluate(self, context: object) -> object:
        return DecisionRequirement.NOT_REQUIRED


class _StubWebSearchExecutor:
    def search_sync(self, query: str, top_k: int | None = None, **_: object) -> list[object]:
        return []


def test_settings_from_env_does_not_materialize_runtime_policy_or_executor() -> None:
    for cls in (
        LegalBackendSettings,
        DisputeSimBackendSettings,
        ResearchBackendSettings,
        LocalWorkspaceBackendSettings,
    ):
        field_names = {f.name for f in fields(cls)}
        assert "orchestration_decision_requirement_policy" not in field_names
    research_fields = {f.name for f in fields(ResearchBackendSettings)}
    assert "websearch_executor" not in research_fields
    settings = ResearchBackendSettings.from_env()
    assert settings.enable_websearch is not None


@pytest.mark.parametrize(
    ("resolve", "default_fn", "composition_cls"),
    [
        (
            resolve_legal_harness_orchestration_decision_requirement_policy,
            default_legal_harness_orchestration_decision_requirement_policy,
            LegalHostRuntimeComposition,
        ),
        (
            resolve_dispute_sim_harness_orchestration_decision_requirement_policy,
            default_dispute_sim_harness_orchestration_decision_requirement_policy,
            DisputeSimHostRuntimeComposition,
        ),
        (
            resolve_research_harness_orchestration_decision_requirement_policy,
            default_research_harness_orchestration_decision_requirement_policy,
            ResearchHostRuntimeComposition,
        ),
        (
            resolve_local_workspace_harness_orchestration_decision_requirement_policy,
            default_local_workspace_harness_orchestration_decision_requirement_policy,
            LocalWorkspaceHostRuntimeComposition,
        ),
    ],
)
def test_custom_policy_via_runtime_composition_without_settings_mutation(
    resolve: object,
    default_fn: object,
    composition_cls: type,
) -> None:
    custom = _CustomDecisionRequirementPolicy()
    runtime_a = composition_cls(orchestration_decision_requirement_policy=custom)
    runtime_b = composition_cls()
    policy_a = resolve(runtime_a.orchestration_decision_requirement_policy)  # type: ignore[call-arg, attr-defined]
    policy_b = resolve(runtime_b.orchestration_decision_requirement_policy)  # type: ignore[call-arg, attr-defined]
    assert policy_a is custom
    assert type(policy_b) is type(default_fn())  # type: ignore[operator]


def test_research_websearch_executor_replaceable_without_settings_mutation() -> None:
    settings = ResearchBackendSettings.from_env()
    settings_id = id(settings)
    executor_a = _StubWebSearchExecutor()
    executor_b = _StubWebSearchExecutor()
    runtime_a = ResearchHostRuntimeComposition(websearch_executor=executor_a)
    runtime_b = ResearchHostRuntimeComposition(websearch_executor=executor_b)
    assert runtime_a.websearch_executor is executor_a
    assert runtime_b.websearch_executor is executor_b
    assert id(settings) == settings_id
    wiring_a = wire_research_tools(
        settings=settings,
        websearch_executor=runtime_a.websearch_executor,
    )
    wiring_b = wire_research_tools(
        settings=settings,
        websearch_executor=runtime_b.websearch_executor,
    )
    assert wiring_a.wiring_context.websearch_executor is executor_a
    assert wiring_b.wiring_context.websearch_executor is executor_b


def test_research_settings_ast_has_no_websearch_executor() -> None:
    from pathlib import Path

    path = Path("applications/research_application/host/settings.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assert node.target.id != "websearch_executor"
