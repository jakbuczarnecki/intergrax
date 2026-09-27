# © Artur Czarnecki. All rights reserved.

"""HARNESS-W4-R1 — dependency attempt boundary materialization."""

from __future__ import annotations

import pytest

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPolicy,
    DependencyConcurrencyPolicyBinding,
    DependencyConcurrencyAdmissionConfiguration,
)
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    ProductionRuntimeToolInvokerCompositionError,
    build_production_runtime_tool_invoker,
)
from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
    ToolDependencyAttemptBoundaryMaterializationError,
    materialize_tool_dependency_attempt_boundary,
)
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from pydantic import BaseModel

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    value: int


def _tool_config(tool_id: str = "probe.tool") -> DependencyConcurrencyAdmissionConfiguration:
    identity = DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.TOOL,
        value=tool_id,
    )
    policy = DependencyConcurrencyPolicy(
        max_concurrent_calls=1,
        overload_mode=DependencyConcurrencyOverloadMode.REJECT,
    )
    return DependencyConcurrencyAdmissionConfiguration(
        bindings=(DependencyConcurrencyPolicyBinding(identity=identity, policy=policy),),
    )


def test_materialize_non_production_without_config_returns_none() -> None:
    assert materialize_tool_dependency_attempt_boundary(None, production_mode=False) is None


def test_materialize_strict_production_without_config_fails_closed() -> None:
    with pytest.raises(ToolDependencyAttemptBoundaryMaterializationError):
        materialize_tool_dependency_attempt_boundary(None, production_mode=True)


def test_materialize_strict_production_without_tool_binding_fails_closed() -> None:
    llm_only = DependencyConcurrencyAdmissionConfiguration(
        bindings=(
            DependencyConcurrencyPolicyBinding(
                identity=DependencyConcurrencyIdentity(
                    kind=DependencyConcurrencyKind.LLM_PROVIDER,
                    value="openai",
                ),
                policy=DependencyConcurrencyPolicy(
                    max_concurrent_calls=1,
                    overload_mode=DependencyConcurrencyOverloadMode.REJECT,
                ),
            ),
        ),
    )
    with pytest.raises(ToolDependencyAttemptBoundaryMaterializationError):
        materialize_tool_dependency_attempt_boundary(llm_only, production_mode=True)


def test_materialize_strict_production_with_tool_binding_succeeds() -> None:
    boundary = materialize_tool_dependency_attempt_boundary(
        _tool_config(),
        production_mode=True,
    )
    assert boundary is not None
    boundary.close()


def test_build_production_runtime_tool_invoker_rejects_missing_boundary() -> None:
    from tests.unit.runtime.nexus.tools.test_gr10_r9_orchestration_mse import (
        _RecordingMseBoundary,
    )
    from intergrax.runtime.wiring.agent_runtime_governance_factory import (
        build_agent_runtime_governance_boundary,
        default_lab_capability_grants,
    )

    registry = FakeRegistry(
        ToolContract(
            tool_id="probe.tool",
            name="probe",
            description="probe",
            input_schema=_In,
            output_schema=_Out,
            side_effects=False,
            error_mapping={},
            risk_level=ToolRiskLevel.LOW,
        ),
    )
    with pytest.raises(ProductionRuntimeToolInvokerCompositionError, match="dependency_attempt_boundary"):
        build_production_runtime_tool_invoker(
            registry=registry,
            production_mode=True,
            agent_runtime_governance=build_agent_runtime_governance_boundary(
                capability_grants=default_lab_capability_grants("tenant-c5"),
            ),
            meaningful_side_effect_authorization=_RecordingMseBoundary(allow=True),
        )
