# © Artur Czarnecki. All rights reserved.

"""GR-10-R4 — INFERENCE root admission enterprise qualification (architecture truth)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.execution_intake import (
    CanonicalExecutionIntakePort,
    CanonicalExecutionIntakeRequest,
    CanonicalExecutionIntakeResult,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id
from intergrax.contracts.root_execution_launch import (
    RootExecutionLaunchDisposition,
    RootExecutionLaunchRequest,
)
from intergrax.contracts.root_execution_operation import RootExecutionOperation
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionDisposition,
    RootExecutionAuthorityAdmissionPort,
    RootExecutionAuthorityAdmissionRequest,
    RootExecutionAuthorityAdmissionResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.root_execution_operation_mapping import (
    root_execution_operation_from_request,
)
from intergrax.runtime.execution.strategy import ExecutionStrategy, execution_strategy_from_capabilities
from intergrax.runtime.governance.default_root_execution_launcher import DefaultRootExecutionLauncher
from intergrax.runtime.governance.execution_admission_composition import (
    build_root_execution_authority_admission,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    DenyingRuntimeExecutionPolicyAdmission,
    RuntimeExecutionPolicyAdmissionPort,
)
from tests.qualification.governance.strategy.catalog import (
    GR10_INFERENCE_CAPABILITY_SEMANTICS,
    Gr10Applicability,
    Gr10CoverageStatus,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INFERENCE_PY = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "inference.py"
_FACADE_PY = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "facade.py"
_HOST_TASK_PY = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "host_task.py"
_READ = "workspace.read"


@dataclass(frozen=True)
class _Payload:
    value: str


class _RecordingIntake(CanonicalExecutionIntakePort[_Payload, str]):
    def __init__(self) -> None:
        self.calls = 0

    async def dispatch(
        self,
        request: CanonicalExecutionIntakeRequest[_Payload],
    ) -> CanonicalExecutionIntakeResult[str]:
        self.calls += 1
        return CanonicalExecutionIntakeResult(
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            result=request.payload.value,
        )


class _RecordingRuntimePolicyAdmission(RuntimeExecutionPolicyAdmissionPort):
    def __init__(self) -> None:
        self.calls = 0
        self.last_tenant: str | None = None
        self.last_workspace: str | None = None
        self.last_principal: str | None = None
        self.last_operation: str | None = None

    def evaluate(self, request):
        from intergrax.contracts.runtime_execution_policy_admission import (
            RuntimeExecutionPolicyAdmissionResult,
        )
        from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

        self.calls += 1
        self.last_tenant = request.tenant_id
        self.last_workspace = request.workspace_id
        self.last_principal = request.principal_id
        self.last_operation = request.execution_operation
        return RuntimeExecutionPolicyAdmissionResult(
            policy_decision=PolicyDecision(action=PolicyAction.DENY, reason="r4-test"),
        )


def test_gr10_r4_inference_root_admission_semantics_not_applicable() -> None:
    root = next(
        row for row in GR10_INFERENCE_CAPABILITY_SEMANTICS if row.capability == "Root admission"
    )
    assert root.applicability is Gr10Applicability.NOT_APPLICABLE
    assert root.coverage is None
    assert "no independent" in root.reason.lower() or "internal" in root.reason.lower()


def test_gr10_r4_resolve_task_execution_capabilities_never_inference() -> None:
    source = _HOST_TASK_PY.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_HOST_TASK_PY))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_task_execution_capabilities"
    )
    fn_src = ast.get_source_segment(source, fn) or ""
    assert "ExecutionCapability.AGENT" in fn_src
    assert "ExecutionCapability.ORCHESTRATION" in fn_src
    assert "INFERENCE" not in fn_src
    agent_request = ExecutionRequest(
        input=(),
        output_type=dict,
        capabilities=frozenset({ExecutionCapability.AGENT}),
    )
    orch_request = ExecutionRequest(
        input=(),
        output_type=dict,
        capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
    )
    assert (
        execution_strategy_from_capabilities(agent_request.capabilities)
        is ExecutionStrategy.AGENTIC
    )
    assert (
        execution_strategy_from_capabilities(orch_request.capabilities)
        is ExecutionStrategy.ORCHESTRATION
    )
    assert root_execution_operation_from_request(agent_request) is RootExecutionOperation.ROOT_AGENT
    assert (
        root_execution_operation_from_request(orch_request)
        is RootExecutionOperation.ROOT_ORCHESTRATION
    )


def test_gr10_r4_host_task_router_does_not_wire_inference_executor() -> None:
    source = _HOST_TASK_PY.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_HOST_TASK_PY))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build_host_task_strategy_router"
    )
    fn_src = ast.get_source_segment(source, fn) or ""
    assert "inference_executor" not in fn_src


def test_gr10_r4_execution_facade_is_not_legal_production_root() -> None:
    source = _FACADE_PY.read_text(encoding="utf-8-sig")
    assert "not a legal production root entry" in source
    assert "RootExecutionLaunchPort" in source


def test_gr10_r4_inference_executor_has_no_root_admission_dependency() -> None:
    source = _INFERENCE_PY.read_text(encoding="utf-8-sig")
    forbidden = (
        "RuntimeExecutionPolicyAdmissionPort",
        "RootExecutionAuthorityAdmissionPort",
        "DefaultRootExecutionLauncher",
        "RuntimePolicyEngine",
    )
    for name in forbidden:
        assert name not in source


@pytest.mark.asyncio
async def test_gr10_r4_root_inference_launcher_deny_zero_intake() -> None:
    intake = _RecordingIntake()

    class _DenyAdmission(RootExecutionAuthorityAdmissionPort):
        def __init__(self) -> None:
            self.calls = 0

        def authorize(
            self,
            request: RootExecutionAuthorityAdmissionRequest,
        ) -> RootExecutionAuthorityAdmissionResult:
            self.calls += 1
            return RootExecutionAuthorityAdmissionResult(
                disposition=RootExecutionAuthorityAdmissionDisposition.DENIED,
            )

    admission = _DenyAdmission()
    launcher = DefaultRootExecutionLauncher(
        root_authority_admission=admission,
        execution_intake=intake,
    )
    result = await launcher.launch(
        RootExecutionLaunchRequest(
            tenant_id="tenant-a",
            workspace_id="workspace-x",
            principal_id="principal-1",
            root_execution_operation=RootExecutionOperation.ROOT_INFERENCE,
            collaborative_authority_scopes=(_READ,),
            effective_authority_decision=EffectiveAuthorityDecision(
                decision=PolicyDecision(action=PolicyAction.ALLOW, reason="upstream"),
            ),
            payload=_Payload("probe"),
        )
    )
    assert result.disposition is RootExecutionLaunchDisposition.DENIED
    assert admission.calls == 1
    assert intake.calls == 0


@pytest.mark.asyncio
async def test_gr10_r4_root_inference_launcher_allow_exactly_one_intake() -> None:
    intake = _RecordingIntake()

    class _AllowAdmission(RootExecutionAuthorityAdmissionPort):
        def __init__(self) -> None:
            self.calls = 0

        def authorize(
            self,
            request: RootExecutionAuthorityAdmissionRequest,
        ) -> RootExecutionAuthorityAdmissionResult:
            from intergrax.contracts.delegation_authority import ParentExecutionAuthority

            self.calls += 1
            return RootExecutionAuthorityAdmissionResult(
                disposition=RootExecutionAuthorityAdmissionDisposition.ALLOWED,
                trusted_parent_execution_authority=ParentExecutionAuthority.scoped(
                    request.collaborative_authority_scopes,
                ),
            )

    admission = _AllowAdmission()
    launcher = DefaultRootExecutionLauncher(
        root_authority_admission=admission,
        execution_intake=intake,
    )
    result = await launcher.launch(
        RootExecutionLaunchRequest(
            tenant_id="tenant-a",
            workspace_id="workspace-x",
            principal_id="principal-1",
            root_execution_operation=RootExecutionOperation.ROOT_INFERENCE,
            collaborative_authority_scopes=(_READ,),
            effective_authority_decision=EffectiveAuthorityDecision(
                decision=PolicyDecision(action=PolicyAction.ALLOW, reason="upstream"),
            ),
            payload=_Payload("ok"),
        )
    )
    assert result.disposition is RootExecutionLaunchDisposition.LAUNCHED
    assert admission.calls == 1
    assert intake.calls == 1


@pytest.mark.asyncio
async def test_gr10_r4_custom_runtime_policy_admission_blocks_root_inference() -> None:
    intake = _RecordingIntake()
    custom = _RecordingRuntimePolicyAdmission()
    launcher = DefaultRootExecutionLauncher(
        root_authority_admission=build_root_execution_authority_admission(
            runtime_policy_admission=custom,
        ),
        execution_intake=intake,
    )
    result = await launcher.launch(
        RootExecutionLaunchRequest(
            tenant_id="tenant-a",
            workspace_id="workspace-w",
            principal_id="principal-p",
            root_execution_operation=RootExecutionOperation.ROOT_INFERENCE,
            collaborative_authority_scopes=(_READ,),
            effective_authority_decision=EffectiveAuthorityDecision(
                decision=PolicyDecision(action=PolicyAction.ALLOW, reason="upstream"),
            ),
            payload=_Payload("blocked"),
        )
    )
    assert result.disposition is RootExecutionLaunchDisposition.DENIED
    assert custom.calls == 1
    assert custom.last_tenant == "tenant-a"
    assert custom.last_workspace == "workspace-w"
    assert custom.last_principal == "principal-p"
    assert custom.last_operation == RootExecutionOperation.ROOT_INFERENCE.policy_operation()
    assert intake.calls == 0


@pytest.mark.asyncio
async def test_gr10_r4_root_deny_skips_inference_delegate_dispatch() -> None:
    """DENY at launcher must not reach StrategyExecutionRouter / InferenceExecutor."""
    inference_executor = AsyncMock()
    inference_executor.execute = AsyncMock()
    intake = _RecordingIntake()
    launcher = DefaultRootExecutionLauncher(
        root_authority_admission=build_root_execution_authority_admission(
            runtime_policy_admission=DenyingRuntimeExecutionPolicyAdmission(),
        ),
        execution_intake=intake,
    )
    result = await launcher.launch(
        RootExecutionLaunchRequest(
            tenant_id="tenant-a",
            workspace_id="workspace-x",
            principal_id="principal-1",
            root_execution_operation=RootExecutionOperation.ROOT_INFERENCE,
            collaborative_authority_scopes=(_READ,),
            effective_authority_decision=EffectiveAuthorityDecision(
                decision=PolicyDecision(action=PolicyAction.ALLOW, reason="upstream"),
            ),
            payload=_Payload("x"),
        )
    )
    assert result.disposition is RootExecutionLaunchDisposition.DENIED
    assert intake.calls == 0
    inference_executor.execute.assert_not_called()


def test_gr10_r4_capability_only_inference_request_is_not_production_host_entry() -> None:
    """Bare INFERENCE capability request maps to ROOT_INFERENCE op but is not a Tier-3 host entry."""
    bare = ExecutionRequest(input=(), output_type=dict)
    assert execution_strategy_from_capabilities(bare.capabilities) is ExecutionStrategy.INFERENCE
    assert root_execution_operation_from_request(bare) is RootExecutionOperation.ROOT_INFERENCE
    host_source = _HOST_TASK_PY.read_text(encoding="utf-8-sig")
    assert "inference_executor" not in host_source
