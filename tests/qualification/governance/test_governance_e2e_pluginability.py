# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-4 — contract-first pluginability via composition (no monkeypatch)."""

from __future__ import annotations

from dataclasses import dataclass

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
from intergrax.contracts.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionPort,
    RuntimeExecutionPolicyAdmissionRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.default_root_execution_launcher import DefaultRootExecutionLauncher
from intergrax.runtime.governance.execution_admission_composition import (
    build_root_execution_authority_admission,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
    DenyingRuntimeExecutionPolicyAdmission,
    RuntimeExecutionPolicyAdmissionResult,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


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


class _CustomDenyAdmission(RuntimeExecutionPolicyAdmissionPort):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self,
        request: RuntimeExecutionPolicyAdmissionRequest,
    ) -> RuntimeExecutionPolicyAdmissionResult:
        self.calls += 1
        del request
        return RuntimeExecutionPolicyAdmissionResult(
            policy_decision=PolicyDecision(
                action=PolicyAction.DENY,
                reason="gov_final_4_custom_admission",
            ),
        )


def _launch_request() -> RootExecutionLaunchRequest[_Payload]:
    from intergrax.contracts.admitted_root_governance_identity import (
        AdmittedRootGovernanceIdentity,
    )

    return RootExecutionLaunchRequest(
        admitted_governance_identity=AdmittedRootGovernanceIdentity(
            tenant_id="tenant-plugin",
            workspace_id="workspace-plugin",
            principal_id="principal-plugin",
        ),
        root_execution_operation=RootExecutionOperation.ROOT_AGENT,
        collaborative_authority_scopes=("workspace.read",),
        effective_authority_decision=EffectiveAuthorityDecision(
            decision=PolicyDecision(action=PolicyAction.ALLOW, reason="upstream"),
        ),
        payload=_Payload("ok"),
    )


@pytest.mark.asyncio
async def test_scenario_plugin_custom_runtime_execution_policy_admission_via_composition_launcher() -> None:
    custom_policy = _CustomDenyAdmission()
    root_admission = build_root_execution_authority_admission(
        runtime_policy_admission=custom_policy,
    )
    recording_intake = _RecordingIntake()
    launcher = DefaultRootExecutionLauncher(
        root_authority_admission=root_admission,
        execution_intake=recording_intake,
    )
    result = await launcher.launch(_launch_request())
    assert custom_policy.calls == 1
    assert result.disposition is RootExecutionLaunchDisposition.DENIED
    assert recording_intake.calls == 0


def test_scenario_plugin_allowing_vs_denying_admission_via_composition() -> None:
    allow = AllowingRuntimeExecutionPolicyAdmission()
    deny = DenyingRuntimeExecutionPolicyAdmission()
    request = RuntimeExecutionPolicyAdmissionRequest(
        tenant_id="tenant-plugin",
        workspace_id="workspace-plugin",
        principal_id="principal-plugin",
        collaborative_authority_scopes=("workspace.read",),
    )
    assert allow.evaluate(request).policy_decision.action is PolicyAction.ALLOW
    assert deny.evaluate(request).policy_decision.action is PolicyAction.DENY
