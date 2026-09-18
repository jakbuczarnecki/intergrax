# © Artur Czarnecki. All rights reserved.

"""GR-2-R3 root launcher qualification."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.execution_intake import (
    CanonicalExecutionIntakePort,
    CanonicalExecutionIntakeRequest,
    CanonicalExecutionIntakeResult,
)
from intergrax.contracts.root_execution_launch import RootExecutionLaunchDisposition
from intergrax.contracts.root_execution_operation import RootExecutionOperation
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionDisposition,
    RootExecutionAuthorityAdmissionPort,
    RootExecutionAuthorityAdmissionRequest,
    RootExecutionAuthorityAdmissionResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id
from intergrax.runtime.governance.default_root_execution_launcher import DefaultRootExecutionLauncher
from intergrax.runtime.governance.execution_admission_composition import (
    build_root_execution_authority_admission,
)
from testing_support.root_execution_launch_fixtures import sample_admitted_root_governance_identity
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
    DenyingRuntimeExecutionPolicyAdmission,
)

pytestmark = pytest.mark.unit

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


class _StaticAdmission(RootExecutionAuthorityAdmissionPort):
    def __init__(self, disposition: RootExecutionAuthorityAdmissionDisposition) -> None:
        self.calls = 0
        self._disposition = disposition

    def authorize(
        self,
        request: RootExecutionAuthorityAdmissionRequest,
    ) -> RootExecutionAuthorityAdmissionResult:
        self.calls += 1
        if self._disposition is not RootExecutionAuthorityAdmissionDisposition.ALLOWED:
            return RootExecutionAuthorityAdmissionResult(disposition=self._disposition)
        from intergrax.contracts.delegation_authority import ParentExecutionAuthority

        return RootExecutionAuthorityAdmissionResult(
            disposition=RootExecutionAuthorityAdmissionDisposition.ALLOWED,
            trusted_parent_execution_authority=ParentExecutionAuthority.scoped(
                request.collaborative_authority_scopes,
            ),
        )


def _launch_request(
    operation: RootExecutionOperation,
) -> object:
    from intergrax.contracts.root_execution_launch import RootExecutionLaunchRequest

    return RootExecutionLaunchRequest(
        admitted_governance_identity=sample_admitted_root_governance_identity(),
        root_execution_operation=operation,
        collaborative_authority_scopes=(_READ,),
        effective_authority_decision=EffectiveAuthorityDecision(
            decision=PolicyDecision(action=PolicyAction.ALLOW, reason="upstream"),
        ),
        payload=_Payload("ok"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        RootExecutionOperation.ROOT_INFERENCE,
        RootExecutionOperation.ROOT_AGENT,
        RootExecutionOperation.ROOT_ORCHESTRATION,
    ],
)
async def test_launcher_allow_runs_admission_and_intake_once(operation: RootExecutionOperation) -> None:
    intake = _RecordingIntake()
    admission = _StaticAdmission(RootExecutionAuthorityAdmissionDisposition.ALLOWED)
    launcher = DefaultRootExecutionLauncher(
        root_authority_admission=admission,
        execution_intake=intake,
    )
    result = await launcher.launch(_launch_request(operation))
    assert result.disposition is RootExecutionLaunchDisposition.LAUNCHED
    assert admission.calls == 1
    assert intake.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        RootExecutionOperation.ROOT_INFERENCE,
        RootExecutionOperation.ROOT_AGENT,
        RootExecutionOperation.ROOT_ORCHESTRATION,
    ],
)
async def test_launcher_deny_skips_intake(operation: RootExecutionOperation) -> None:
    intake = _RecordingIntake()
    admission = _StaticAdmission(RootExecutionAuthorityAdmissionDisposition.DENIED)
    launcher = DefaultRootExecutionLauncher(root_authority_admission=admission, execution_intake=intake)
    result = await launcher.launch(_launch_request(operation))
    assert result.disposition is RootExecutionLaunchDisposition.DENIED
    assert intake.calls == 0


@pytest.mark.asyncio
async def test_launcher_policy_plugin_switch_without_consumer_change() -> None:
    intake = _RecordingIntake()
    allow_launcher = DefaultRootExecutionLauncher(
        root_authority_admission=build_root_execution_authority_admission(
            runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
        ),
        execution_intake=intake,
    )
    deny_launcher = DefaultRootExecutionLauncher(
        root_authority_admission=build_root_execution_authority_admission(
            runtime_policy_admission=DenyingRuntimeExecutionPolicyAdmission(),
        ),
        execution_intake=intake,
    )
    request = _launch_request(RootExecutionOperation.ROOT_AGENT)
    allow = await allow_launcher.launch(request)
    deny = await deny_launcher.launch(request)
    assert allow.disposition is RootExecutionLaunchDisposition.LAUNCHED
    assert deny.disposition is RootExecutionLaunchDisposition.DENIED
