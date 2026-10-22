# © Artur Czarnecki. All rights reserved.

"""QUAL-X-P2 — governance / security / tenant negative ingress regression matrix (Q04/Q08)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id
from intergrax.contracts.execution_intake import (
    CanonicalExecutionIntakePort,
    CanonicalExecutionIntakeRequest,
    CanonicalExecutionIntakeResult,
)
from intergrax.contracts.root_execution_launch import (
    RootExecutionLaunchDisposition,
    RootExecutionLaunchRequest,
)
from intergrax.contracts.root_execution_operation import RootExecutionOperation
from intergrax.contracts.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.default_root_execution_launcher import DefaultRootExecutionLauncher
from intergrax.runtime.governance.execution_admission_composition import (
    build_root_execution_authority_admission,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    DenyingRuntimeExecutionPolicyAdmission,
)
from testing_support.root_execution_launch_fixtures import sample_admitted_root_governance_identity
from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_missing_from_collection,
)
from tests.qualification.qual_x._p2_ingress_catalog import (
    QUAL_X_P2_INGRESS_CATALOG,
    QUAL_X_P2_START_HEAD,
    QualXP2EvidenceStrength,
    qual_x_p2_all_proof_pytest_node_ids,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
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


def _launch_request(operation: RootExecutionOperation) -> RootExecutionLaunchRequest[_Payload, str]:
    return RootExecutionLaunchRequest(
        admitted_governance_identity=sample_admitted_root_governance_identity(),
        root_execution_operation=operation,
        collaborative_authority_scopes=(_READ,),
        effective_authority_decision=EffectiveAuthorityDecision(
            decision=PolicyDecision(action=PolicyAction.ALLOW, reason="upstream"),
        ),
        payload=_Payload("ok"),
    )


def test_qual_x_p2_catalog_covers_g1_through_g8() -> None:
    ids = {entry.ingress_id for entry in QUAL_X_P2_INGRESS_CATALOG}
    assert ids == {f"G{i}" for i in range(1, 9)}


def test_qual_x_p2_each_ingress_declares_direct_negative_proof() -> None:
    for entry in QUAL_X_P2_INGRESS_CATALOG:
        assert entry.primary_proof_node_ids, entry.ingress_id
        assert entry.strength is QualXP2EvidenceStrength.DIRECT, entry.ingress_id


def test_qual_x_p2_all_proof_pytest_node_ids_are_collectable() -> None:
    combined = qual_x_p2_all_proof_pytest_node_ids()
    missing, proc = pytest_nodes_missing_from_collection(combined, _REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, f"missing={len(missing)}: {sorted(missing)}"


def test_qual_x_p2_start_head_documented() -> None:
    assert len(QUAL_X_P2_START_HEAD) == 40


def test_qual_x_p2_g1_runtime_policy_admission_rejects_blank_tenant_before_evaluator() -> None:
    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        RuntimeExecutionPolicyAdmissionRequest(
            tenant_id="   ",
            workspace_id="workspace-x",
            principal_id="principal-1",
            collaborative_authority_scopes=("workspace.read",),
        )
    configured = RuntimeExecutionPolicyAdmissionRequest(
        tenant_id="tenant-a",
        workspace_id="workspace-x",
        principal_id="principal-1",
        collaborative_authority_scopes=("workspace.read",),
    )
    deny = DenyingRuntimeExecutionPolicyAdmission().evaluate(configured)
    assert deny.policy_decision.action is PolicyAction.DENY


@pytest.mark.asyncio
async def test_qual_x_p2_q05_policy_deny_blocks_intake_without_physical_dispatch() -> None:
    intake = _RecordingIntake()
    launcher = DefaultRootExecutionLauncher(
        root_authority_admission=build_root_execution_authority_admission(
            runtime_policy_admission=DenyingRuntimeExecutionPolicyAdmission(),
        ),
        execution_intake=intake,
    )
    result = await launcher.launch(_launch_request(RootExecutionOperation.ROOT_AGENT))
    assert result.disposition is RootExecutionLaunchDisposition.DENIED
    assert intake.calls == 0
