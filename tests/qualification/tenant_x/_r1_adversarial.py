# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 targeted adversarial probes (synthetic / unit-level)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import TaskId, RunId
from intergrax.contracts.runtime_execution_admission import RootExecutionAuthorityAdmissionRequest
from intergrax.runtime.execution.agent_runtime_io import RuntimeRequest


def assert_runtime_request_tenant_a_cannot_become_b() -> None:
    req = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s",
        message="m",
        task_id=TaskId("task_00000000000000000000000000000001"),
        run_id=RunId("run_00000000000000000000000000000001"),
        tenant_id="tenant-a",
        metadata={"tenant_id": "tenant-b"},
    )
    with pytest.raises(ValueError, match="cannot override"):
        req.to_envelope()


def assert_root_admission_missing_tenant_fails_closed() -> None:
    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        RootExecutionAuthorityAdmissionRequest(
            tenant_id="   ",
            workspace_id="ws",
            principal_id="p",
            collaborative_authority_scopes=(),
            effective_authority_decision=_minimal_effective_authority(),
            root_execution_operation=_minimal_root_op(),
        )


def _minimal_effective_authority():
    from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
    from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

    return EffectiveAuthorityDecision(
        decision=PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="r1-probe",
            policy_rule_id="tenant-x.r1.probe",
        ),
    )


def _minimal_root_op():
    from intergrax.contracts.root_execution_operation import RootExecutionOperation

    return RootExecutionOperation.ROOT_AGENT


def assert_functional_evidence_rejects_missing_tenant() -> None:
    from intergrax.contracts.functional_evidence.tenant import (
        FunctionalEvidenceTenantIdentityError,
        require_tenant_id_from_exec_ctx,
    )

    class _EmptyCtx:
        canonical_request_identity = None

    with pytest.raises(FunctionalEvidenceTenantIdentityError):
        require_tenant_id_from_exec_ctx(_EmptyCtx())  # type: ignore[arg-type]
