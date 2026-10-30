# © Artur Czarnecki. All rights reserved.

"""TENANT-X replay-head carrier reconciliation @ ROADMAP-REPLAY-X L2 corrective wave."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._inventory import raw_discovery_candidates

TENANT_X_FINAL_IMMUTABLE_CARRIER_TOTAL: Final[int] = 800
TENANT_X_REPLAY_HEAD_CARRIER_TOTAL: Final[int] = 819
TENANT_X_REPLAY_CARRIER_DELTA: Final[int] = TENANT_X_REPLAY_HEAD_CARRIER_TOTAL - TENANT_X_FINAL_IMMUTABLE_CARRIER_TOTAL

# Surfaces introduced after TENANT-X-FINAL evidence @ 5ccb72f… (prod-effective tenant_id carriers).
REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS: Final[frozenset[str]] = frozenset(
    {
        "intergrax/applications/_shared/mcp_harness_principal.py::McpRunAgentIdentityResolution.tenant_id::class.field.tenant",
        "intergrax/contracts/certified_owner_loss_evidence.py::ExternalOperationOwnershipEpochV1.tenant_id::class.field.tenant",
        "intergrax/contracts/external_operation_identity.py::ExternalOperationIdentity.tenant_id::class.field.tenant",
        "intergrax/contracts/governance/history_continuity.py::ContinuityHead.tenant_id::class.field.tenant",
        "intergrax/contracts/governance/history_continuity.py::ContinuityScope.tenant_id::class.field.tenant",
        "intergrax/contracts/governed_external_work_create_follow_on.py::GovernedExternalWorkCreateFollowOnRecordV1.tenant_id::class.field.tenant",
        "intergrax/contracts/idempotency_store.py::InvocationClaim.tenant_id::class.field.tenant",
        "intergrax/contracts/sanctioned_external_work_operation.py::SanctionedExternalWorkAcceptQuoteRequest.tenant_id::class.field.tenant",
        "intergrax/contracts/sanctioned_external_work_operation.py::SanctionedExternalWorkCancelRequest.tenant_id::class.field.tenant",
        "intergrax/contracts/sanctioned_external_work_operation.py::SanctionedExternalWorkCreateRequest.tenant_id::class.field.tenant",
        "intergrax/contracts/tools/tool_invocation_wiring_resolution.py::ToolInvocationContext.tenant_id::class.field.tenant",
        "intergrax/contracts/worker_execution_commit_authority_loss_decision.py::WorkerExecutionCommitAuthorityLossDecisionV1.tenant_id::class.field.tenant",
        "intergrax/contracts/worker_execution_ownership_revocation.py::WorkerExecutionOwnershipRevocationAuthorizationV1.tenant_id::class.field.tenant",
        "intergrax/contracts/worker_execution_ownership_revocation.py::WorkerExecutionOwnershipRevocationV1.tenant_id::class.field.tenant",
        "intergrax/runtime/governance/continuity_providers/primary_register.py::HeadRecord.tenant_id::class.field.tenant",
        "intergrax/runtime/governance/continuity_providers/primary_register.py::TransitionRecord.tenant_id::class.field.tenant",
        "intergrax/runtime/governance/continuity_providers/witness_log.py::WitnessView.tenant_id::class.field.tenant",
        "intergrax/runtime/governance/local_history_continuity.py::_PreparedTransition.tenant_id::class.field.tenant",
        "intergrax/runtime/governance/offline_history_continuity.py::_Command.tenant_id::class.field.tenant",
    }
)


def assert_replay_carrier_delta_classified() -> None:
    assert len(REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS) == TENANT_X_REPLAY_CARRIER_DELTA
    known = {c.surface_id for c in raw_discovery_candidates()}
    missing = REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS - known
    assert not missing, f"missing replay delta carriers: {sorted(missing)[:5]}"
