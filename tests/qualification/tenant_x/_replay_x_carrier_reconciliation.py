# © Artur Czarnecki. All rights reserved.

"""TENANT-X replay-head carrier reconciliation @ ROADMAP-REPLAY-X L2 corrective wave."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Final

from tests.qualification.tenant_x._inventory import raw_discovery_candidates

TENANT_X_FINAL_IMMUTABLE_CARRIER_TOTAL: Final[int] = 800
# Re-pinned once by T-IGX-041-I (PDC-X P7): 800 + 19 (REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS)
# + 15 (REPLAY_P7_ADDED_CLASSIFICATION) - 3 (REPLAY_P7_REMOVED_SURFACE_IDS). Was 819.
TENANT_X_REPLAY_HEAD_CARRIER_TOTAL: Final[int] = 831
# Net delta of the pin against TENANT-X-FINAL: 19 classified at ROADMAP-REPLAY-X plus the P7 net (+12).
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


# --- T-IGX-041-I (PDC-X P7): carriers added and removed after the pin commit 91649e5d8 ---------------
# Closed role vocabulary of the classification below.
REPLAY_P7_ROLE_TENANT_SCOPED: Final[str] = "TENANT_SCOPED"  # owning tenant of the record or request; an isolation input
REPLAY_P7_ROLE_TENANT_FREE: Final[str] = "TENANT_FREE"  # explicitly not an isolation input (a reason is mandatory)
REPLAY_P7_ROLES: Final[frozenset[str]] = frozenset({REPLAY_P7_ROLE_TENANT_SCOPED, REPLAY_P7_ROLE_TENANT_FREE})

_S: Final[str] = REPLAY_P7_ROLE_TENANT_SCOPED
_SUFFIX: Final[str] = ".tenant_id::class.field.tenant"

# surface_id -> (role, reason). Reasons come from reading each class; the classifier inventory values
# (family, requirement) are not changed here.
REPLAY_P7_ADDED_CLASSIFICATION: Final[Mapping[str, tuple[str, str]]] = {
    "intergrax/applications/_shared/evaluation_wiring.py::ApplicationEvaluationWiring" + _SUFFIX: (
        _S,
        "Optional tenant (str | None) the Tier-3 evaluation wiring is bound to: "
        "wire_application_evaluation(tenant_id=...) binds the registry to it; None keeps the legacy unscoped registry.",
    ),
    "intergrax/contracts/artifact_ref.py::ArtifactScope" + _SUFFIX: (
        _S,
        "Required non-empty owning tenant of an artifact scope (tenant and optional workspace).",
    ),
    "intergrax/contracts/evaluation_contracts.py::EvalComparison" + _SUFFIX: (
        _S,
        "Required owning tenant of an advisory baseline-vs-candidate comparison record.",
    ),
    "intergrax/contracts/evaluation_contracts.py::EvalRun" + _SUFFIX: (
        _S,
        "Required owning tenant of one evaluation run record.",
    ),
    "intergrax/eval/typed_run.py::EvalRunProvenance" + _SUFFIX: (
        _S,
        "Required tenant among the pinned inputs of an evaluation run (frozen dataclass).",
    ),
    "intergrax/memory/retention_sweep.py::MemoryRetentionSweepOutput" + _SUFFIX: (
        _S,
        "Required tenant of the memory retention sweep whose counters the output reports.",
    ),
    "intergrax/memory/retention_sweep.py::MemoryRetentionSweepPayload" + _SUFFIX: (
        _S,
        "Required tenant of the sweep request in the background-task wire payload; the payload carries no authority "
        "(identity is resolved), so the field names the tenant to sweep and is not a trust root.",
    ),
    "intergrax/memory/retention_sweep.py::MemoryRetentionSweepReport" + _SUFFIX: (
        _S,
        "Required tenant of one memory retention sweep outcome (ids only, never memory content).",
    ),
    "intergrax/runtime/architecture/online_evaluation_registry.py::TenantOnlineEvaluationObservation" + _SUFFIX: (
        _S,
        "Optional tenant of an online evaluation observation: None only for legacy untenanted records, and a "
        "non-None value must be non-blank.",
    ),
    "intergrax/runtime/long_running/recurrence.py::RecurrenceRule" + _SUFFIX: (
        _S,
        "Required tenant of the paused task the recurrence rule targets (one task_id and resume_token).",
    ),
    "intergrax/runtime/nexus/artifacts/context_reference.py::ArtifactReferenceToken" + _SUFFIX: (
        _S,
        "Required tenant of the artifact version the self-verifying reference token points to.",
    ),
    "intergrax/runtime/nexus/artifacts/governed_access.py::ArtifactReadRequest" + _SUFFIX: (
        _S,
        "Required tenant of a governed artifact read request.",
    ),
    "intergrax/runtime/nexus/artifacts/governed_access.py::GovernedDownloadGrant" + _SUFFIX: (
        _S,
        "Required tenant a governed download grant was issued for (together with its artifact_id).",
    ),
    "intergrax/runtime/nexus/artifacts/retention.py::ArtifactRetentionSweepOutput" + _SUFFIX: (
        _S,
        "Required tenant of the artifact retention sweep whose result the output reports.",
    ),
    "intergrax/runtime/nexus/artifacts/retention.py::RetentionSweepReport" + _SUFFIX: (
        _S,
        "Required tenant of one artifact retention sweep report.",
    ),
}

# Surfaces of the 819-pin that no longer exist: B-D8 (92ee26bf3) removed the tenant field from these
# application artifact reference models.
REPLAY_P7_REMOVED_SURFACE_IDS: Final[frozenset[str]] = frozenset(
    {
        "intergrax/applications/contracts/application_artifacts.py::ApplicationArtifactRef" + _SUFFIX,
        "intergrax/applications/contracts/application_artifacts.py::SandboxArtifactRef" + _SUFFIX,
        "intergrax/applications/contracts/application_artifacts.py::WorkspaceArtifactRef" + _SUFFIX,
    }
)

# SHA-256 (replay_class_field_ids_sha256) of all 831 class-field surface ids at the T-IGX-041-I re-pin.
# A swap that keeps the count (one carrier removed, another added) changes it.
TENANT_X_REPLAY_CLASS_FIELD_IDS_SHA256: Final[str] = "6f81e7ea484fcfa80fa54e3646e320de35111be35d0c83a605a0aceabfadbb7e"


def replay_class_field_ids_sha256(ids: Iterable[str]) -> str:
    """SHA-256 of the sorted class-field surface ids joined by newlines (no trailing newline)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def replay_carrier_delta_violations(
    *,
    discovered: frozenset[str],
    pin_total: int,
    final_total: int = TENANT_X_FINAL_IMMUTABLE_CARRIER_TOTAL,
    base_ids: frozenset[str] = REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS,
    added: Mapping[str, tuple[str, str]] | None = None,
    removed: frozenset[str] | None = None,
    ids_sha256: str | None = None,
) -> list[str]:
    """Return every way the discovered class-field carriers disagree with the classification.

    Pure: the arguments default to the module data, so tests can pass mutated copies.
    ``discovered`` is the set of class-field ``surface_id`` values found by the scan.
    """
    added_map = REPLAY_P7_ADDED_CLASSIFICATION if added is None else added
    removed_ids = REPLAY_P7_REMOVED_SURFACE_IDS if removed is None else removed
    expected_digest = TENANT_X_REPLAY_CLASS_FIELD_IDS_SHA256 if ids_sha256 is None else ids_sha256
    added_ids = frozenset(added_map)
    found: list[str] = []

    expected_delta = len(base_ids) + len(added_ids) - len(removed_ids)
    if pin_total != final_total + expected_delta:
        found.append(
            f"pin {pin_total} != final {final_total} + base {len(base_ids)} + added {len(added_ids)}"
            f" - removed {len(removed_ids)}"
        )
    if len(discovered) - final_total != expected_delta:
        found.append(
            f"delta {len(discovered) - final_total} (discovered {len(discovered)}) != classified delta {expected_delta}"
        )
    if len(discovered) != pin_total:
        found.append(f"discovered total {len(discovered)} != pin {pin_total}")

    for name, left, right in (
        ("base/added", base_ids, added_ids),
        ("base/removed", base_ids, removed_ids),
        ("added/removed", added_ids, removed_ids),
    ):
        both = left & right
        if both:
            found.append(f"classification overlap {name}: {sorted(both)[:3]}")

    gone = (base_ids | added_ids) - discovered
    if gone:
        found.append(f"classified carriers not discovered: {sorted(gone)[:5]}")
    present = removed_ids & discovered
    if present:
        found.append(f"removed carriers still discovered: {sorted(present)[:5]}")

    for sid in sorted(added_ids):
        entry = added_map[sid]
        role, reason = entry if isinstance(entry, tuple) and len(entry) == 2 else (None, None)
        if role not in REPLAY_P7_ROLES:
            found.append(f"role {role!r} of {sid} is outside {sorted(REPLAY_P7_ROLES)}")
        if not isinstance(reason, str) or not reason.strip():
            found.append(f"reason of {sid} is missing")

    actual_digest = replay_class_field_ids_sha256(discovered)
    if actual_digest != expected_digest:
        found.append(f"class-field ids digest {actual_digest} != pinned {expected_digest}")
    return found


def assert_replay_carrier_delta_classified() -> None:
    discovered = frozenset(c.surface_id for c in raw_discovery_candidates() if c.discovery_kind == "class.field.tenant")
    violations = replay_carrier_delta_violations(
        discovered=discovered,
        pin_total=TENANT_X_REPLAY_HEAD_CARRIER_TOTAL,
        ids_sha256=TENANT_X_REPLAY_CLASS_FIELD_IDS_SHA256,
    )
    assert not violations, f"replay carrier delta violations: {violations}"
