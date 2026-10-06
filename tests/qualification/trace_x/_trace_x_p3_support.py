# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P3 tool / provider / side-effect authorization attribution SSOT (qualification-local)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.trace_x._trace_x_p0_support import repo_root

TRACE_X_P3_START_HEAD: Final[str] = "3572e6ed1c894859ac419770126e02d79e07208e"

MANDATORY_FRZ_P3_IDS: Final[tuple[str, ...]] = (
    "FRZ-TRC-03",
    "FRZ-TRC-04",
    "FRZ-TRC-06",
)

OUT_OF_SCOPE_FRZ: Final[tuple[str, ...]] = tuple(
    f"FRZ-TRC-{i:02d}"
    for i in range(1, 13)
    if f"FRZ-TRC-{i:02d}" not in MANDATORY_FRZ_P3_IDS
)

PROPOSED_CHILD_ON_BLOCKER: Final[str] = "TRACE-X-P3-R1"


class P3GateResult(StrEnum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"


class P3ReadinessStatus(StrEnum):
    READY_FOR_AUDIT = "READY FOR AUDIT"
    BLOCKED = "BLOCKED"


class FrzTrcP3Disposition(StrEnum):
    READY_FOR_INDEPENDENT_CLOSURE_REVIEW = "READY FOR INDEPENDENT CLOSURE REVIEW"
    BLOCKED = "BLOCKED"


class P3BlockerClassification(StrEnum):
    IN_SCOPE_BLOCKER = "IN-SCOPE BLOCKER"
    TRACKED_FREEZE_DEBT = "TRACKED FREEZE DEBT"
    ENVIRONMENT_TEST_ISSUE = "ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED"


class P3Concern(StrEnum):
    TOOL_INVOCATION = "tool_invocation"
    PROVIDER_INVOCATION = "provider_invocation"
    SIDE_EFFECT_AUTHORIZATION = "side_effect_authorization"
    GOVERNANCE_EVIDENCE = "governance_evidence"
    ATTESTATION_EXPORT = "attestation_export"


class P3IdentityField(StrEnum):
    TENANT = "tenant_id"
    TASK = "task_id"
    RUN = "run_id"
    ATTEMPT = "attempt_id"
    EXECUTION = "execution_id"
    TOOL = "tool_id"
    PROVIDER = "provider_id"
    PROVIDER_INVOCATION = "invocation_id"
    GOVERNANCE_EVIDENCE = "evidence_id"
    EFFECT = "effect_invocation_ref"


@dataclass(frozen=True, slots=True)
class P3IdentityCoverage:
    concern: P3Concern
    tenant: bool
    task: bool
    run: bool
    attempt: bool
    execution: bool
    concern_specific_id: bool
    authorization_id: bool
    notes: str


@dataclass(frozen=True, slots=True)
class P3OwnershipRow:
    concern: str
    semantic_owner: str
    producer: str
    persistence: str
    reader: str
    execution_identity: str
    authorization_identity: str
    tenant_scope: str
    fail_closed: str


@dataclass(frozen=True, slots=True)
class P3InScopeBlocker:
    blocker_id: str
    classification: P3BlockerClassification
    frz_id: str
    summary: str
    remediation: str


@dataclass(frozen=True, slots=True)
class P3EnterpriseAuditRow:
    area: str
    result: P3GateResult


@dataclass(frozen=True, slots=True)
class P3AdversarialCase:
    case_id: str
    concern: P3Concern
    description: str
    expected: str
    evidence_test: str


GOVERNED_BOUNDARY_MODULE: Final[str] = (
    "intergrax/contracts/execution_evidence/boundary_event.py"
)
ATTESTATION_BOUNDARY_MODULE: Final[str] = (
    "intergrax/runtime/attestation/execution_boundary_event.py"
)
TOOL_TRACE_BRIDGE_MODULE: Final[str] = "intergrax/runtime/events/trace_bridge.py"
TOOL_INVOKER_MODULE: Final[str] = "intergrax/runtime/nexus/tools/invoker.py"
GOVERNANCE_EVIDENCE_CONTRACT: Final[str] = (
    "intergrax/contracts/governed_execution_governance_evidence.py"
)

IDENTITY_MATRIX: Final[tuple[P3IdentityCoverage, ...]] = (
    P3IdentityCoverage(
        concern=P3Concern.TOOL_INVOCATION,
        tenant=True,
        task=True,
        run=True,
        attempt=True,
        execution=True,
        concern_specific_id=True,
        authorization_id=False,
        notes="RuntimeEvent envelope via trace→bridge; tool_id in TraceBridgePayload",
    ),
    P3IdentityCoverage(
        concern=P3Concern.PROVIDER_INVOCATION,
        tenant=True,
        task=True,
        run=True,
        attempt=False,
        execution=False,
        concern_specific_id=True,
        authorization_id=False,
        notes="Governed ExecutionBoundaryEvent + ProviderInvocationSection — run/task only",
    ),
    P3IdentityCoverage(
        concern=P3Concern.SIDE_EFFECT_AUTHORIZATION,
        tenant=True,
        task=True,
        run=True,
        attempt=True,
        execution=True,
        concern_specific_id=True,
        authorization_id=True,
        notes="GovernanceDecisionEvidenceFact CAN carry full correlation when populated",
    ),
    P3IdentityCoverage(
        concern=P3Concern.ATTESTATION_EXPORT,
        tenant=True,
        task=True,
        run=True,
        attempt=False,
        execution=False,
        concern_specific_id=True,
        authorization_id=False,
        notes="ExecutionBoundaryEventV1 — non-authoritative export; step_id not ExecutionId",
    ),
)

OWNERSHIP_MATRIX: Final[tuple[P3OwnershipRow, ...]] = (
    P3OwnershipRow(
        concern="tool invocation runtime facts",
        semantic_owner="RuntimeEvent",
        producer="RuntimeToolInvoker → state.trace_event → trace_event_to_runtime_event",
        persistence="EvidencePersistencePort / InMemoryRuntimeEventStore",
        reader="ExecutionReconstructor",
        execution_identity="tenant + TaskId + RunId + AttemptId + ExecutionId on RuntimeEvent",
        authorization_identity="governance gates pre-invoke (not in RuntimeEvent payload)",
        tenant_scope="task.tenant_id + RuntimeEvent.tenant_id",
        fail_closed="tool_invocation_error; TOOL_DENIED; no fabricated RuntimeEvent on trace failure",
    ),
    P3OwnershipRow(
        concern="governed provider boundary evidence",
        semantic_owner="ExecutionBoundaryEvent (governed_execution_boundary_event.v1)",
        producer="compose_execution_boundary_event / governed external work host",
        persistence="execution evidence persistence ports",
        reader="attestation / proof consumers (non-execution-authority)",
        execution_identity="task_id + run_id only (no AttemptId/ExecutionId fields)",
        authorization_identity="PolicyDecisionSection + optional GovernanceEvidenceSection pointer",
        tenant_scope="tenant_id optional on event",
        fail_closed="compose requires ALLOW proof + policy bundle refs",
    ),
    P3OwnershipRow(
        concern="harness attestation export",
        semantic_owner="ExecutionBoundaryEventV1",
        producer="ExecutionBoundaryEmitter",
        persistence="BoundaryEventBuffer (export projection)",
        reader="external receipt adapters",
        execution_identity="task_id + run_id + step_id (not canonical ExecutionId)",
        authorization_identity="HarnessBoundaryPolicyVerdictV1 (export-only)",
        tenant_scope="tenant_id on export event",
        fail_closed="unsigned; explicitly non-authoritative",
    ),
    P3OwnershipRow(
        concern="governance decision evidence",
        semantic_owner="GovernanceDecisionEvidenceFact",
        producer="GovernanceEvidenceRecorder",
        persistence="GovernanceEvidencePersistence",
        reader="audit / reconstruction consumers",
        execution_identity="optional AttemptId + ExecutionId; has_full_execution_correlation",
        authorization_identity="evidence_id + decision + request_digest",
        tenant_scope="tenant_id required",
        fail_closed="persistence may reject incomplete correlation per policy",
    ),
    P3OwnershipRow(
        concern="meaningful side-effect authorization",
        semantic_owner="Governance (MeaningfulSideEffectAuthorizationPort)",
        producer="RuntimeToolInvoker / orchestration MSE composition",
        persistence="decision material + GovernanceDecisionEvidenceFact projection",
        reader="tool invoker pre-external-effect gates",
        execution_identity="GovernedContinuationCorrelation (typed execution ids)",
        authorization_identity="fresh authorize() per invocation scope",
        tenant_scope="tenant on governance context",
        fail_closed="DENY prevents ToolExecutor entry; test_fresh_side_effect_authorization",
    ),
)

P3_IN_SCOPE_BLOCKERS: Final[tuple[P3InScopeBlocker, ...]] = (
    P3InScopeBlocker(
        blocker_id="P3-B04-01",
        classification=P3BlockerClassification.IN_SCOPE_BLOCKER,
        frz_id="FRZ-TRC-04",
        summary=(
            "Governed ExecutionBoundaryEvent and ProviderInvocationSection lack canonical "
            "AttemptId/ExecutionId; multi-execution runs cannot join provider invocation "
            "to exact execution without heuristic."
        ),
        remediation=PROPOSED_CHILD_ON_BLOCKER,
    ),
    P3InScopeBlocker(
        blocker_id="P3-B06-01",
        classification=P3BlockerClassification.IN_SCOPE_BLOCKER,
        frz_id="FRZ-TRC-06",
        summary=(
            "Side-effect→authorization exact chain requires execution-bound governance evidence "
            "linked to governed boundary/provider facts; boundary event cannot express "
            "execution-level effect attribution without contract evolution."
        ),
        remediation=PROPOSED_CHILD_ON_BLOCKER,
    ),
)

FRZ_TRC_03_DISPOSITION: Final[FrzTrcP3Disposition] = (
    FrzTrcP3Disposition.READY_FOR_INDEPENDENT_CLOSURE_REVIEW
)
FRZ_TRC_04_DISPOSITION: Final[FrzTrcP3Disposition] = FrzTrcP3Disposition.BLOCKED
FRZ_TRC_06_DISPOSITION: Final[FrzTrcP3Disposition] = FrzTrcP3Disposition.BLOCKED

P3_READINESS: Final[P3ReadinessStatus] = P3ReadinessStatus.BLOCKED

ADVERSARIAL_CASES: Final[tuple[P3AdversarialCase, ...]] = (
    P3AdversarialCase("T1", P3Concern.TOOL_INVOCATION, "Tool under E1", "RuntimeEvent.execution_id == E1", "test_txp3_adversarial_t1_tool_reconstructs_exact_execution"),
    P3AdversarialCase("T2", P3Concern.TOOL_INVOCATION, "E1/E2 same tool", "distinct ExecutionId on TOOL_REQUESTED", "test_txp3_adversarial_t2_multi_execution_tool_separation"),
    P3AdversarialCase("T3", P3Concern.TOOL_INVOCATION, "Same tool_id across attempts", "attempt partition separates facts", "test_txp3_adversarial_t3_multi_attempt_separation"),
    P3AdversarialCase("T4", P3Concern.TOOL_INVOCATION, "Cross-tenant", "scope validation / isolation", "test_reconstruction_groups_by_attempt"),
    P3AdversarialCase("T5", P3Concern.TOOL_INVOCATION, "Trace start failure", "non-blocking; no fabricated canonical attribution", "test_txp3_q22_optional_observability_not_authority"),
    P3AdversarialCase("P1", P3Concern.PROVIDER_INVOCATION, "Two executions same run", "cannot prove separation on boundary", "test_txp3_q11_provider_execution_id_blocker"),
    P3AdversarialCase("P2", P3Concern.PROVIDER_INVOCATION, "Same provider across attempts", "no heuristic merge", "test_txp3_q14_no_heuristic_provider_join"),
    P3AdversarialCase("P3", P3Concern.PROVIDER_INVOCATION, "No execution relation", "unattributable / incomplete", "test_txp3_q11_provider_execution_id_blocker"),
    P3AdversarialCase("P4", P3Concern.PROVIDER_INVOCATION, "Cross-tenant", "fail closed on scope", "test_corrupted_runtime_persistence_fails_closed"),
    P3AdversarialCase("P5", P3Concern.PROVIDER_INVOCATION, "invocation_id contract", "ProviderInvocationSection requires invocation_id", "test_txp3_q04_governed_boundary_contract_fields"),
    P3AdversarialCase("S1", P3Concern.SIDE_EFFECT_AUTHORIZATION, "ALLOW exact execution", "fresh authorize + evidence when populated", "test_fresh_side_effect_authorization"),
    P3AdversarialCase("S2", P3Concern.SIDE_EFFECT_AUTHORIZATION, "DENY", "no ToolExecutor", "test_fresh_side_effect_authorization"),
    P3AdversarialCase("S3", P3Concern.SIDE_EFFECT_AUTHORIZATION, "ALLOW E1 for E2", "rejected", "test_gr10_r11_r4_runtime_tool_invoker_exact_correlation"),
    P3AdversarialCase("S4", P3Concern.SIDE_EFFECT_AUTHORIZATION, "cross-agent", "rejected", "test_fresh_side_effect_authorization"),
    P3AdversarialCase("S5", P3Concern.SIDE_EFFECT_AUTHORIZATION, "wrong scope", "rejected", "test_gr10_r11_r4_runtime_tool_invoker_exact_correlation"),
    P3AdversarialCase("S6", P3Concern.SIDE_EFFECT_AUTHORIZATION, "DENY after ALLOW", "DENY wins", "test_fresh_side_effect_authorization"),
    P3AdversarialCase("S7", P3Concern.SIDE_EFFECT_AUTHORIZATION, "recorder unavailable", "governance decision ≠ recorder", "test_txp3_q25_evidence_recorder_non_authoritative"),
    P3AdversarialCase("S8", P3Concern.SIDE_EFFECT_AUTHORIZATION, "effect without auth evidence", "FRZ-TRC-06 blocked", "test_txp3_q21_effect_authorization_exact_or_blocker"),
)

TENANT_ISOLATION_AUDIT_P3: Final[dict[str, str]] = {
    "tenant_scope_applicable": "YES",
    "canonical_tenant_identity": "tenant_id",
    "tenant_owner": "Task / RuntimeEvent / GovernanceDecisionEvidenceFact",
    "tool_propagation": "Task.tenant_id → trace bridge → RuntimeEvent.tenant_id",
    "provider_propagation": "ExecutionBoundaryEvent.tenant_id; GovernedProofProfile.tenant_id",
    "authorization_propagation": "GovernanceDecisionEvidenceFact.tenant_id required",
    "cross_tenant_negatives": "runtime/causal persistence scope validation (P2)",
    "fail_closed": "scope validators on reconstruction reads",
    "result": "PASS",
}

ENTERPRISE_AUDIT_MATRIX_P3: Final[tuple[P3EnterpriseAuditRow, ...]] = (
    P3EnterpriseAuditRow("tool invocation exact execution attribution", P3GateResult.PASS),
    P3EnterpriseAuditRow("provider invocation exact execution attribution", P3GateResult.BLOCKED),
    P3EnterpriseAuditRow("side effect exact authorization attribution", P3GateResult.BLOCKED),
    P3EnterpriseAuditRow("multi-execution isolation", P3GateResult.PASS),
    P3EnterpriseAuditRow("multi-attempt isolation", P3GateResult.PASS),
    P3EnterpriseAuditRow("provider retry semantics", P3GateResult.BLOCKED),
    P3EnterpriseAuditRow("effect certainty", P3GateResult.PASS),
    P3EnterpriseAuditRow("stale authorization rejection", P3GateResult.PASS),
    P3EnterpriseAuditRow("cross-agent rejection", P3GateResult.PASS),
    P3EnterpriseAuditRow("invocation scope binding", P3GateResult.PASS),
    P3EnterpriseAuditRow("tenant isolation", P3GateResult.PASS),
    P3EnterpriseAuditRow("exactly-one evidence owners", P3GateResult.PASS),
    P3EnterpriseAuditRow("no attestation authority promotion", P3GateResult.PASS),
    P3EnterpriseAuditRow("Governance ≠ Execution", P3GateResult.PASS),
    P3EnterpriseAuditRow("evidence non-authoritative", P3GateResult.PASS),
    P3EnterpriseAuditRow("no heuristic joins", P3GateResult.BLOCKED),
    P3EnterpriseAuditRow("contracts over implementations", P3GateResult.PASS),
    P3EnterpriseAuditRow("strong typing", P3GateResult.PASS),
    P3EnterpriseAuditRow("pluginability/replaceability", P3GateResult.PASS),
    P3EnterpriseAuditRow("FRZ-TRC-03 readiness", P3GateResult.PASS),
    P3EnterpriseAuditRow("FRZ-TRC-04 readiness", P3GateResult.BLOCKED),
    P3EnterpriseAuditRow("FRZ-TRC-06 readiness", P3GateResult.BLOCKED),
)

ATTRIBUTION_CHAIN_TOOL: Final[str] = (
    "ExecutionId (active identity) → RuntimeToolInvoker.trace_event(tool_invocation_*) "
    "→ trace_event_to_runtime_event → RuntimeEvent (TOOL_*) → ExecutionReconstructor"
)

ATTRIBUTION_CHAIN_PROVIDER_BLOCKED: Final[str] = (
    "ExecutionId → ??? — governed ExecutionBoundaryEvent stops at task_id/run_id; "
    "STOP — ARCHITECTURE DECISION REQUIRED (TRACE-X-P3-R1)"
)

ATTRIBUTION_CHAIN_EFFECT_BLOCKED: Final[str] = (
    "ExecutionId → tool/MSE authorization → GovernanceDecisionEvidenceFact (when populated) "
    "→ governed boundary — cannot prove exact effect↔decision↔provider without execution on boundary"
)


def governed_boundary_model_text() -> str:
    return (repo_root() / GOVERNED_BOUNDARY_MODULE).read_text(encoding="utf-8")


def attestation_boundary_model_text() -> str:
    return (repo_root() / ATTESTATION_BOUNDARY_MODULE).read_text(encoding="utf-8")


def assert_single_reconstruction_owner() -> None:
    from tests.qualification.trace_x._trace_x_p2_support import assert_single_reconstruction_owner as _p2

    _p2()
