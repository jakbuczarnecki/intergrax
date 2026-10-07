# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P2 typed execution causal reconstruction inventory SSOT (qualification-local)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.trace_x._trace_x_p0_support import repo_root

TRACE_X_P2_START_HEAD: Final[str] = "f85ec3697742d41a9beb476e7cb9cc0aeae956d4"

MANDATORY_FRZ_P2_IDS: Final[tuple[str, ...]] = ("FRZ-TRC-01",)

OUT_OF_SCOPE_FRZ: Final[tuple[str, ...]] = (
    "FRZ-TRC-02",
    "FRZ-TRC-03",
    "FRZ-TRC-04",
    "FRZ-TRC-05",
    "FRZ-TRC-06",
    "FRZ-TRC-07",
    "FRZ-TRC-08",
    "FRZ-TRC-09",
    "FRZ-TRC-10",
    "FRZ-TRC-11",
    "FRZ-TRC-12",
)


class P2GateResult(StrEnum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"


class P2ReadinessStatus(StrEnum):
    READY_FOR_AUDIT = "READY FOR AUDIT"
    BLOCKED = "BLOCKED"


class FrzTrcP2Disposition(StrEnum):
    READY_FOR_INDEPENDENT_CLOSURE_REVIEW = "READY FOR INDEPENDENT CLOSURE REVIEW"
    BLOCKED = "BLOCKED"


class P2BlockerClassification(StrEnum):
    IN_SCOPE_BLOCKER = "IN-SCOPE BLOCKER"
    TRACKED_FREEZE_DEBT = "TRACKED FREEZE DEBT"
    ENVIRONMENT_TEST_ISSUE = "ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED"


class P2AdversarialCaseId(StrEnum):
    CASE_A = "A"
    CASE_B = "B"
    CASE_C = "C"
    CASE_D = "D"
    CASE_E = "E"
    CASE_F = "F"


@dataclass(frozen=True, slots=True)
class P2OwnershipRow:
    concern: str
    semantic_owner: str
    writer: str
    persistence: str
    reader: str


@dataclass(frozen=True, slots=True)
class P2EvidenceRow:
    evidence_id: str
    summary: str
    evidence_tests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class P2AdversarialCase:
    case_id: P2AdversarialCaseId
    description: str
    expected: str
    evidence_test: str


@dataclass(frozen=True, slots=True)
class P2EnterpriseAuditRow:
    area: str
    result: P2GateResult


@dataclass(frozen=True, slots=True)
class P2InScopeBlocker:
    blocker_id: str
    classification: P2BlockerClassification
    summary: str


OWNERSHIP_MATRIX: Final[tuple[P2OwnershipRow, ...]] = (
    P2OwnershipRow(
        concern="runtime execution event facts",
        semantic_owner="RuntimeEvent",
        writer="runtime event recorders",
        persistence="EvidencePersistencePort",
        reader="ExecutionReconstructor",
    ),
    P2OwnershipRow(
        concern="transport→runtime causal relation",
        semantic_owner="PlatformCausalEvidence",
        writer="admit_background_execution_handler",
        persistence="CausalEvidencePersistence",
        reader="ExecutionReconstructor",
    ),
    P2OwnershipRow(
        concern="parent→child execution topology",
        semantic_owner="ExecutionLineage",
        writer="lineage admission hooks",
        persistence="ExecutionLineagePersistence",
        reader="ExecutionLineageReader / ExecutionReconstructor",
    ),
    P2OwnershipRow(
        concern="derived factual reconstruction",
        semantic_owner="ExecutionReconstructor",
        writer="n/a (derived read)",
        persistence="none (non-persisted)",
        reader="ExecutionReconstructionReader consumers",
    ),
)

RECONSTRUCTION_CONSUMERS: Final[tuple[str, ...]] = (
    "intergrax/runtime/diagnostics/diagnostic_orchestrator.py",
    "intergrax/runtime/diagnostics/diagnostic_read_service.py",
    "intergrax/runtime/diagnostics/execution_reconstruction_read_session.py",
    "intergrax/runtime/observability/historical_reconstruction.py",
    "intergrax/runtime/runtime_inspection/adapters/execution_reconstruction.py",
)

CANONICAL_CONTRACTS: Final[tuple[str, ...]] = (
    "intergrax/contracts/execution_evidence/persistence_port.py",
    "intergrax/contracts/platform_causal_evidence.py",
    "intergrax/contracts/execution_lineage.py",
    "intergrax/contracts/execution_reconstruction.py",
    "intergrax/contracts/execution_reconstruction_models.py",
    "intergrax/contracts/execution_reconstruction_lineage.py",
)

P2_EVIDENCE_MATRIX: Final[tuple[P2EvidenceRow, ...]] = (
    P2EvidenceRow(
        "P2-E01",
        "transport → attempt → execution via PlatformCausalEvidence.target",
        ("test_txp2_adversarial_case_a_causal_in_lineage", "test_txp1_q04_runtime_execution_ref_complete_typed_identity"),
    ),
    P2EvidenceRow(
        "P2-E02",
        "RuntimeEvent retains tenant/task/run/attempt/execution identity",
        ("test_reconstruction_groups_by_attempt", "test_txp2_q09_runtime_execution_id_retained"),
    ),
    P2EvidenceRow(
        "P2-E03",
        "lineage parent topology for covered executions",
        ("test_single_segment_parent_chain", "test_txp2_q10_lineage_parent_topology_retained"),
    ),
    P2EvidenceRow(
        "P2-E04",
        "facts attached to correct AttemptId",
        ("test_reconstruction_groups_by_attempt",),
    ),
    P2EvidenceRow(
        "P2-E05",
        "multi-attempt retry separation",
        ("test_worker_retry_mints_new_attempt_and_new_evidence", "test_txp2_q13_retry_multi_attempt_separation"),
    ),
    P2EvidenceRow(
        "P2-E06",
        "nested/fan-out child topology",
        ("test_nested_fan_out_parents",),
    ),
    P2EvidenceRow(
        "P2-E07",
        "multi-segment resume continuity",
        ("test_multi_segment_resume_topology",),
    ),
    P2EvidenceRow(
        "P2-E08",
        "execution-position ordering (not timestamp)",
        ("test_ordering_follows_execution_position_not_timestamp",),
    ),
    P2EvidenceRow(
        "P2-E09",
        "cross-scope / contradictory durable facts fail closed",
        (
            "test_corrupted_causal_persistence_fails_closed",
            "test_txp2_adversarial_case_b_causal_outside_lineage",
            "test_txp2_adversarial_case_c_runtime_contradicts_lineage",
        ),
    ),
    P2EvidenceRow(
        "P2-E10",
        "incomplete evidence never promoted to COMPLETE",
        (
            "test_degraded_attempt_is_partial",
            "test_truncated_when_max_records_exceeded",
            "test_txp2_adversarial_case_d_partial_not_corruption",
            "test_txp2_adversarial_case_e_lineage_unavailable",
            "test_txp2_adversarial_case_f_runtime_truncated",
        ),
    ),
)

ADVERSARIAL_CASES: Final[tuple[P2AdversarialCase, ...]] = (
    P2AdversarialCase(
        P2AdversarialCaseId.CASE_A,
        "Causal target execution belongs to reconstructed lineage",
        "PASS",
        "test_txp2_adversarial_case_a_causal_in_lineage",
    ),
    P2AdversarialCase(
        P2AdversarialCaseId.CASE_B,
        "Complete lineage but causal ExecutionId outside topology",
        "ExecutionReconstructionIntegrityError",
        "test_txp2_adversarial_case_b_causal_outside_lineage",
    ),
    P2AdversarialCase(
        P2AdversarialCaseId.CASE_C,
        "Runtime event ExecutionId contradicts complete lineage",
        "ExecutionReconstructionIntegrityError",
        "test_txp2_adversarial_case_c_runtime_contradicts_lineage",
    ),
    P2AdversarialCase(
        P2AdversarialCaseId.CASE_D,
        "PARTIAL lineage — missing membership not auto-corruption",
        "truthful incomplete reconstruction",
        "test_txp2_adversarial_case_d_partial_not_corruption",
    ),
    P2AdversarialCase(
        P2AdversarialCaseId.CASE_E,
        "Lineage backend unavailable",
        "read_status=UNAVAILABLE; no fabricated chain",
        "test_txp2_adversarial_case_e_lineage_unavailable",
    ),
    P2AdversarialCase(
        P2AdversarialCaseId.CASE_F,
        "Runtime history truncated",
        "RuntimeHistoryCompleteness.TRUNCATED",
        "test_txp2_adversarial_case_f_runtime_truncated",
    ),
)

P2_IN_SCOPE_BLOCKERS: Final[tuple[P2InScopeBlocker, ...]] = ()

TENANT_ISOLATION_AUDIT_P2: Final[dict[str, str]] = {
    "tenant_scope_applicable": "YES",
    "reconstruction_scope": "tenant_id + TaskId + RunId on read",
    "cross_tenant_path": "test_corrupted_causal_persistence_fails_closed; test_corrupted_runtime_persistence_fails_closed",
    "result": "PASS",
}

FRZ_TRC_01_DISPOSITION: Final[FrzTrcP2Disposition] = (
    FrzTrcP2Disposition.READY_FOR_INDEPENDENT_CLOSURE_REVIEW
)

P2_READINESS: Final[P2ReadinessStatus] = P2ReadinessStatus.READY_FOR_AUDIT

ENTERPRISE_AUDIT_MATRIX_P2: Final[tuple[P2EnterpriseAuditRow, ...]] = (
    P2EnterpriseAuditRow("exactly-one reconstruction owner", P2GateResult.PASS),
    P2EnterpriseAuditRow("contracts over implementations", P2GateResult.PASS),
    P2EnterpriseAuditRow("strong typing", P2GateResult.PASS),
    P2EnterpriseAuditRow("provider replaceability", P2GateResult.PASS),
    P2EnterpriseAuditRow("tenant isolation", P2GateResult.PASS),
    P2EnterpriseAuditRow("runtime scope integrity", P2GateResult.PASS),
    P2EnterpriseAuditRow("causal scope integrity", P2GateResult.PASS),
    P2EnterpriseAuditRow("lineage scope integrity", P2GateResult.PASS),
    P2EnterpriseAuditRow("cross-source ExecutionId coherence", P2GateResult.PASS),
    P2EnterpriseAuditRow("attempt isolation", P2GateResult.PASS),
    P2EnterpriseAuditRow("parent-child topology", P2GateResult.PASS),
    P2EnterpriseAuditRow("multi-segment continuity", P2GateResult.PASS),
    P2EnterpriseAuditRow("deterministic ordering", P2GateResult.PASS),
    P2EnterpriseAuditRow("as-of isolation", P2GateResult.PASS),
    P2EnterpriseAuditRow("partial/truncated truthfulness", P2GateResult.PASS),
    P2EnterpriseAuditRow("backend unavailable truthfulness", P2GateResult.PASS),
    P2EnterpriseAuditRow("corruption fail-closed", P2GateResult.PASS),
    P2EnterpriseAuditRow("no heuristic reconstruction", P2GateResult.PASS),
    P2EnterpriseAuditRow("no reconstruction persistence", P2GateResult.PASS),
    P2EnterpriseAuditRow("Governance ≠ reconstruction", P2GateResult.PASS),
    P2EnterpriseAuditRow("Observability/Diagnostics non-authoritative", P2GateResult.PASS),
    P2EnterpriseAuditRow("regression protection", P2GateResult.PASS),
    P2EnterpriseAuditRow("FRZ-TRC-01 readiness", P2GateResult.PASS),
)

RECONSTRUCTION_CHAIN: Final[str] = (
    "MessageBusTaskRef → PlatformCausalEvidence.target (RuntimeExecutionRef) → "
    "AttemptId/ExecutionId → RuntimeEvent history → ExecutionLineage topology → "
    "ExecutionReconstruction"
)


def assert_single_reconstruction_owner() -> None:
    from tests.qualification.trace_x._trace_x_p0_support import discover_sensitive_classes

    recon_impls = discover_sensitive_classes().get("ExecutionReconstructor", [])
    assert recon_impls == [
        "intergrax/runtime/observability/reconstruction/execution_reconstruction.py"
    ]


def reconstruction_module_text() -> str:
    return (
        repo_root()
        / "intergrax/runtime/observability/reconstruction/execution_reconstruction.py"
    ).read_text(encoding="utf-8")
