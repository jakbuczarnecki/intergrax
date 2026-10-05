# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P1 typed transport/lineage adversarial inventory SSOT (qualification-local)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from tests.qualification.trace_x._trace_x_p0_support import repo_root

TRACE_X_P1_START_HEAD: Final[str] = "19eda895d7db3c11758d77aaf8a96d746fa7c304"

MANDATORY_FRZ_P1_IDS: Final[tuple[str, ...]] = ("FRZ-TRC-02", "FRZ-TRC-12")

PROPOSED_CHILD_ON_BLOCKER: Final[str] = "TRACE-X-P1-R1"


class P1Concern(StrEnum):
    TRANSPORT_RUNTIME_MAPPING = "transport_runtime_mapping"
    PARENT_CHILD_CAUSALITY = "parent_child_causality"


class P1CoverageStatus(StrEnum):
    PROVEN = "proven"
    BLOCKED = "blocked"
    PARTIAL = "partial"


class P1GateResult(StrEnum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"


class P1BlockerClassification(StrEnum):
    IN_SCOPE_BLOCKER = "IN-SCOPE BLOCKER"
    TRACKED_FREEZE_DEBT = "TRACKED FREEZE DEBT"
    ENVIRONMENT_TEST_ISSUE = "ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED"


class P1ReadinessStatus(StrEnum):
    READY_FOR_AUDIT = "READY FOR AUDIT"
    BLOCKED = "BLOCKED"


class FrzTrcP1Disposition(StrEnum):
    READY_FOR_INDEPENDENT_CLOSURE_REVIEW = "READY FOR INDEPENDENT CLOSURE REVIEW"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class P1Path:
    path_id: str
    concern: P1Concern
    semantic_owner: str
    contract: str
    producer: str
    persistence: str
    reader: str | None
    tenant_scope: str
    fail_closed_semantics: str
    evidence_tests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class P1TransportEntrypoint:
    entrypoint_id: str
    module_path: str
    producer_symbol: str
    admission_gate: str
    evidence_before_handler: bool


@dataclass(frozen=True, slots=True)
class P1LineagePath:
    lineage_path_id: str
    parent_source: str
    child_identity_source: str
    scope: str
    durability: str
    expected_reconstruction: str
    failure_behavior: str
    evidence_tests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class P1InScopeBlocker:
    blocker_id: str
    classification: P1BlockerClassification
    summary: str
    architecture_decision_required: str


@dataclass(frozen=True, slots=True)
class P1OwnershipRow:
    concern: str
    semantic_owner: str
    writer: str
    persistence: str
    reader: str


@dataclass(frozen=True, slots=True)
class P1EnterpriseAuditRow:
    area: str
    result: P1GateResult


@dataclass(frozen=True, slots=True)
class P1DegradedLineageCase:
    case_id: str
    description: str
    evidence_tests: tuple[str, ...]


OWNERSHIP_MATRIX: Final[tuple[P1OwnershipRow, ...]] = (
    P1OwnershipRow(
        concern="transport→runtime relation",
        semantic_owner="PlatformCausalEvidence",
        writer="admit_background_execution_handler / build_transport_triggered_execution_evidence",
        persistence="CausalEvidencePersistence",
        reader="ExecutionReconstructor / causal evidence queries",
    ),
    P1OwnershipRow(
        concern="parent→child topology",
        semantic_owner="ExecutionLineage",
        writer="ExecutionLineageChildAdmissionHook / ExecutionLineageRootAdmissionHook",
        persistence="ExecutionLineagePersistence",
        reader="ExecutionLineageReader / ExecutionReconstructor",
    ),
)

TRANSPORT_ENTRYPOINTS: Final[tuple[P1TransportEntrypoint, ...]] = (
    P1TransportEntrypoint(
        entrypoint_id="TXP1-T01",
        module_path="intergrax/queueing/worker/dispatcher.py",
        producer_symbol="Celery dispatcher worker path",
        admission_gate="admit_background_execution_handler",
        evidence_before_handler=True,
    ),
    P1TransportEntrypoint(
        entrypoint_id="TXP1-T02",
        module_path="intergrax/background_tasks/worker_runtime.py",
        producer_symbol="WorkerRuntime",
        admission_gate="admit_background_execution_handler",
        evidence_before_handler=True,
    ),
    P1TransportEntrypoint(
        entrypoint_id="TXP1-T03",
        module_path="intergrax/queueing/providers/broker_worker_base.py",
        producer_symbol="BrokerWorkerBase",
        admission_gate="admit_background_execution_handler",
        evidence_before_handler=True,
    ),
    P1TransportEntrypoint(
        entrypoint_id="TXP1-T04",
        module_path="intergrax/queueing/providers/document_store/colocated_worker.py",
        producer_symbol="DocumentStoreTaskWorker",
        admission_gate="admit_background_execution_handler",
        evidence_before_handler=True,
    ),
)

LINEAGE_PATHS: Final[tuple[P1LineagePath, ...]] = (
    P1LineagePath(
        lineage_path_id="TXP1-L01",
        parent_source="none (root)",
        child_identity_source="mint_root_execution_identity",
        scope="ExecutionLineageAttemptScope tenant/task/run/attempt",
        durability="durable when backend available",
        expected_reconstruction="parent_execution_id=None; execution_id==segment_root",
        failure_behavior="ExecutionLineageUnavailableError on root → delegate blocked (admission order test)",
        evidence_tests=("test_root_admission_before_delegate",),
    ),
    P1LineagePath(
        lineage_path_id="TXP1-L02",
        parent_source="require_active_execution_id()",
        child_identity_source="mint_child_execution_id via ChildExecutionRunner",
        scope="active attempt scope",
        durability="durable admit_child before delegate",
        expected_reconstruction="exact parent_execution_id edge in ExecutionLineage",
        failure_behavior="ExecutionLineageIntegrityError → child does not execute",
        evidence_tests=(
            "test_root_and_child_admissions",
            "test_child_lineage_hook_auto_attached",
            "test_conflicting_parent_fails_closed",
        ),
    ),
    P1LineagePath(
        lineage_path_id="TXP1-L03",
        parent_source="same durable root",
        child_identity_source="ChildExecutionRunner siblings",
        scope="same attempt",
        durability="durable per successful admission",
        expected_reconstruction="multiple child edges from shared parent",
        failure_behavior="conflicting parent fails closed; failed admission blocks delegate",
        evidence_tests=(
            "test_sibling_after_failed_child_admission_may_execute_when_durable",
        ),
    ),
    P1LineagePath(
        lineage_path_id="TXP1-L05",
        parent_source="prior attempt segment",
        child_identity_source="new attempt mint",
        scope="per AttemptId",
        durability="segment scoped",
        expected_reconstruction="attempt-isolated lineage prefix",
        failure_behavior="cross-attempt scope mismatch → integrity error",
        evidence_tests=("test_worker_retry_mints_new_attempt_and_new_evidence",),
    ),
    P1LineagePath(
        lineage_path_id="TXP1-L06",
        parent_source="predecessor segment root",
        child_identity_source="resume segment admission",
        scope="ExecutionLineageAttemptScope",
        durability="durable when resume clean",
        expected_reconstruction="segment list with predecessor link",
        failure_behavior="unclean resume marks degraded",
        evidence_tests=("test_resume_segment_with_predecessor", "test_unclean_resume_marks_degraded"),
    ),
    P1LineagePath(
        lineage_path_id="TXP1-L07",
        parent_source="active parent ExecutionId",
        child_identity_source="ChildExecutionRunner on ExecutionLineageUnavailableError",
        scope="same attempt",
        durability="durable admission required before delegate; attempt may be degraded",
        expected_reconstruction="PARTIAL completeness; no fabricated parent edge for failed admission",
        failure_behavior="delegate blocked; later sibling may run after its own durable admission",
        evidence_tests=(
            "test_child_admission_unavailable_blocks_delegate_and_marks_degraded",
            "test_sibling_after_failed_child_admission_may_execute_when_durable",
            "test_failed_child_admission_reconstruction_honest",
            "test_degraded_attempt_is_partial",
        ),
    ),
)

DEGRADED_LINEAGE_CASES: Final[tuple[P1DegradedLineageCase, ...]] = (
    P1DegradedLineageCase("Case A", "durable child admission succeeds", ("test_root_and_child_admissions",)),
    P1DegradedLineageCase(
        "Case B",
        "integrity conflict → child blocked",
        ("test_conflicting_parent_fails_closed",),
    ),
    P1DegradedLineageCase(
        "Case C",
        "child admission backend unavailable → delegate blocked; attempt degraded",
        ("test_child_admission_unavailable_blocks_delegate_and_marks_degraded",),
    ),
    P1DegradedLineageCase(
        "Case D",
        "mark_degraded unavailable → delegate still blocked",
        ("test_mark_degraded_unavailable_still_blocks_delegate",),
    ),
    P1DegradedLineageCase(
        "Case E",
        "sibling from durable root after failed child admission may execute",
        ("test_sibling_after_failed_child_admission_may_execute_when_durable",),
    ),
    P1DegradedLineageCase(
        "Case F",
        "reconstruction reports PARTIAL/degraded; no invented edge",
        ("test_degraded_attempt_is_partial",),
    ),
)

P1_TRANSPORT_PATHS: Final[tuple[P1Path, ...]] = (
    P1Path(
        path_id="TXP1-P-T01",
        concern=P1Concern.TRANSPORT_RUNTIME_MAPPING,
        semantic_owner="PlatformCausalEvidence",
        contract="intergrax/contracts/platform_causal_evidence.py",
        producer="intergrax/runtime/background_execution/required_audit_evidence.py",
        persistence="CausalEvidencePersistence",
        reader="ExecutionReconstructor",
        tenant_scope="evidence.tenant_id == source.tenant_id == target.tenant_id",
        fail_closed_semantics="tenant mismatch / missing ExecutionId / persistence failure blocks handler",
        evidence_tests=(
            "test_tenant_mismatch_fails_closed",
            "test_required_evidence_backend_failure_blocks_handler_and_wraps_cause",
            "test_broker_worker_path_persists_required_causal_evidence",
        ),
    ),
    P1Path(
        path_id="TXP1-P-T02",
        concern=P1Concern.PARENT_CHILD_CAUSALITY,
        semantic_owner="ExecutionLineage",
        contract="intergrax/contracts/execution_lineage.py",
        producer="intergrax/runtime/execution/lineage/admission.py",
        persistence="ExecutionLineagePersistence",
        reader="ExecutionLineageReader",
        tenant_scope="ExecutionLineageAttemptScope.tenant_id",
        fail_closed_semantics="scope/integrity mismatch → ExecutionLineageIntegrityError",
        evidence_tests=(
            "test_conflicting_parent_fails_closed",
            "test_duplicate_identical_child_is_idempotent",
        ),
    ),
)

P1_IN_SCOPE_BLOCKERS: Final[tuple[P1InScopeBlocker, ...]] = ()

P1_RESOLVED_BLOCKERS: Final[tuple[tuple[str, str], ...]] = (
    (
        "P1-BLK-DEGRADED-LINEAGE-01",
        "RESOLVED PENDING INDEPENDENT AUDIT (TRACE-X-P1-R1 strict durable child admission)",
    ),
)

TENANT_ISOLATION_AUDIT_P1: Final[dict[str, str]] = {
    "tenant_scope_applicable": "YES",
    "canonical_tenant_identity": "tenant_id",
    "tenant_owner": "RuntimeExecutionRef / PlatformCausalEvidence / ExecutionLineageAttemptScope",
    "transport_propagation": "MessageBusTaskRef → PlatformCausalEvidence → RuntimeExecutionRef",
    "lineage_propagation": "ExecutionLineageAttemptScope",
    "state_isolation": "supporting STATE-X evidence only",
    "provider_config_isolation": "not P1 closure",
    "evidence_trace_isolation": "P1 direct scope",
    "async_recovery_continuity": "supporting only; TRACE-X-P6",
    "cross_tenant_path": (
        "test_tenant_mismatch_fails_closed; test_list_segments_tenant_isolation; "
        "test_tenant_isolation_for_same_transport_id"
    ),
    "fail_closed_behavior": (
        "PlatformCausalEvidence tenant validator; lineage scope tenant on read/admit"
    ),
    "adversarial_evidence": (
        "test_tenant_mismatch_fails_closed, test_list_segments_tenant_isolation, "
        "test_terminal_store_tenant_isolation"
    ),
    "result": "PASS",
}

FRZ_TRC_12_DISPOSITION: Final[FrzTrcP1Disposition] = (
    FrzTrcP1Disposition.READY_FOR_INDEPENDENT_CLOSURE_REVIEW
)

FRZ_TRC_02_DISPOSITION: Final[FrzTrcP1Disposition] = (
    FrzTrcP1Disposition.READY_FOR_INDEPENDENT_CLOSURE_REVIEW
)

P1_READINESS: Final[P1ReadinessStatus] = P1ReadinessStatus.READY_FOR_AUDIT

EXECUTED_CHILD_WITHOUT_DURABLE_PARENT_EDGE: Final[bool] = False

ENTERPRISE_AUDIT_MATRIX_P1: Final[tuple[P1EnterpriseAuditRow, ...]] = (
    P1EnterpriseAuditRow("transport/runtime identity domain separation", P1GateResult.PASS),
    P1EnterpriseAuditRow("full runtime target identity", P1GateResult.PASS),
    P1EnterpriseAuditRow("causal tenant binding", P1GateResult.PASS),
    P1EnterpriseAuditRow("all transport entrypoints covered", P1GateResult.PASS),
    P1EnterpriseAuditRow("no transport bypass", P1GateResult.PASS),
    P1EnterpriseAuditRow("evidence-before-handler", P1GateResult.PASS),
    P1EnterpriseAuditRow("evidence persistence failure fail-closed", P1GateResult.PASS),
    P1EnterpriseAuditRow("retry/redelivery semantics", P1GateResult.PASS),
    P1EnterpriseAuditRow("legacy evidence safety", P1GateResult.PASS),
    P1EnterpriseAuditRow("lineage exactly-one owner", P1GateResult.PASS),
    P1EnterpriseAuditRow("child hook sanctioned path", P1GateResult.PASS),
    P1EnterpriseAuditRow("admission-before-delegate", P1GateResult.PASS),
    P1EnterpriseAuditRow("child/parent identity correctness", P1GateResult.PASS),
    P1EnterpriseAuditRow("lineage scope integrity", P1GateResult.PASS),
    P1EnterpriseAuditRow("conflicting parent rejection", P1GateResult.PASS),
    P1EnterpriseAuditRow("reconstruction non-heuristic", P1GateResult.PASS),
    P1EnterpriseAuditRow("corrupt lineage fail-closed", P1GateResult.PASS),
    P1EnterpriseAuditRow("missing lineage not fabricated", P1GateResult.PASS),
    P1EnterpriseAuditRow("degraded lineage semantics", P1GateResult.PASS),
    P1EnterpriseAuditRow("strict child admission on lineage unavailable", P1GateResult.PASS),
    P1EnterpriseAuditRow("contracts over implementations", P1GateResult.PASS),
    P1EnterpriseAuditRow("strong typing", P1GateResult.PASS),
    P1EnterpriseAuditRow("pluginability / replaceability", P1GateResult.PASS),
    P1EnterpriseAuditRow("Governance/Execution separation", P1GateResult.PASS),
    P1EnterpriseAuditRow("tenant isolation P1", P1GateResult.PASS),
    P1EnterpriseAuditRow("FRZ-TRC-02 readiness", P1GateResult.PASS),
    P1EnterpriseAuditRow("FRZ-TRC-12 readiness", P1GateResult.PASS),
)

_PRODUCTION_SCAN_ROOTS: Final[tuple[str, ...]] = ("intergrax",)

_ADMISSION_GATE = "admit_background_execution_handler"


def _python_files_under(rel: str) -> list[Path]:
    root = repo_root() / rel
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _calls_admit_background_execution(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == _ADMISSION_GATE:
            return True
        if isinstance(func, ast.Attribute) and func.attr == _ADMISSION_GATE:
            return True
    return False


def discover_production_admit_background_callers() -> frozenset[str]:
    found: set[str] = set()
    for rel in _PRODUCTION_SCAN_ROOTS:
        for path in _python_files_under(rel):
            rel_path = path.relative_to(repo_root()).as_posix()
            if rel_path == "intergrax/runtime/background_execution/required_audit_evidence.py":
                continue
            if _calls_admit_background_execution(path):
                found.add(rel_path)
    return frozenset(found)


def expected_transport_caller_modules() -> frozenset[str]:
    return frozenset(row.module_path for row in TRANSPORT_ENTRYPOINTS)


def assert_transport_closed_world_complete() -> None:
    discovered = discover_production_admit_background_callers()
    expected = expected_transport_caller_modules()
    assert discovered == expected, f"transport caller drift: {discovered} != {expected}"


def assert_no_duplicate_ownership() -> None:
    owners = [row.semantic_owner for row in OWNERSHIP_MATRIX]
    assert len(owners) == len(set(owners))
