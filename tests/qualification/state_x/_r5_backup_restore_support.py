# © Artur Czarnecki. All rights reserved.

"""STATE-X-R5 — backup/restore responsibility and recovery consistency support."""

from __future__ import annotations

import ast
import re
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tests.qualification.state_x.inventory import (
    CURRENT_STATE_X_FAMILY_IDS,
    HISTORICAL_BASE_FAMILY_IDS,
    MANDATORY_FAMILY_IDS,
    BackupRestoreResponsibility,
    STATE_X_FAMILY_INVENTORY,
    StateFamilyInventoryEntry,
)

STATE_X_R5_PRE_AUDIT_HEAD: Final[str] = "61faf8f317125b996526ceceaf5754b9c28073d6"

STATE_X_R4_ACCEPTED_CLOSURE_SHA: Final[str] = "61faf8f317125b996526ceceaf5754b9c28073d6"

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PLATFORM_VALIDATION_OWNER: Final[str] = (
    "Integrax — family contract load validators + cross-family recovery gates"
)
_OPERATOR_PHYSICAL_OWNER: Final[str] = "Backend/operator — physical snapshot/restore transport"

_FORBIDDEN_RESPONSIBILITY_TOKENS: Final[frozenset[str]] = frozenset(
    {"UNKNOWN", "TBD", "BEST EFFORT", "BEST_EFFORT"},
)

_PERSISTENCE_CONTRACT_GLOB: Final[tuple[str, ...]] = (
    "intergrax/runtime/long_running/persistence_contract.py",
    "intergrax/runtime/execution/decision_checkpoint_persistence.py",
    "intergrax/runtime/events/persistence_contract.py",
    "intergrax/runtime/human/persistence_contract.py",
    "intergrax/contracts/idempotency_store.py",
    "intergrax/agents/persistence/compensation_queue_store.py",
    "intergrax/agents/persistence/checkpoint_store.py",
)


@dataclass(frozen=True, slots=True)
class BackupRestoreFamilyRecord:
    family_id: str
    responsibility: BackupRestoreResponsibility
    physical_backup_unit: str
    semantic_restore_dependencies: tuple[str, ...]
    supported_restore_semantics: str
    restore_validation_owner: str
    partial_restore_behavior: str
    corruption_behavior: str
    tenant_behavior: str
    provider_scope: str
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhysicalBackupUnitRecord:
    provider_store: str
    physical_backup_unit: str
    state_x_families: tuple[str, ...]
    operator_responsibility: str
    platform_semantic_validation: str


@dataclass(frozen=True, slots=True)
class SemanticRecoveryFlowRecord:
    recovery_flow: str
    required_families: tuple[str, ...]
    optional_families: tuple[str, ...]
    evidence_only_families: tuple[str, ...]
    validation_owner: str
    failure_behavior: str


def _entry(family_id: str) -> StateFamilyInventoryEntry:
    for item in STATE_X_FAMILY_INVENTORY:
        if item.family_id == family_id:
            return item
    raise KeyError(family_id)


def _is_durable_responsibility(resp: BackupRestoreResponsibility) -> bool:
    return resp is BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY


_R5_FAMILY_EXTENSION: Final[dict[str, tuple[str, tuple[str, ...], str, str, str, str, str, tuple[str, ...]]]] = {
    "SX-F01": (
        "SQLiteTaskCheckpointStore: one SQLite DB file (task_checkpoints + embedded runtime tree); "
        "KV/DocumentStore providers: operator partition",
        ("SX-F02", "SX-F04", "SX-F05", "SX-F06", "SX-F09", "SX-F13"),
        "Operator restores physical unit; platform validates schema/identity/revision before resume",
        "Missing terminal/idempotency/lineage when required → FAIL CLOSED via recovery gates",
        "Malformed JSON/schema/CAS violation → load or resume rejection",
        "tenant_id on TaskCheckpoint; cross-tenant restore rejected at validation",
        "SQLiteTaskCheckpointStore; InMemory reference (non-durable)",
        ("R4-Q05..Q15", "R5-Q10..Q13", "test_state_x_r1_checkpoint_resume_terminal"),
    ),
    "SX-F02": (
        "Colocated in SX-F01 TaskCheckpoint physical unit (runtime_checkpoint_json payload)",
        ("SX-F01", "SX-F06"),
        "Restored only as validated component of parent checkpoint stream",
        "Independent F02 restore without F01 → N/A (not a separate physical unit)",
        "Validated during checkpoint parse/resume",
        "Inherited from parent checkpoint tenant binding",
        "Embedded in TaskCheckpointPersistence implementations",
        ("R4-R1 execution tree mismatch tests", "R5-Q07"),
    ),
    "SX-F03": (
        "SQLiteDecisionCheckpointPersistence: one SQLite DB; InMemory non-durable reference",
        ("SX-F05", "SX-F10", "SX-F04"),
        "Operator restores decision DB; platform restore_decision_checkpoint_state + CAS",
        "Decision store skew vs terminal/idempotency → FAIL CLOSED on effect recovery (R2)",
        "Blob codec / revision mismatch → explicit error",
        "DecisionFinalizationKey.tenant_id partition",
        "SQLite + InMemory decision checkpoint providers",
        ("STATE-X-R2 qualification", "R5-Q35"),
    ),
    "SX-F04": (
        "AttemptLifecycle durable provider partition (KV/DocumentStore/InMemory reference)",
        ("SX-F01", "SX-F05"),
        "Preserve attempt generation; resume vs new attempt via AttemptLifecycleService",
        "Stale attempt vs checkpoint → FAIL CLOSED at admission",
        "Schema version + transition guards",
        "Run/tenant binding on durable keys",
        "KvAttemptLifecycleStore; DocumentStoreAttemptLifecycleStore; InMemory reference",
        ("R2/R4 attempt admission", "R5-Q17"),
    ),
    "SX-F05": (
        "CheckpointStoreExecutionTerminalStore: SX-F01 SQLite file; "
        "else KV/DocumentStore/InMemory provider partition",
        ("SX-F01", "SX-F04", "SX-F10"),
        "Terminal truth restored with operator transport; normalize_terminal_record fail-closed",
        "Terminal newer/missing vs checkpoint → resume/effect FAIL CLOSED",
        "normalize_terminal_record denial",
        "ExecutionTerminalRecord.tenant_id",
        "Checkpoint colocated + KV + DocumentStore + InMemory",
        ("R4-Q19", "R1 terminal composition", "R5-Q15"),
    ),
    "SX-F06": (
        "DocumentStore/InMemory lineage provider partition",
        ("SX-F01", "SX-F02"),
        "Lineage seal/root must match checkpoint on resume",
        "Lineage skew → FAIL CLOSED in checkpoint_resume_validation",
        "Segment seal validation on load",
        "Tenant-scoped lineage keys",
        "DocumentStoreExecutionLineagePersistence; InMemory reference",
        ("R4-R1 lineage tests", "R5-Q16"),
    ),
    "SX-F07": (
        "SQLite runtime_events DB; DocumentStore/memory reference providers",
        (),
        "Evidence continuity optional for execution; loss must not mint execution truth",
        "Missing events when checkpoint present → SAFE CONTINUE for execution (non-authoritative)",
        "RuntimeEventPersistenceIntegrityError",
        "Event tenant fields preserved; not resume authority",
        "SQLiteRuntimeEventStore; DocumentBacked; InMemory",
        ("inventory restore_consumers", "R5-Q25"),
    ),
    "SX-F08": (
        "N/A — separate trace SQLite file in runtime bundle (non-authoritative for execution recovery)",
        (),
        "Loss-tolerant trace projection; automatic reconstruction NOT certified by R5; "
        "must not alter checkpoint/terminal/idempotency truth",
        "Trace loss → observability only; not execution-recovery authority; no guaranteed rebuild",
        "Reader validation only",
        "Tenant on trace rows where modeled",
        "SQLiteRunTraceStore; open_run_trace_store composition",
        ("R5-Q26", "inventory projection_or_truth=READ_MODEL"),
    ),
    "SX-F09": (
        "KV/DocumentStore budget persistence partition",
        ("SX-F01",),
        "Reload persisted ledger snapshot — not configured reset",
        "Budget missing with checkpoint → policy-defined FAIL CLOSED on enforcement hosts",
        "RunBudgetPersistenceError on corrupt snapshots",
        "Tenant partition on durable backends",
        "KvRunBudgetPersistence; DocumentStoreRunBudgetPersistence",
        ("R3 budget restore tests", "R5-Q18"),
    ),
    "SX-F10": (
        "SQLite idempotency DB; Redis namespace; InMemory reference",
        ("SX-F05", "SX-F11"),
        "Idempotency backup required for effectful recovery paths relying on store",
        "Checkpoint restored without idempotency → external effect must not blind replay (UNCERTAIN)",
        "Provider record validation",
        "Tenant-scoped idempotency keys",
        "SQLiteIdempotencyStore; RedisIdempotencyStore; InMemory",
        ("R3-R2 idempotency matrix", "R5-Q19", "R5-Q41"),
    ),
    "SX-F11": (
        "SQLite compensation queue DB; InMemory reference",
        ("SX-F10", "SX-F05"),
        "Operator must restore queue consistently; platform does not infer completion from absence",
        "Missing/stale compensation with restored execution → FAIL CLOSED / operator MUST restore",
        "Queue record validation",
        "Tenant on compensation records",
        "SQLiteCompensationQueueStore; InMemoryCompensationQueueStore",
        ("R3-R2 compensation matrix", "R5-Q20", "R5-Q21"),
    ),
    "SX-F12": (
        "SQLite human_decisions DB (bundle path separate from task_checkpoints)",
        ("SX-F01",),
        "Historical human decision = evidence; fresh Governance admission still required",
        "HumanDecision without checkpoint → cannot execute alone; missing decision → no synthesized approval",
        "Contract validation on load",
        "tenant on human decision records",
        "SQLiteHumanDecisionStore",
        ("R3-R3 HITL", "R4-R1 HITL replay", "R5-Q22", "R5-Q23"),
    ),
    "SX-F13": (
        "Colocated in SQLiteTaskCheckpointStore file (scheduled_resumes + scheduler_ledger)",
        ("SX-F01",),
        "Scheduler tables restored with same SQLite file as checkpoints",
        "Partial scheduler restore without checkpoint file → FAIL CLOSED for scheduled resume",
        "Claim lease/fence validation",
        "Tenant via scheduled resume rows",
        "SQLiteTaskCheckpointStore scheduler tables",
        ("R3-R4 scheduler", "R5-Q12"),
    ),
    "SX-F14": (
        "SQLiteAgentCheckpointStore DB; InMemory reference",
        ("SX-F15",),
        "Agent-local durable state; run/tenant/agent/revision validation before resume",
        "Skew vs host composition → FAIL CLOSED",
        "CheckpointRevisionConflictError; corrupt JSON fail-closed",
        "AgentRunCheckpoint.tenant_id",
        "SQLiteAgentCheckpointStore; InMemoryAgentCheckpointStore",
        ("R3-R5 agent checkpoint", "R5-Q24"),
    ),
    "SX-F15": (
        "N/A — coordination only; no independent durable store",
        ("SX-F01", "SX-F05", "SX-F10", "SX-F11"),
        "Coordinates underlying family validators; no standalone backup unit",
        "Cross-family mismatch → FAIL CLOSED per family gates (not heuristic repair)",
        "Per-family corrupt handling",
        "Tenant enforced in underlying families",
        "decision_recovery.py orchestration",
        ("R5-Q27", "inventory COORDINATION_ONLY"),
    ),
    "SX-F16": (
        "KV/DocumentStore partitions for background identity records",
        ("SX-F04", "SX-F05"),
        "Operator restores identity backend; platform validates codec + conflict rules",
        "Missing identity with active transport redelivery → admission fail closed",
        "Invalid v1/v2 / conflict → BackgroundExecutionIdentityConflictError",
        "tenant_id + provider namespace partition",
        "KvBackgroundExecutionIdentityPersistence; DocumentStoreBackgroundExecutionIdentityPersistence",
        ("NPSC conformance", "background_execution unit tests", "SXF-R1-Q12..Q13"),
    ),
    "SX-F17": (
        "ExecutionContinuationDurableBacking file/KV unit",
        ("SX-F01", "SX-F05"),
        "Operator restores continuation backing; platform CAS + episode invariants",
        "Corrupt/missing episode when durable required → fail closed",
        "Corrupt snapshot bytes rejected",
        "tenant via execution continuation identity binding",
        "ExecutionContinuationDurableStateFilePersistence",
        ("GR-5 restart qualification", "SXF-R1-Q15..Q17"),
    ),
    "SX-F18": (
        "KV/DocumentStore per-tenant run deadline authority documents",
        ("SX-F01",),
        "Operator restores deadline store; platform validates tenant+run binding",
        "Stale restore cannot widen deadline authority",
        "Corrupt codec fail closed",
        "tenant_id + run_id key",
        "KvExecutionDeadlinePersistence; DocumentStoreExecutionDeadlinePersistence",
        ("deadline_authority unit tests", "SXF-R1-Q19..Q22"),
    ),
    "SX-F19": (
        "DocumentStore delegated correlation partition",
        ("SX-F04", "SX-F05"),
        "Operator restores correlation documents; REQUIRED mode rejects silent fallback",
        "Conflicting correlation → DelegatedInvocationCorrelationPersistenceError",
        "Corrupt document fail closed",
        "tenant on DelegatedInvocationCorrelationRecord",
        "DocumentStoreDelegatedInvocationCorrelationStore",
        ("delegated correlation durability tests", "SXF-R1-Q24..Q26"),
    ),
    "SX-F20": (
        "DocumentStore suspended operation descriptor partition",
        ("SX-F17", "SX-F12"),
        "Operator restores descriptor documents; claim invariants validated on load",
        "Descriptor conflict / invariant violation → fail closed",
        "Corrupt descriptor payload rejected",
        "tenant via descriptor / governed correlation",
        "DocumentStoreSuspendedExecutionOperationStore",
        ("suspended_operation store tests", "SXF-R1 broad-exclusion audit"),
    ),
}


def build_r5_family_records(
    family_ids: tuple[str, ...],
) -> tuple[BackupRestoreFamilyRecord, ...]:
    records: list[BackupRestoreFamilyRecord] = []
    for family_id in family_ids:
        entry = _entry(family_id)
        ext = _R5_FAMILY_EXTENSION[family_id]
        records.append(
            BackupRestoreFamilyRecord(
                family_id=family_id,
                responsibility=entry.backup_restore_responsibility,
                physical_backup_unit=ext[0],
                semantic_restore_dependencies=ext[1],
                supported_restore_semantics=ext[2],
                restore_validation_owner=_PLATFORM_VALIDATION_OWNER,
                partial_restore_behavior=ext[3],
                corruption_behavior=ext[4] or entry.corrupt_partial_handling,
                tenant_behavior=ext[5],
                provider_scope=ext[6],
                evidence=ext[7],
            ),
        )
    return tuple(records)


R5_BACKUP_RESTORE_FAMILY_MATRIX: Final[tuple[BackupRestoreFamilyRecord, ...]] = (
    build_r5_family_records(MANDATORY_FAMILY_IDS)
)

CURRENT_R5_BACKUP_RESTORE_FAMILY_MATRIX: Final[tuple[BackupRestoreFamilyRecord, ...]] = (
    build_r5_family_records(CURRENT_STATE_X_FAMILY_IDS)
)

PHYSICAL_BACKUP_UNIT_MATRIX: Final[tuple[PhysicalBackupUnitRecord, ...]] = (
    PhysicalBackupUnitRecord(
        "SQLiteTaskCheckpointStore",
        "One SQLite database file: task_checkpoints, scheduled_resumes, scheduler_ledger, "
        "task_execution_terminal (when terminal capability used)",
        ("SX-F01", "SX-F02", "SX-F13", "SX-F05"),
        _OPERATOR_PHYSICAL_OWNER,
        _PLATFORM_VALIDATION_OWNER,
    ),
    PhysicalBackupUnitRecord(
        "SQLiteDecisionCheckpointPersistence",
        "One SQLite database file for decision checkpoint blobs",
        ("SX-F03",),
        _OPERATOR_PHYSICAL_OWNER,
        _PLATFORM_VALIDATION_OWNER,
    ),
    PhysicalBackupUnitRecord(
        "SQLiteRuntimePersistenceBundle (multi-file)",
        "Independent DB paths: trace, runtime_events, task_checkpoints, human_decisions, "
        "idempotency, … — NOT one atomic cross-file unit",
        ("SX-F07", "SX-F08", "SX-F10", "SX-F12"),
        _OPERATOR_PHYSICAL_OWNER,
        "Integrax defines fail-closed semantics when files restored inconsistently",
    ),
    PhysicalBackupUnitRecord(
        "SQLiteIdempotencyStore / RedisIdempotencyStore",
        "Idempotency DB file or Redis logical namespace",
        ("SX-F10",),
        _OPERATOR_PHYSICAL_OWNER,
        _PLATFORM_VALIDATION_OWNER,
    ),
    PhysicalBackupUnitRecord(
        "SQLiteCompensationQueueStore",
        "One SQLite database file for compensation queue",
        ("SX-F11",),
        _OPERATOR_PHYSICAL_OWNER,
        _PLATFORM_VALIDATION_OWNER,
    ),
    PhysicalBackupUnitRecord(
        "SQLiteAgentCheckpointStore",
        "One SQLite database file for agent run checkpoints",
        ("SX-F14",),
        _OPERATOR_PHYSICAL_OWNER,
        _PLATFORM_VALIDATION_OWNER,
    ),
    PhysicalBackupUnitRecord(
        "KV/DocumentStore durable providers",
        "Backend partition per provider contract (terminal, lineage, attempt, budget)",
        ("SX-F04", "SX-F05", "SX-F06", "SX-F09"),
        _OPERATOR_PHYSICAL_OWNER,
        _PLATFORM_VALIDATION_OWNER,
    ),
)

SEMANTIC_RECOVERY_FLOW_MATRIX: Final[tuple[SemanticRecoveryFlowRecord, ...]] = (
    SemanticRecoveryFlowRecord(
        "task_resume",
        ("SX-F01", "SX-F02", "SX-F04", "SX-F05", "SX-F06", "SX-F09"),
        ("SX-F10", "SX-F11"),
        ("SX-F07", "SX-F12"),
        "checkpoint_resume_validation + LongRunningCoordinator",
        "FAIL CLOSED on identity/schema/stale/terminal/lineage mismatch",
    ),
    SemanticRecoveryFlowRecord(
        "scheduled_task_resume",
        ("SX-F01", "SX-F13", "SX-F05"),
        ("SX-F04", "SX-F06"),
        ("SX-F07",),
        "Scheduler + checkpoint_resume_validation",
        "FAIL CLOSED when ledger/schedule inconsistent with checkpoint file",
    ),
    SemanticRecoveryFlowRecord(
        "decision_recovery",
        ("SX-F03", "SX-F05", "SX-F10"),
        ("SX-F04",),
        ("SX-F07",),
        "decision_recovery.py + R2 validators",
        "FAIL CLOSED on stale decision or uncertain effect state",
    ),
    SemanticRecoveryFlowRecord(
        "retry_new_attempt",
        ("SX-F04", "SX-F05", "SX-F06"),
        ("SX-F01", "SX-F10"),
        ("SX-F07",),
        "AttemptLifecycleService + terminal admission",
        "FAIL CLOSED — FRZ-REC-05 fork semantics not closed in R5",
    ),
    SemanticRecoveryFlowRecord(
        "effect_replay",
        ("SX-F10", "SX-F05", "SX-F11"),
        ("SX-F01",),
        ("SX-F07",),
        "IdempotencyPreEffectCoordinator + compensation wiring",
        "UNCERTAIN/RETRYABLE — no duplicate external effect",
    ),
    SemanticRecoveryFlowRecord(
        "compensation_recovery",
        ("SX-F11", "SX-F10", "SX-F05"),
        ("SX-F01",),
        (),
        "Compensation workers + R3-R2 gates",
        "FAIL CLOSED if required compensation truth missing",
    ),
    SemanticRecoveryFlowRecord(
        "HITL_resume",
        ("SX-F01", "SX-F12"),
        (),
        ("SX-F07",),
        "Governance admission + checkpoint validation",
        "Restored HumanDecision cannot mint authority; FAIL CLOSED without admission",
    ),
    SemanticRecoveryFlowRecord(
        "agent_checkpoint_resume",
        ("SX-F14",),
        ("SX-F15",),
        (),
        "AgentCheckpointStore validators + host composition",
        "FAIL CLOSED on tenant/run/agent/revision mismatch",
    ),
    SemanticRecoveryFlowRecord(
        "side_effect_coordination",
        ("SX-F01", "SX-F05", "SX-F10", "SX-F11"),
        (),
        ("SX-F07",),
        "SX-F15 coordination — underlying store validators",
        "FAIL CLOSED on cross-family effect mismatch",
    ),
)


def assert_frz_rec_08_r5_completeness() -> None:
    matrix_ids = {row.family_id for row in R5_BACKUP_RESTORE_FAMILY_MATRIX}
    assert matrix_ids == set(HISTORICAL_BASE_FAMILY_IDS)


def assert_state_x_current_backup_restore_completeness() -> None:
    matrix_ids = {row.family_id for row in CURRENT_R5_BACKUP_RESTORE_FAMILY_MATRIX}
    assert matrix_ids == set(CURRENT_STATE_X_FAMILY_IDS)
    for row in R5_BACKUP_RESTORE_FAMILY_MATRIX:
        inv = _entry(row.family_id)
        assert row.responsibility is inv.backup_restore_responsibility
        blob = " ".join(
            (
                row.physical_backup_unit,
                row.supported_restore_semantics,
                row.partial_restore_behavior,
                row.corruption_behavior,
                row.tenant_behavior,
                row.provider_scope,
            ),
        ).upper()
        for token in _FORBIDDEN_RESPONSIBILITY_TOKENS:
            assert token not in blob, f"{row.family_id} contains forbidden token {token}"
        assert row.supported_restore_semantics.strip()
        assert row.partial_restore_behavior.strip()
        assert row.corruption_behavior.strip()
        assert row.tenant_behavior.strip()
        if _is_durable_responsibility(row.responsibility):
            assert row.physical_backup_unit.strip()
            assert not row.physical_backup_unit.startswith("N/A — rebuildable")
    for flow in SEMANTIC_RECOVERY_FLOW_MATRIX:
        assert flow.recovery_flow.strip()
        assert flow.validation_owner.strip()
        assert flow.failure_behavior.strip()
        assert flow.required_families or flow.optional_families or flow.evidence_only_families


def persistence_contracts_lack_backup_api() -> bool:
    pattern = re.compile(r"^\s*def\s+(backup|restore|export_snapshot|import_snapshot)\s*\(", re.M)
    for rel in _PERSISTENCE_CONTRACT_GLOB:
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        if pattern.search(text):
            return False
    return True


def sqlite_runtime_bundle_paths_are_distinct() -> bool:
    path = _REPO_ROOT / "intergrax/integrations/providers/relational_store/sqlite/paths.py"
    text = path.read_text(encoding="utf-8")
    return "task_checkpoints" in text and "trace" in text and "runtime_events" in text


def run_trace_not_resume_consumer() -> bool:
    inv = _entry("SX-F08")
    return inv.restore_consumers == ("Not used as resume truth",)


_RESUME_AUTHORITY_SOURCE_PATHS: Final[tuple[str, ...]] = (
    "intergrax/runtime/long_running/checkpoint_resume_validation.py",
    "intergrax/runtime/long_running/coordinator.py",
    "intergrax/runtime/cancellation/resume_admission.py",
    "intergrax/runtime/execution/decision_recovery.py",
    "intergrax/runtime/tools/idempotency_pre_effect_coordinator.py",
)


def run_trace_excluded_from_execution_recovery_authority_sources() -> bool:
    needles = ("RunTraceStore", "run_trace_store", "SQLiteRunTraceStore")
    for rel in _RESUME_AUTHORITY_SOURCE_PATHS:
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        if any(token in text for token in needles):
            return False
    return True


@dataclass(frozen=True, slots=True)
class R5BehavioralEvidenceRef:
    evidence_id: str
    module: str
    test_name: str


R5_FRZ_REC_08_BEHAVIORAL_EVIDENCE: Final[tuple[R5BehavioralEvidenceRef, ...]] = (
    R5BehavioralEvidenceRef(
        "checkpoint_terminal_skew",
        "tests.qualification.state_x._r5_q1_cross_store_restore_tests",
        "test_r5_q1_old_checkpoint_newer_terminal_restore_skew_fails_closed",
    ),
    R5BehavioralEvidenceRef(
        "compensation_idempotency_skew",
        "tests.qualification.state_x._r5_q1_cross_store_restore_tests",
        "test_r5_q1_old_compensation_queue_newer_idempotency_no_duplicate_effect",
    ),
    R5BehavioralEvidenceRef(
        "corrupt_terminal_restore",
        "tests.qualification.state_x._r5_q1_cross_store_restore_tests",
        "test_r5_q1_terminal_backend_corruption_is_not_treated_as_absence",
    ),
    R5BehavioralEvidenceRef(
        "corrupt_durable_restore",
        "tests.qualification.state_x._r5_backup_restore_qualification_tests",
        "test_r5_q28_corrupt_task_checkpoint_fail_closed",
    ),
    R5BehavioralEvidenceRef(
        "tenant_restore",
        "tests.qualification.state_x._r5_backup_restore_qualification_tests",
        "test_r5_q30_cross_tenant_restored_state_rejected",
    ),
    R5BehavioralEvidenceRef(
        "authority_preservation",
        "tests.qualification.state_x._r5_backup_restore_qualification_tests",
        "test_r5_q32_restore_cannot_mint_execution_authority",
    ),
    R5BehavioralEvidenceRef(
        "run_trace_non_authoritative",
        "tests.qualification.state_x._r5_q1_cross_store_restore_tests",
        "test_r5_q1_run_trace_loss_is_non_authoritative_not_claimed_rebuildable",
    ),
)


def assert_frz_rec_08_behavioral_evidence_complete() -> None:
    for ref in R5_FRZ_REC_08_BEHAVIORAL_EVIDENCE:
        mod = importlib.import_module(ref.module)
        fn = getattr(mod, ref.test_name, None)
        assert callable(fn), f"missing behavioral evidence {ref.evidence_id}: {ref.test_name}"


def sx_f15_has_no_persistence_contract() -> bool:
    inv = _entry("SX-F15")
    return len(inv.contract_references) == 0


def decision_recovery_source_references_families() -> bool:
    path = _REPO_ROOT / "intergrax/runtime/execution/decision_recovery.py"
    text = path.read_text(encoding="utf-8")
    return "idempotency" in text.lower() or "terminal" in text.lower()
