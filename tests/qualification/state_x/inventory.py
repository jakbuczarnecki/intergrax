# © Artur Czarnecki. All rights reserved.

"""STATE-X-P0 canonical state-family inventory (mechanical SSOT for SX-F01..SX-F15)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

STATE_X_P0_AUDITED_HEAD: Final[str] = "658cf95864970d7c8bde1c7005cf178747c68b55"

MANDATORY_FAMILY_IDS: Final[tuple[str, ...]] = tuple(f"SX-F{i:02d}" for i in range(1, 16))


class ProjectionOrTruth(StrEnum):
    CANONICAL_TRUTH = "CANONICAL_TRUTH"
    DURABLE_COMPONENT_OF_CANONICAL_TRUTH = "DURABLE_COMPONENT_OF_CANONICAL_TRUTH"
    DERIVED_PROJECTION = "DERIVED_PROJECTION"
    READ_MODEL = "READ_MODEL"
    COORDINATION_ONLY = "COORDINATION_ONLY"
    CONFIGURATION_INPUT = "CONFIGURATION_INPUT"
    EPHEMERAL_RUNTIME_STATE = "EPHEMERAL_RUNTIME_STATE"
    BLOCKED_ARCHITECTURE_DECISION = "BLOCKED_ARCHITECTURE_DECISION"


class SemanticOwnershipRole(StrEnum):
    CANONICAL_OWNER = "CANONICAL_OWNER"
    COMPONENT_OWNER = "COMPONENT_OWNER"
    NO_TRUTH_OWNERSHIP = "NO_TRUTH_OWNERSHIP"


class AuthorityRole(StrEnum):
    MINT_AUTHORITY = "MINT_AUTHORITY"
    CONSTRAIN_CURRENT_AUTHORITY = "CONSTRAIN_CURRENT_AUTHORITY"
    HISTORICAL_EVIDENCE_ONLY = "HISTORICAL_EVIDENCE_ONLY"
    NOT_AUTHORITY_BEARING = "NOT_AUTHORITY_BEARING"


class IdentityRole(StrEnum):
    MINT_IDENTITY = "MINT_IDENTITY"
    PRESERVE_ON_RESTORE = "PRESERVE_ON_RESTORE"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    NOT_IDENTITY_OWNER = "NOT_IDENTITY_OWNER"


class BackupRestoreResponsibility(StrEnum):
    PLATFORM_SUPPORTED_BACKUP_RESTORE = "PLATFORM-SUPPORTED BACKUP/RESTORE"
    BACKEND_OPERATOR_WITH_CONSISTENCY = (
        "BACKEND/OPERATOR RESPONSIBILITY WITH EXPLICIT PLATFORM CONSISTENCY REQUIREMENTS"
    )
    NOT_DURABLE_REBUILDABLE = "NOT DURABLE / REBUILDABLE PROJECTION"
    NOT_APPLICABLE = "NOT APPLICABLE"


class TenantAuditDisposition(StrEnum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"
    NA_WITH_EVIDENCE = "N/A-WITH-EVIDENCE"


class StateFamilyP0Status(StrEnum):
    BASELINE_INVENTORIED = "BASELINE_INVENTORIED"
    READY_FOR_CHILD_CERTIFICATION = "READY_FOR_CHILD_CERTIFICATION"


class BlockerClassification(StrEnum):
    IN_SCOPE_BLOCKER = "IN-SCOPE BLOCKER"
    TRACKED_FREEZE_DEBT = "TRACKED FREEZE DEBT"
    ENVIRONMENT_TEST_ISSUE = "ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED"
    SUPERSEDED = "SUPERSEDED WITH CURRENT-HEAD EVIDENCE"


@dataclass(frozen=True, slots=True)
class StateFamilyInventoryEntry:
    family_id: str
    semantic_name: str
    semantic_owner: str
    semantic_ownership_role: SemanticOwnershipRole
    canonical_contract: str | None
    contract_path: str | None
    composition_owner: str
    production_paths: tuple[str, ...]
    implementation_symbols: tuple[str, ...]
    underlying_family_ids: tuple[str, ...]
    writers: tuple[str, ...]
    readers: tuple[str, ...]
    restore_consumers: tuple[str, ...]
    tenant_scope: str
    concurrency_control: str
    stale_state_rule: str
    authority_role: AuthorityRole
    identity_role: IdentityRole
    projection_or_truth: ProjectionOrTruth
    current_status: StateFamilyP0Status
    applicable_frz: tuple[str, ...]
    backup_restore_responsibility: BackupRestoreResponsibility
    tenant_audit_disposition: TenantAuditDisposition
    corrupt_partial_handling: str
    fork_retry_resume_notes: str
    external_effect_uncertainty_notes: str


@dataclass(frozen=True, slots=True)
class StateXKnownBlocker:
    blocker_id: str
    title: str
    owner_child: str
    classification: BlockerClassification
    evidence: str


STATE_X_KNOWN_BLOCKERS: Final[tuple[StateXKnownBlocker, ...]] = (
    StateXKnownBlocker(
        "SX-B01",
        "NexusLoop checkpoint_store typing vs wire_execution_terminal_store "
        "(GOV-X2 Q2-D1 / ExecutionTerminalPersistenceCapability)",
        "STATE-X-R1",
        BlockerClassification.TRACKED_FREEZE_DEBT,
        "GOV_X2_GOVERNANCE_EXECUTION_E2E_CERTIFICATION.md Q2-D1; "
        "NexusLoop.__init__ TaskCheckpointPersistence | None; "
        "wire_execution_terminal_store expects ExecutionTerminalPersistenceCapability | None; "
        "runtime isinstance guard in execution_terminal/persistence.py",
    ),
    StateXKnownBlocker(
        "SX-B02",
        "Checkpoint historical authority / missing task_snapshot semantics",
        "STATE-X-R1",
        BlockerClassification.IN_SCOPE_BLOCKER,
        "checkpoint_resume_validation._parse_checkpoint_historical_authority; "
        "SQLite task_snapshot_json NOT NULL; P0 lock: CURRENT CANON REQUIRES SNAPSHOT "
        "(valid Task material) for authority parse; legacy-empty semantics → R1 ADR",
    ),
    StateXKnownBlocker(
        "SX-B03",
        "R1-SQLITE-ENV-01 collaborative-work SQLite bootstrap",
        "STATE-X-R4 / QUAL-X / PROD-Q",
        BlockerClassification.ENVIRONMENT_TEST_ISSUE,
        "PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST R1-SQLITE-ENV-01; "
        "not causal to R1 host boundary; requires reproducible classification on current HEAD",
    ),
)


_FRZ_STA_REC: Final[tuple[str, ...]] = (
    "FRZ-STA-01",
    "FRZ-STA-02",
    "FRZ-STA-03",
    "FRZ-STA-04",
    "FRZ-STA-05",
    "FRZ-STA-06",
    "FRZ-STA-07",
    "FRZ-STA-08",
    "FRZ-REC-01",
    "FRZ-REC-02",
    "FRZ-REC-03",
    "FRZ-REC-04",
    "FRZ-REC-05",
    "FRZ-REC-06",
    "FRZ-REC-07",
    "FRZ-REC-08",
    "FRZ-REC-09",
    "FRZ-REC-10",
    "FRZ-TEN-04",
    "FRZ-TEN-08",
)


def _frz(*codes: str) -> tuple[str, ...]:
    return codes


STATE_X_FAMILY_INVENTORY: Final[tuple[StateFamilyInventoryEntry, ...]] = (
    StateFamilyInventoryEntry(
        "SX-F01",
        "Task / Runtime Checkpoint State",
        "Long-running checkpoint subsystem (TaskCheckpoint + RuntimeCheckpoint semantics)",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "TaskCheckpointPersistence",
        "intergrax/runtime/long_running/persistence_contract.py",
        "Host/Nexus composition wires store; semantic owner remains long-running checkpoint",
        ("intergrax/runtime/long_running/store.py",),
        ("SQLiteTaskCheckpointStore",),
        (),
        ("LongRunningCoordinator", "NexusLoop checkpoint save path", "Scheduler resume writer"),
        (
            "LongRunningCoordinator",
            "NexusLoop",
            "Debug/read surfaces via TaskCheckpointReader",
            "Scheduler",
        ),
        ("LongRunningCoordinator", "NexusLoop resume", "Background reentry admission"),
        "tenant_id on TaskCheckpoint; required on reader APIs",
        "expected_revision CAS on save; scheduler claim/fence tables colocated optional",
        "Stale revision → write rejection; resume validation rejects terminal/lineage mismatch",
        AuthorityRole.HISTORICAL_EVIDENCE_ONLY,
        IdentityRole.PRESERVE_ON_RESTORE,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-01", "FRZ-STA-03", "FRZ-STA-05", "FRZ-STA-08", "FRZ-REC-01", "FRZ-REC-02", "FRZ-TEN-04", "FRZ-TEN-08"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "Schema columns + JSON validation on load; CAS; malformed checkpoint resume paths",
        "Resume continues stream; retry/fork validated via lineage + attempt lifecycle (not checkpoint alone)",
        "Terminal/lineage/idempotency consulted before retry semantics",
    ),
    StateFamilyInventoryEntry(
        "SX-F02",
        "Runtime Execution Tree Snapshot",
        "Long-running checkpoint subsystem (component of RuntimeCheckpoint)",
        SemanticOwnershipRole.COMPONENT_OWNER,
        "RuntimeCheckpoint.execution_tree / ExecutionTreeSnapshot",
        "intergrax/runtime/long_running/execution_tree_checkpoint.py",
        "Embedded in TaskCheckpointPersistence payloads; not independent Nexus tree truth",
        ("intergrax/runtime/long_running/store.py",),
        ("ExecutionTreeSnapshot",),
        (),
        ("NexusLoop orchestration", "CheckpointBuilder"),
        ("Resume validation", "Nexus execution tree restore"),
        ("NexusLoop", "checkpoint_resume_validation"),
        "Scoped via parent TaskCheckpoint tenant/task",
        "Revision CAS on parent checkpoint stream",
        "Superseded tree via new checkpoint revision",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.REFERENCE_ONLY,
        ProjectionOrTruth.DURABLE_COMPONENT_OF_CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-02", "FRZ-STA-08", "FRZ-REC-04"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "Validated as part of checkpoint parse/resume",
        "Moves with checkpoint resume; no independent fork store",
        "N/A — structure not external-effect authority",
    ),
    StateFamilyInventoryEntry(
        "SX-F03",
        "Decision Durable Checkpoint State",
        "Decision orchestration / decision recovery plane",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "DecisionCheckpointPersistence / DecisionCheckpointState",
        "intergrax/runtime/execution/decision_checkpoint_persistence.py",
        "Execution host composition (Nexus/decision runtime bindings)",
        (
            "intergrax/runtime/execution/sqlite_decision_checkpoint_persistence.py",
            "intergrax/runtime/execution/in_memory_decision_checkpoint_persistence.py",
        ),
        (
            "SQLiteDecisionCheckpointPersistence",
            "InMemoryDecisionCheckpointPersistence",
        ),
        (),
        ("Decision recovery orchestration", "Active decision checkpoint binding"),
        ("Decision recovery", "Governance/decision resume consumers"),
        ("Decision recovery", "Decision orchestration resume"),
        "DecisionFinalizationKey.tenant_id in primary key",
        "snapshot_revision + expected_revision CAS (SQLite BEGIN IMMEDIATE)",
        "StaleDecisionCheckpointWriteError on revision mismatch",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.PRESERVE_ON_RESTORE,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-01", "FRZ-STA-03", "FRZ-STA-05", "FRZ-REC-04", "FRZ-REC-06", "FRZ-TEN-04"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "Blob codec + restore_decision_checkpoint_state; CAS stale detection",
        "Distinct from task checkpoint; replay semantics owned by STATE-X-R2",
        "Paired with idempotency/terminal for effect certainty (R2)",
    ),
    StateFamilyInventoryEntry(
        "SX-F04",
        "Attempt Lifecycle State",
        "AttemptLifecycleService",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "AttemptLifecycleStore",
        "intergrax/contracts/attempt_lifecycle.py",
        "NexusLoop / queue workers / background execution composition",
        (
            "intergrax/runtime/execution/attempt_lifecycle/persistence.py",
            "intergrax/runtime/execution/attempt_lifecycle/service.py",
        ),
        (
            "InMemoryAttemptLifecycleStore",
            "KvAttemptLifecycleStore",
            "DocumentStoreAttemptLifecycleStore",
            "AttemptLifecycleService",
        ),
        (),
        ("AttemptLifecycleService",),
        ("NexusLoop", "Queue workers", "Background admission"),
        ("Resume/retry admission", "reentry_admission"),
        "tenant via run binding / store partition where durable",
        "generation + CAS on durable providers",
        "Must not re-derive from task checkpoint when store present",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.PRESERVE_ON_RESTORE,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-01", "FRZ-STA-05", "FRZ-REC-03", "FRZ-REC-05", "FRZ-TEN-08"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "Schema version in encoded state; transition guards in service",
        "transition_to_next_attempt vs resume — R2 certification",
        "Terminal store consulted for retry authority",
    ),
    StateFamilyInventoryEntry(
        "SX-F05",
        "Execution Terminal State",
        "Execution terminal semantic plane (ExecutionTerminalService owner)",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "ExecutionTerminalStore / ExecutionTerminalRecord",
        "intergrax/contracts/execution_terminal.py",
        "wire_execution_terminal_store composition; not TaskCheckpointPersistence semantics",
        ("intergrax/runtime/execution/execution_terminal/persistence.py",),
        (
            "InMemoryExecutionTerminalStore",
            "KvExecutionTerminalStore",
            "DocumentStoreExecutionTerminalStore",
            "CheckpointStoreExecutionTerminalStore",
        ),
        (),
        ("ExecutionTerminalService", "Admission paths recording terminal outcomes"),
        ("Resume admission", "Background execution", "NexusLoop"),
        ("checkpoint_resume_validation", "reentry_admission"),
        "tenant_id on ExecutionTerminalRecord",
        "Provider-specific; normalize_terminal_record fail-closed",
        "Terminal denial on resume when sealed",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.REFERENCE_ONLY,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-01", "FRZ-STA-08", "FRZ-REC-02", "FRZ-REC-06", "FRZ-TEN-08"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "normalize_terminal_record; provider preconditions",
        "Terminal records persist; retry requires new attempt semantics",
        "Unknown external outcome must not bypass terminal/idempotency (R3/R2)",
    ),
    StateFamilyInventoryEntry(
        "SX-F06",
        "Execution Lineage Durable State",
        "Execution lineage subsystem",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "ExecutionLineagePersistence",
        "intergrax/contracts/execution_lineage.py",
        "Active lineage binding + host persistence wiring",
        (
            "intergrax/runtime/execution/lineage/persistence.py",
            "intergrax/runtime/execution/lineage/document_store_persistence.py",
        ),
        (
            "InMemoryExecutionLineagePersistence",
            "DocumentStoreExecutionLineagePersistence",
        ),
        (),
        ("Lineage persistence adapters", "Seal/segment writers"),
        ("checkpoint_resume_validation", "Resume orchestration"),
        ("checkpoint_resume_validation", "Long-running resume"),
        "tenant-scoped lineage keys",
        "Provider CAS / revision semantics per implementation",
        "Checkpoint cannot overwrite lineage truth",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.PRESERVE_ON_RESTORE,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-01", "FRZ-REC-04", "FRZ-REC-09", "FRZ-TEN-08"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "Validation on load; segment seal checks",
        "Segment root vs checkpoint root matching on resume",
        "Lineage seal informs retry eligibility",
    ),
    StateFamilyInventoryEntry(
        "SX-F07",
        "Runtime Event / Evidence Persistence",
        "Runtime events / evidence plane",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "RuntimeEventPersistence",
        "intergrax/runtime/events/persistence_contract.py",
        "Observability composition; not recovery command source",
        (
            "intergrax/runtime/events/stores/sqlite_runtime_event_store.py",
            "intergrax/runtime/events/stores/memory_runtime_event_store.py",
            "intergrax/runtime/events/stores/document_backed_runtime_event_store.py",
            "intergrax/runtime/events/stores/validating_runtime_event_store.py",
        ),
        (
            "SQLiteRuntimeEventStore",
            "InMemoryRuntimeEventStore",
            "DocumentBackedRuntimeEventStore",
            "ValidatingRuntimeEventPersistence",
        ),
        (),
        ("Runtime event emitters", "Evidence adapters"),
        ("Observability", "Diagnostics", "Audit readers"),
        ("Not authoritative for resume commands",),
        "tenant on persisted events where required",
        "Append-oriented; integrity errors surfaced",
        "Not canonical execution authority",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.NOT_IDENTITY_OWNER,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-02", "FRZ-STA-07", "FRZ-REC-04", "FRZ-TEN-04"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.NA_WITH_EVIDENCE,
        "RuntimeEventPersistenceIntegrityError; schema validation",
        "Events do not fork execution truth",
        "Evidence does not prove external effect outcome alone",
    ),
    StateFamilyInventoryEntry(
        "SX-F08",
        "Run Trace / Read Projection",
        "Nexus tracing read plane",
        SemanticOwnershipRole.NO_TRUTH_OWNERSHIP,
        "RunTraceStore / RunTraceReader",
        "intergrax/contracts/run_trace_store.py",
        "open_run_trace_store composition; debug/task event readers",
        (
            "intergrax/runtime/nexus/tracing/sqlite_run_trace_store.py",
            "intergrax/runtime/nexus/tracing/store.py",
        ),
        ("SQLiteRunTraceStore", "open_run_trace_store"),
        (),
        ("Nexus tracing writers", "Task event projection"),
        ("RunTraceReader", "Debug router", "Tool registry bindings"),
        ("Not used as resume truth",),
        "tenant on persisted runs where modeled",
        "Append/read model",
        "Stale trace does not override checkpoint",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.REFERENCE_ONLY,
        ProjectionOrTruth.READ_MODEL,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-02", "FRZ-REC-04"),
        BackupRestoreResponsibility.NOT_DURABLE_REBUILDABLE,
        TenantAuditDisposition.NA_WITH_EVIDENCE,
        "Reader validation; not recovery gate",
        "Projection may rebuild from events/checkpoints",
        "N/A",
    ),
    StateFamilyInventoryEntry(
        "SX-F09",
        "Execution Budget Durable State",
        "Execution budget / cost plane",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "RunBudgetPersistence / ExecutionBudgetLedger",
        "intergrax/runtime/execution/budget/persistence.py",
        "DurableRunBudgetLedgerFactory + host budget wiring",
        ("intergrax/runtime/execution/budget/persistence.py",),
        (
            "KvRunBudgetPersistence",
            "DocumentStoreRunBudgetPersistence",
            "DurableRunBudgetLedgerFactory",
        ),
        (),
        ("DurableRunBudgetLedgerFactory", "Budget enforcement hooks"),
        ("Budget enforcement", "Restore on resume hosts"),
        ("Resume budget restore",),
        "tenant partition on document/KV backends",
        "Persisted ledger snapshots; reservation semantics in ledger",
        "Configured RunBudget != persisted ledger (FRZ-STA-06)",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.REFERENCE_ONLY,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-06", "FRZ-STA-03", "FRZ-REC-09"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "RunBudgetPersistenceError on corrupt snapshots",
        "Restore reloads reservations/consumption — R3",
        "Budget does not gate external effect certainty",
    ),
    StateFamilyInventoryEntry(
        "SX-F10",
        "Idempotency State",
        "Side-effect deduplication plane",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "IdempotencyStore",
        "intergrax/contracts/idempotency_store.py",
        "Host/Nexus composition separate from checkpoint store",
        (
            "intergrax/runtime/tools/in_memory_idempotency_store.py",
            "intergrax/runtime/tools/sqlite_idempotency_store.py",
            "intergrax/runtime/persistence/sqlite_composition.py",
        ),
        (
            "InMemoryIdempotencyStore",
            "SQLiteIdempotencyStore",
            "create_sqlite_idempotency_store",
        ),
        (),
        ("Tool runtime", "Effect executors"),
        ("Replay/retry guards",),
        ("Recovery/replay eligibility",),
        "tenant-scoped keys per contract",
        "Key-level record CAS / upsert semantics per provider",
        "Must not heuristically rebuild from events if store exists",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.NOT_IDENTITY_OWNER,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-01", "FRZ-REC-04", "FRZ-REC-07", "FRZ-TEN-04"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "Provider-specific record validation",
        "Shared across retries; fork may require new keys — R3",
        "Unknown outcome requires idempotency read — FRZ-REC-07",
    ),
    StateFamilyInventoryEntry(
        "SX-F11",
        "Compensation Queue State",
        "Compensation queue semantic owner",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "CompensationQueueStore",
        "intergrax/agents/persistence/compensation_queue_store.py",
        "Host composition (separate from Nexus checkpoint_store)",
        ("intergrax/agents/persistence/compensation_queue_store.py",),
        (
            "InMemoryCompensationQueueStore",
            "SQLiteCompensationQueueStore",
        ),
        (),
        ("Compensation processors",),
        ("Compensation workers",),
        ("Recovery compensation flows",),
        "tenant on queue records",
        "Claim/processing semantics per store",
        "Not a shadow scheduler",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.NOT_IDENTITY_OWNER,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-01", "FRZ-REC-07", "FRZ-REC-05"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "SQLite/queue validation",
        "Queue items terminal independently of checkpoint",
        "Compensation pairs with idempotency/terminal",
    ),
    StateFamilyInventoryEntry(
        "SX-F12",
        "Human Decision / HITL Persistence",
        "Human decision persistence plane",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "HumanDecisionPersistence",
        "intergrax/runtime/human/persistence_contract.py",
        "Human governance composition",
        (
            "intergrax/runtime/human/store.py",
            "intergrax/runtime/persistence/sqlite_composition.py",
        ),
        (
            "SQLiteHumanDecisionStore",
            "create_sqlite_human_decision_store",
        ),
        (),
        ("HITL flows", "Agent governance grant lifecycle"),
        ("Governance admission", "Audit"),
        ("Governance re-admission on resume — not authority reuse",),
        "tenant on human decision records",
        "Durable append/update per contract",
        "persisted human decision != reusable execution authority",
        AuthorityRole.HISTORICAL_EVIDENCE_ONLY,
        IdentityRole.REFERENCE_ONLY,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-07", "FRZ-REC-02", "FRZ-TEN-04"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "Contract validation on load",
        "Resume requires fresh governance admission",
        "Human evidence does not prove external effect",
    ),
    StateFamilyInventoryEntry(
        "SX-F13",
        "Scheduler Durable State",
        "Long-running scheduler (WHEN to resume)",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "ScheduledResumePersistence / SchedulerLedger",
        "intergrax/runtime/long_running/persistence_contract.py",
        "Colocated in TaskCheckpointPersistence implementations; semantic scheduler owner",
        ("intergrax/runtime/long_running/store.py",),
        ("SQLiteTaskCheckpointStore",),
        (),
        ("Scheduler service", "claim_action writers"),
        ("Scheduler", "Resume dispatcher"),
        ("Scheduled resume consumer",),
        "ledger keys; tenant via scheduled resume rows",
        "claim_action lease/fence; has_action completion",
        "Scheduler does not mint execution authority",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.NOT_IDENTITY_OWNER,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-01", "FRZ-REC-01", "FRZ-TEN-08"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "Claim lease expiry; action ledger",
        "Resume trigger only",
        "N/A",
    ),
    StateFamilyInventoryEntry(
        "SX-F14",
        "Agent Checkpoint State (ACP)",
        "Agent persistence / ACP step checkpoint plane",
        SemanticOwnershipRole.CANONICAL_OWNER,
        "AgentCheckpointStore",
        "intergrax/agents/persistence/checkpoint_store.py",
        "Host resolve_host_agent_checkpoint_store; NexusLoop.agent_checkpoint_store surface",
        ("intergrax/agents/persistence/checkpoint_store.py",),
        (
            "InMemoryAgentCheckpointStore",
            "SQLiteAgentCheckpointStore",
        ),
        (),
        ("ACP agent steps",),
        ("Agent resume", "Host harness"),
        ("Agent-local resume",),
        "tenant_id on AgentRunCheckpoint",
        "expected_revision CAS on save",
        "Independent semantic family from TaskCheckpoint (agent-local durable state)",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.PRESERVE_ON_RESTORE,
        ProjectionOrTruth.CANONICAL_TRUTH,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-STA-01", "FRZ-STA-02", "FRZ-REC-03"),
        BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY,
        TenantAuditDisposition.BLOCKED,
        "CheckpointRevisionConflictError; step regression guards",
        "Agent run scoped — not task checkpoint fork",
        "SideEffectRecord coordination — SX-F15",
    ),
    StateFamilyInventoryEntry(
        "SX-F15",
        "Execution / Compensation / Side-Effect Recovery State",
        "Cross-family recovery coordination (underlying families retain semantic ownership)",
        SemanticOwnershipRole.NO_TRUTH_OWNERSHIP,
        None,
        None,
        "Nexus/host composition coordinates; each family retains semantic owner",
        ("intergrax/runtime/execution/decision_recovery.py",),
        (
            "resume_decision_from_durable_state",
            "reconcile_checkpoint_with_durable_finalization",
        ),
        ("SX-F01", "SX-F05", "SX-F10", "SX-F11"),
        ("Effect executors", "Compensation workers", "Recovery orchestration"),
        ("Recovery orchestration", "Admission"),
        ("Unified recovery admission",),
        "tenant continuity enforced in underlying families; F15 coordinates only",
        "Family-specific CAS/fencing in underlying stores",
        "No duplicated semantic truth across families",
        AuthorityRole.NOT_AUTHORITY_BEARING,
        IdentityRole.REFERENCE_ONLY,
        ProjectionOrTruth.COORDINATION_ONLY,
        StateFamilyP0Status.BASELINE_INVENTORIED,
        _frz("FRZ-REC-04", "FRZ-REC-07", "FRZ-STA-02"),
        BackupRestoreResponsibility.NOT_APPLICABLE,
        TenantAuditDisposition.NA_WITH_EVIDENCE,
        "Per-family corrupt handling; R3/R2 certification",
        "resume != retry != fork — family-specific",
        "FRZ-REC-07 matrix — R2/R3",
    ),
)


STATE_X_P0_ALLOWLIST_PATHS: Final[frozenset[str]] = frozenset(
    {
        "docs/project/maintainers/qualification/STATE_X_P0_STATE_TRUTH_OWNERSHIP_RECOVERY_BASELINE.md",
        "tests/qualification/state_x/inventory.py",
        "tests/qualification/state_x/test_state_x_p0_baseline.py",
        "tests/qualification/state_x/__init__.py",
    },
)
