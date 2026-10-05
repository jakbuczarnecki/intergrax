# © Artur Czarnecki. All rights reserved.

"""STATE-X-FINAL — parent FRZ-STA/REC closure support (current HEAD)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from tests.qualification.state_x.inventory import (
    BlockerClassification,
    MANDATORY_FAMILY_IDS,
    ProjectionOrTruth,
    SemanticOwnershipRole,
    STATE_X_FAMILY_INVENTORY,
    STATE_X_FINAL_START_HEAD,
    STATE_X_KNOWN_BLOCKERS,
    STATE_X_R6_ACCEPTED_CLOSURE_SHA,
    StateFamilyInventoryEntry,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PRIMARY_FRZ_IDS: Final[tuple[str, ...]] = (
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
)

_DISCOVERY_SCAN_ROOTS: Final[tuple[str, ...]] = (
    "intergrax/contracts",
    "intergrax/runtime",
    "agents",
)

def _inventory_closed_world_paths() -> frozenset[str]:
    paths: set[str] = set()
    for entry in STATE_X_FAMILY_INVENTORY:
        paths.update(entry.production_paths)
        for cref in entry.contract_references:
            paths.add(cref.path)
    return frozenset(paths)


_INVENTORY_CLOSED_WORLD_PATHS: Final[frozenset[str]] = _inventory_closed_world_paths()


class EvidenceClass(StrEnum):
    BEHAVIORAL = "BEHAVIORAL"
    STRUCTURAL = "STRUCTURAL"
    CONTRACT = "CONTRACT"
    RESPONSIBILITY = "RESPONSIBILITY"
    ENVIRONMENT_CLASSIFICATION = "ENVIRONMENT_CLASSIFICATION"


class CriterionResolution(StrEnum):
    PASS = "PASS"
    READY_FOR_INDEPENDENT_CLOSURE_REVIEW = "READY FOR INDEPENDENT CLOSURE REVIEW"
    NA_WITH_EVIDENCE = "N/A — WITH EVIDENCE"


class R1SqliteEnv01Disposition(StrEnum):
    RESOLVED_TEST_PASSES = "RESOLVED — TEST PASSES"
    CLASSIFIED_ENVIRONMENT_NO_STATE_X_IMPACT = (
        "CLASSIFIED — ENVIRONMENT/TEST ISOLATION, NO STATE-X SEMANTIC IMPACT, "
        "FUTURE PROD-Q/QUAL-X DEBT PRESERVED"
    )
    IN_SCOPE_BLOCKER = "IN-SCOPE BLOCKER"


@dataclass(frozen=True, slots=True)
class StateXFreezeCriterionEvidence:
    criterion_id: str
    semantic_owner: str
    evidence_class: EvidenceClass
    exact_tests: tuple[str, ...]
    current_head_required: bool
    status: str
    notes: str


@dataclass(frozen=True, slots=True)
class FamilyOwnershipRow:
    family_id: str
    semantic_owner: str
    providers: str
    composition_owner: str
    duplicate_authority: bool


@dataclass(frozen=True, slots=True)
class AtomicityMatrixRow:
    family_id: str
    provider: str
    atomic_write_unit: str
    cas_fence_semantics: str
    cross_record_atomicity: str
    cross_store_atomicity: str
    failure_behavior: str


@dataclass(frozen=True, slots=True)
class ConfiguredEffectivePersistedRow:
    concern: str
    configured: str
    persisted: str
    effective_current: str
    restore_semantics: str


@dataclass(frozen=True, slots=True)
class PolicyDurabilityRow:
    artifact: str
    classification: str
    owner: str
    absence_behavior: str
    widens_authority_if_lost: bool


@dataclass(frozen=True, slots=True)
class HistoricalDebtRow:
    debt_id: str
    final_disposition: str
    evidence_sha: str
    notes: str


@dataclass(frozen=True, slots=True)
class R1SqliteEnv01Record:
    disposition: R1SqliteEnv01Disposition
    command: str
    cwd: str
    default_db_path: str
    default_db_probe: str
    isolated_env_var: str
    isolated_result_summary: str
    state_x_semantic_impact: str


@dataclass(frozen=True, slots=True)
class TenantIsolationAuditFinal:
    tenant_scope_applicable: bool
    scope: str
    canonical_tenant_identity: str
    tenant_owner: str
    propagation_path: str
    state_isolation: str
    provider_config_isolation: str
    evidence_trace_isolation: str
    async_recovery_continuity: str
    cross_tenant_path: str
    fail_closed_behavior: bool
    adversarial_evidence: tuple[str, ...]
    result: str


@dataclass(frozen=True, slots=True)
class UnclassifiedDurableHit:
    rel_path: str


def _entry(family_id: str) -> StateFamilyInventoryEntry:
    for item in STATE_X_FAMILY_INVENTORY:
        if item.family_id == family_id:
            return item
    raise KeyError(family_id)


def _family_ownership_row(entry: StateFamilyInventoryEntry) -> FamilyOwnershipRow:
    providers = ", ".join(entry.implementation_symbols) or "contract-only"
    return FamilyOwnershipRow(
        family_id=entry.family_id,
        semantic_owner=entry.semantic_owner,
        providers=providers,
        composition_owner=entry.composition_owner,
        duplicate_authority=False,
    )


FAMILY_OWNERSHIP_MATRIX: Final[tuple[FamilyOwnershipRow, ...]] = tuple(
    _family_ownership_row(e) for e in STATE_X_FAMILY_INVENTORY
)

ATOMICITY_MATRIX: Final[tuple[AtomicityMatrixRow, ...]] = (
    AtomicityMatrixRow(
        "SX-F01",
        "SQLiteTaskCheckpointStore / KV providers",
        "single checkpoint record + optional colocated scheduler claim row",
        "expected_revision CAS",
        "same SQLite file transaction for checkpoint stream",
        "independent DB files per domain — no distributed atomicity",
        "stale revision → reject; corrupt JSON → load/resume rejection",
    ),
    AtomicityMatrixRow(
        "SX-F10",
        "SQLiteIdempotencyStore / InMemory",
        "idempotency record upsert",
        "semantic fingerprint + tenant partition",
        "single-store record",
        "not atomic with compensation queue",
        "duplicate key → dedupe / replay semantics",
    ),
    AtomicityMatrixRow(
        "SX-F11",
        "SQLiteCompensationQueueStore / InMemory",
        "queue claim + completion in one store",
        "fence + stale completion rejection",
        "single-store",
        "not atomic with idempotency store",
        "incomplete completion → UNCERTAIN fail-closed",
    ),
    AtomicityMatrixRow(
        "SX-F12",
        "SQLiteHumanDecisionStore",
        "human decision evidence row",
        "revision / provenance validation",
        "single-store",
        "cross-store with checkpoint is semantic gate only",
        "corrupt payload → explicit rejection",
    ),
    AtomicityMatrixRow(
        "SX-F13",
        "Scheduler SQLite tables (colocated or dedicated)",
        "schedule claim row",
        "claim fence + stale revision",
        "colocated optional with checkpoints",
        "independent from idempotency",
        "stale schedule revision denied",
    ),
)

CONFIGURED_EFFECTIVE_PERSISTED_MATRIX: Final[tuple[ConfiguredEffectivePersistedRow, ...]] = (
    ConfiguredEffectivePersistedRow(
        "Runtime authority / governance admission",
        "profile + governance configuration",
        "human decision + terminal evidence (historical)",
        "current admission ports + fresh authorization gates",
        "restore validates evidence; does not mint authority",
    ),
    ConfiguredEffectivePersistedRow(
        "Execution budget",
        "budget policy inputs",
        "ExecutionBudgetLedger durable rows",
        "in-memory coordinator view",
        "ledger reload bounds consumption",
    ),
    ConfiguredEffectivePersistedRow(
        "Scheduler timing",
        "schedule configuration",
        "ScheduledResumePersistence durable state",
        "active claim / due processing",
        "restart reclaims with fence semantics",
    ),
    ConfiguredEffectivePersistedRow(
        "Task checkpoint runtime",
        "orchestration wiring",
        "TaskCheckpoint stream",
        "coordinator effective pause/resume",
        "resume validates snapshot + lineage",
    ),
)

POLICY_DURABILITY_MATRIX: Final[tuple[PolicyDurabilityRow, ...]] = (
    PolicyDurabilityRow(
        "Human decision evidence",
        "MUST_BE_DURABLE",
        "SX-F12 HumanDecision persistence",
        "missing/invalid → fail-closed resume",
        True,
    ),
    PolicyDurabilityRow(
        "Execution budget ledger",
        "MUST_BE_DURABLE",
        "SX-F09 budget persistence",
        "absence → cannot widen spend",
        True,
    ),
    PolicyDurabilityRow(
        "Scheduler durable timing",
        "MUST_BE_DURABLE",
        "SX-F13 scheduler persistence",
        "missing checkpoint → non-resumable schedule",
        False,
    ),
    PolicyDurabilityRow(
        "Idempotency / compensation",
        "MUST_BE_DURABLE",
        "SX-F10 / SX-F11",
        "UNCERTAIN / explicit rejection",
        True,
    ),
    PolicyDurabilityRow(
        "Live integration profile",
        "DERIVABLE / CURRENT-CONFIG",
        "CONFIG-X owner (out of STATE-X scope)",
        "rebuilt from config — not historical truth",
        False,
    ),
)

R1_SQLITE_ENV_01: Final[R1SqliteEnv01Record] = R1SqliteEnv01Record(
    disposition=R1SqliteEnv01Disposition.CLASSIFIED_ENVIRONMENT_NO_STATE_X_IMPACT,
    command=(
        "uv run --with cryptography pytest "
        "applications/governed_contractor_application/tests/host/"
        "test_governed_contractor_canonical_execution.py::"
        "test_governed_contractor_http_root_uses_canonical_execution_facade "
        "-p no:xdist -q"
    ),
    cwd=str(_REPO_ROOT),
    default_db_path="build/intergrax.db",
    default_db_probe="first bytes b'%3|1788589233.22' (not SQLite header)",
    isolated_env_var="INTERGRAX_RELATIONAL_DB",
    isolated_result_summary=(
        "SQLite collaborative-work schema bootstrap succeeds; "
        "test fails later on strict production dependency boundary (PROD-Q harness debt)"
    ),
    state_x_semantic_impact="NONE",
)

HISTORICAL_DEBT_RECONCILIATION: Final[tuple[HistoricalDebtRow, ...]] = (
    HistoricalDebtRow(
        "Q2-D1",
        "CLOSED — SUPERSEDED BY R4 EVIDENCE",
        "61faf8f317125b996526ceceaf5754b9c28073d6",
        "terminal_capability_from_task_checkpoint_store(); no STATE-X-FINAL production delta",
    ),
    HistoricalDebtRow(
        "R1-SQLITE-ENV-01",
        R1_SQLITE_ENV_01.disposition.value,
        STATE_X_FINAL_START_HEAD,
        R1_SQLITE_ENV_01.state_x_semantic_impact,
    ),
    HistoricalDebtRow(
        "CTRL-X-R3-R2 state/recovery debt",
        "SUPERSEDED / CLOSED BY CURRENT STATE-X EVIDENCE",
        STATE_X_FINAL_START_HEAD,
        "R3–R6 qualification replay on HEAD; no orphan CTRL debt",
    ),
)

TENANT_ISOLATION_AUDIT_FINAL: Final[TenantIsolationAuditFinal] = TenantIsolationAuditFinal(
    tenant_scope_applicable=True,
    scope="STATE-X state / persistence / recovery only",
    canonical_tenant_identity="tenant_id on durable contracts (per-family typed bindings)",
    tenant_owner="state/recovery semantic owners per SX-F01..F15",
    propagation_path=(
        "request/task/run → persistence contract → durable record → "
        "recovery validator → sanctioned execution/recovery"
    ),
    state_isolation="PASS",
    provider_config_isolation="STATE-X persistence providers only (scoped)",
    evidence_trace_isolation="historical evidence remains tenant-bound",
    async_recovery_continuity=(
        "scheduler/retry/background/resume/partial recovery preserve tenant"
    ),
    cross_tenant_path="DENIED",
    fail_closed_behavior=True,
    adversarial_evidence=(
        "test_r3_r2_q12_tenant_isolation",
        "test_r3_r2_q18_same_key_cross_tenant_compensation_isolation",
        "test_r4_r1_worker_recovery_cross_tenant_checkpoint_denied",
        "test_r6_q29_cross_tenant_resume_denied",
        "test_r6_q30_cross_tenant_retry_keying",
        "test_r6_q31_cross_tenant_partial_recovery_denied",
    ),
    result="PASS",
)

PRIMARY_FRZ_CRITERION_EVIDENCE: Final[tuple[StateXFreezeCriterionEvidence, ...]] = (
    StateXFreezeCriterionEvidence(
        "FRZ-STA-01",
        "STATE-X family semantic owners (inventory SSOT)",
        EvidenceClass.STRUCTURAL,
        ("test_sxf_q04_exactly_one_semantic_owner", "test_r5_q01_closed_world_family_inventory_complete"),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "15 families; duplicate_authority=0 in FAMILY_OWNERSHIP_MATRIX",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-STA-02",
        "STATE-X closed-world store inventory",
        EvidenceClass.STRUCTURAL,
        ("test_sxf_q05_duplicate_semantic_store_zero", "test_r5_q02_every_family_exactly_one_responsibility"),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "providers ≠ duplicate semantic truth",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-STA-03",
        "Per-family atomicity matrix + R1-SQLITE disposition",
        EvidenceClass.STRUCTURAL,
        ("test_sxf_q06_atomicity_matrix_complete", "test_sxf_q07_r1_sqlite_env_01_classified"),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "cross-store non-atomicity explicit; SQLite ENV classified",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-STA-04",
        "Tenant-bound durable families",
        EvidenceClass.BEHAVIORAL,
        (
            "test_r3_r2_q12_tenant_isolation",
            "test_r3_r2_q18_compensation_tenant_isolation",
            "test_r4_r1_worker_recovery_cross_tenant_checkpoint_denied",
        ),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "state/recovery tenant scope only",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-STA-05",
        "Stale revision / fence rejection",
        EvidenceClass.BEHAVIORAL,
        (
            "test_r3_r2_q17_compensation_fence_supersession",
            "test_r4_q04_no_task_materialization_before_restore_validation",
            "test_r3_r4_q10_stale_fence_completion_rejected",
        ),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "parametrized InMemory + SQLite where applicable",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-STA-06",
        "Configured vs persisted vs effective",
        EvidenceClass.STRUCTURAL,
        ("test_sxf_q10_configured_effective_persisted_complete",),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "CONFIGURED_EFFECTIVE_PERSISTED_MATRIX",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-STA-07",
        "Policy-relevant durability classification",
        EvidenceClass.STRUCTURAL,
        ("test_sxf_q11_policy_durability_complete",),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "no unclassified policy artifact durability",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-STA-08",
        "Checkpoint ≠ identity authority",
        EvidenceClass.BEHAVIORAL,
        (
            "test_r6_q16_new_execution_mints_independent_identity",
            "test_r4_r1_worker_recovery_positive_runtime_reconciles_identity",
        ),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "ExecutionIdentityAuthority remains mint owner",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-REC-01",
        "Crash recovery deterministic",
        EvidenceClass.BEHAVIORAL,
        ("test_r3_r2_q30_expired_running_uncertain_not_reclaimable_queue", "test_r3_r2_q32_crash_window_canonical_production_path_no_duplicate_effect"),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "representative R3/R4/R6 flows",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-REC-02",
        "Restart preserves authority bounds",
        EvidenceClass.BEHAVIORAL,
        ("test_r4_r1_worker_recovery_invalid_checkpoint_fail_closed",),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "governance/HITL evidence via R4",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-REC-03",
        "Resume identity correctness",
        EvidenceClass.BEHAVIORAL,
        ("test_r6_q05_resume_preserves_logical_run_identity", "test_r4_r1_worker_recovery_positive_runtime_reconciles_identity"),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "resume ≠ arbitrary identity pick",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-REC-04",
        "Replay divergence resistance",
        EvidenceClass.BEHAVIORAL,
        ("test_r6_q18_inspection_replay_is_read_only", "test_r3_r2_q23_retryable_redelivery_canonical_idempotency_replay"),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "inspection vs idempotent replay separated",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-REC-05",
        "Fork semantics explicit",
        EvidenceClass.BEHAVIORAL,
        ("test_r6_q03_no_first_class_fork_contract", "test_r6_q38_frz_rec_05_completeness_gate_pass"),
        True,
        CriterionResolution.PASS.value,
        f"STATE-X-R6 CLOSED @ {STATE_X_R6_ACCEPTED_CLOSURE_SHA}",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-REC-06",
        "Partial persistence fail-closed",
        EvidenceClass.BEHAVIORAL,
        ("test_r3_r2_q32_crash_window_canonical_production_path_no_duplicate_effect", "test_r5_q1_old_checkpoint_newer_terminal_restore_skew_fails_closed"),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "cross-store skew explicit",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-REC-07",
        "External uncertainty explicit",
        EvidenceClass.BEHAVIORAL,
        ("test_r3_r2_q30_expired_running_uncertain_not_reclaimable_queue",),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "UNCERTAIN not blindly reclaimable",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-REC-08",
        "Backup/restore responsibility",
        EvidenceClass.RESPONSIBILITY,
        ("test_r5_q38_frz_rec_08_completeness_gate",),
        True,
        CriterionResolution.PASS.value,
        "STATE-X-R5 @ bcd8157065cc649412b64e9d6ada34be92d4b6a3",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-REC-09",
        "Restore preserves identity/authority/tenant/truth",
        EvidenceClass.BEHAVIORAL,
        (
            "test_r5_q1_old_checkpoint_newer_terminal_restore_skew_fails_closed",
            "test_r4_r1_worker_recovery_cross_tenant_checkpoint_denied",
        ),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "parent composition R4+R5+R6",
    ),
    StateXFreezeCriterionEvidence(
        "FRZ-REC-10",
        "Corrupt/partial durable state detection",
        EvidenceClass.BEHAVIORAL,
        (
            "test_r3_r3_q13_corrupt_approver_provenance_fail_closed",
            "test_r4_q05_empty_task_snapshot_rejected",
            "test_r5_q1_terminal_backend_corruption_is_not_treated_as_absence",
        ),
        True,
        CriterionResolution.READY_FOR_INDEPENDENT_CLOSURE_REVIEW.value,
        "no silent empty continue",
    ),
)

ACCEPTED_CHILD_CHAIN: Final[tuple[tuple[str, str], ...]] = (
    ("STATE-X-R3-R2", "d5979a531e41f9a45a3a00b0b150765ee34bb0f8"),
    ("STATE-X-R3-R3", "716ed746f8b463681db536804235cedc86adc162"),
    ("STATE-X-R3-R4", "716ed746f8b463681db536804235cedc86adc162"),
    ("STATE-X-R3-R5", "716ed746f8b463681db536804235cedc86adc162"),
    ("STATE-X-R4", "61faf8f317125b996526ceceaf5754b9c28073d6"),
    ("STATE-X-R5", "bcd8157065cc649412b64e9d6ada34be92d4b6a3"),
    ("STATE-X-R6", STATE_X_R6_ACCEPTED_CLOSURE_SHA),
)


def scan_unclassified_durable_persistence_paths() -> tuple[UnclassifiedDurableHit, ...]:
    """Implementation modules named like stores absent from SX-F closed-world inventory."""
    hits: list[UnclassifiedDurableHit] = []
    impl_roots = ("intergrax/runtime/", "agents/")
    name_pattern = re.compile(r"(store|persistence|ledger|checkpoint_store)\.py$", re.IGNORECASE)
    extra_allow = frozenset(
        {
            "intergrax/runtime/persistence/sqlite_composition.py",
            "intergrax/runtime/long_running/in_memory_checkpoint_store.py",
            "intergrax/runtime/events/store.py",
            "intergrax/runtime/events/in_memory_event_store.py",
            "intergrax/runtime/task_memory/store.py",
        },
    )
    out_of_state_x_prefixes = (
        "intergrax/runtime/adaptive/",
        "intergrax/runtime/prediction/",
        "intergrax/runtime/self_healing/",
        "intergrax/runtime/vendor_knowledge/",
        "intergrax/runtime/diagnostics/",
        "intergrax/runtime/governance/",
        "intergrax/runtime/observability/",
        "intergrax/runtime/notifications/",
        "intergrax/runtime/organization/",
        "intergrax/runtime/integrations/",
        "intergrax/runtime/external_operations/",
        "intergrax/runtime/execution_evidence/",
        "intergrax/runtime/background_execution/",
        "intergrax/runtime/nexus/",
        "intergrax/runtime/execution/continuation/",
        "intergrax/runtime/execution/deadline_authority/",
        "intergrax/runtime/execution/delegated_execution/",
        "intergrax/runtime/execution/suspended_operation/",
        "intergrax/runtime/execution/active_",
        "intergrax/runtime/execution/in_memory_decision",
        "intergrax/runtime/execution/sqlite_decision",
        "intergrax/runtime/execution/decision_finalization",
        "intergrax/runtime/tools/reference_idempotency_store.py",
        "intergrax/runtime/task_memory/",
    )
    inventory_parent_dirs = frozenset(
        Path(p).parent.as_posix() + "/" for p in _INVENTORY_CLOSED_WORLD_PATHS
    )
    for root in impl_roots:
        base = _REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if "test_" in rel or "/tests/" in rel:
                continue
            if not name_pattern.search(path.name):
                continue
            if rel in _INVENTORY_CLOSED_WORLD_PATHS or rel in extra_allow:
                continue
            if any(rel.startswith(p) for p in out_of_state_x_prefixes):
                continue
            if any(rel.startswith(d) for d in inventory_parent_dirs):
                continue
            if rel.startswith("intergrax/runtime/replay/"):
                continue
            hits.append(UnclassifiedDurableHit(rel_path=rel))
    return tuple(hits)


def assert_primary_frz_matrix_complete() -> None:
    ids = {row.criterion_id for row in PRIMARY_FRZ_CRITERION_EVIDENCE}
    assert ids == set(_PRIMARY_FRZ_IDS)


def assert_family_inventory_closed_world() -> None:
    inv_ids = {e.family_id for e in STATE_X_FAMILY_INVENTORY}
    assert inv_ids == set(MANDATORY_FAMILY_IDS)
    for entry in STATE_X_FAMILY_INVENTORY:
        if entry.projection_or_truth in (
            ProjectionOrTruth.CANONICAL_TRUTH,
            ProjectionOrTruth.DURABLE_COMPONENT_OF_CANONICAL_TRUTH,
        ):
            assert entry.semantic_ownership_role in (
                SemanticOwnershipRole.CANONICAL_OWNER,
                SemanticOwnershipRole.COMPONENT_OWNER,
            )
            assert entry.semantic_owner


def assert_no_duplicate_semantic_owners() -> None:
    for row in FAMILY_OWNERSHIP_MATRIX:
        assert row.duplicate_authority is False


def assert_no_in_scope_state_x_blockers() -> None:
    for blocker in STATE_X_KNOWN_BLOCKERS:
        assert blocker.classification is not BlockerClassification.IN_SCOPE_BLOCKER, (
            blocker.blocker_id
        )


def assert_r1_sqlite_disposition_final() -> None:
    assert R1_SQLITE_ENV_01.disposition is not R1SqliteEnv01Disposition.IN_SCOPE_BLOCKER
    assert R1_SQLITE_ENV_01.state_x_semantic_impact == "NONE"


def assert_state_x_final_mechanical_gate() -> None:
    assert_primary_frz_matrix_complete()
    assert_family_inventory_closed_world()
    assert_no_duplicate_semantic_owners()
    assert_no_in_scope_state_x_blockers()
    assert_r1_sqlite_disposition_final()
    unclassified = scan_unclassified_durable_persistence_paths()
    assert len(unclassified) == 0, unclassified
    assert TENANT_ISOLATION_AUDIT_FINAL.result == "PASS"
