# © Artur Czarnecki. All rights reserved.

"""STATE-X-R6 — resume/retry/replay/partial-recovery & fork semantics support."""

from __future__ import annotations

import ast
import importlib
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

STATE_X_R6_PRE_AUDIT_HEAD: Final[str] = "bcd8157065cc649412b64e9d6ada34be92d4b6a3"

STATE_X_R5_ACCEPTED_CLOSURE_SHA: Final[str] = "bcd8157065cc649412b64e9d6ada34be92d4b6a3"

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SCAN_ROOTS: Final[tuple[str, ...]] = (
    "intergrax/contracts",
    "intergrax/runtime",
    "agents",
)

_FORBIDDEN_SEMANTIC_TOKENS: Final[frozenset[str]] = frozenset(
    {"TBD", "MAYBE", "PARTIAL", "UNKNOWN"},
)

_FORK_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(
    r"(^fork_|_fork$|_fork_|Fork[A-Z]|fork_run|fork_execution|clone_execution|branch_execution|ExecutionFork)",
)

_FORK_LIKE_CLASSIFIED_ALLOWLIST: Final[frozenset[str]] = frozenset()


class RecoveryOperationKind(StrEnum):
    RESUME = "resume"
    RETRY = "retry"
    NEW_EXECUTION = "new_execution"
    INSPECTION_REPLAY = "inspection_replay"
    IDEMPOTENT_RESULT_REPLAY = "idempotent_result_replay"
    PARTIAL_RECOVERY = "partial_recovery"
    FIRST_CLASS_FORK = "first_class_fork"


class ForkSupportStatus(StrEnum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"


@dataclass(frozen=True, slots=True)
class RecoveryOperationSemanticsRow:
    kind: RecoveryOperationKind
    supported: bool
    fork_status: ForkSupportStatus | None
    canonical_owner: str | None
    task_semantics: str
    tenant_semantics: str
    run_semantics: str
    attempt_semantics: str
    root_execution_semantics: str
    child_execution_semantics: str
    lineage_semantics: str
    authority_semantics: str
    side_effect_semantics: str
    durability_semantics: str


@dataclass(frozen=True, slots=True)
class RecoveryAuthorityMatrixRow:
    operation: RecoveryOperationKind
    historical_authority_reused: bool
    fresh_authority_required: bool
    can_widen_authority: bool
    notes: str


@dataclass(frozen=True, slots=True)
class RecoverySideEffectMatrixRow:
    operation: RecoveryOperationKind
    may_execute_external_effects: bool
    can_replay_completed_effects: bool
    requires_idempotency: bool
    can_duplicate_physical_effect: bool
    notes: str


@dataclass(frozen=True, slots=True)
class ForkLikeSymbolHit:
    rel_path: str
    symbol_kind: str
    name: str
    lineno: int


@dataclass(frozen=True, slots=True)
class R6BehavioralEvidenceRef:
    evidence_id: str
    module: str
    test_name: str


RECOVERY_OPERATION_MATRIX: Final[tuple[RecoveryOperationSemanticsRow, ...]] = (
    RecoveryOperationSemanticsRow(
        kind=RecoveryOperationKind.RESUME,
        supported=True,
        fork_status=None,
        canonical_owner=(
            "LongRunningCoordinator + checkpoint_resume_validation + "
            "build_execution_tree_resume_plan / checkpoint_builder"
        ),
        task_semantics="Same task_id as checkpoint; resume binds to target task",
        tenant_semantics="Same tenant_id; cross-tenant materialization rejected",
        run_semantics="Same logical run_id as checkpoint historical run",
        attempt_semantics="Active attempt_id may differ from historical checkpoint attempt (sanctioned resume context)",
        root_execution_semantics="New active root execution_id; historical root retained as lineage evidence",
        child_execution_semantics="Completed safe nodes adopted/skipped; resume candidates get new active execution entries",
        lineage_semantics="resumed_from_execution_id + historical execution tree preserved; no identity overwrite",
        authority_semantics="Historical checkpoint is evidence only; current allow via restore/Governance path; no widening",
        side_effect_semantics="May continue execution under normal idempotency boundaries; completed nodes not re-executed",
        durability_semantics="Requires durable checkpoint (+ policy-mandated companion stores for production resume)",
    ),
    RecoveryOperationSemanticsRow(
        kind=RecoveryOperationKind.RETRY,
        supported=True,
        fork_status=None,
        canonical_owner="ExecutionAttemptRetryService + AttemptLifecycleService + mint_retry_attempt_id",
        task_semantics="Same task_id",
        tenant_semantics="Same tenant_id; lifecycle keyed by tenant+run",
        run_semantics="Same run_id; retry must not mint new run",
        attempt_semantics="New attempt_id; previous attempt sealed RETRY_SUPERSEDED when lineage enabled",
        root_execution_semantics="Rebind active execution identity to new attempt context",
        child_execution_semantics="New attempt generation; no sibling branch from stale attempt",
        lineage_semantics="Prior attempt closed superseded; exactly one active attempt per tenant+run",
        authority_semantics="Retry transition does not mint new execution authority; bounded existing authority",
        side_effect_semantics="May re-enter execution; idempotency protects external effects",
        durability_semantics="Production retry requires durable AttemptLifecycleStore (durability_policy gate)",
    ),
    RecoveryOperationSemanticsRow(
        kind=RecoveryOperationKind.NEW_EXECUTION,
        supported=True,
        fork_status=None,
        canonical_owner="DefaultExecutionIdentityAuthority / ExecutionIdentityAuthorityPort + sanctioned intake",
        task_semantics="New or explicit task scope via intake",
        tenant_semantics="Tenant from sanctioned intake; no historical tenant switch",
        run_semantics="New run_id via mint_run_identity",
        attempt_semantics="New attempt_id with new run",
        root_execution_semantics="New root execution_id via mint_root_execution_identity",
        child_execution_semantics="Fresh child IDs minted under new tree",
        lineage_semantics="Independent lineage; not represented as retry/resume of prior run",
        authority_semantics="Fresh authority through sanctioned intake/Governance; no historical inheritance",
        side_effect_semantics="Normal execution effects with idempotency as configured",
        durability_semantics="Per configured persistence mode",
    ),
    RecoveryOperationSemanticsRow(
        kind=RecoveryOperationKind.INSPECTION_REPLAY,
        supported=True,
        fork_status=None,
        canonical_owner="ReplayService + ReplayEngine",
        task_semantics="Read-only query within tenant/run scope",
        tenant_semantics="Tenant-scoped inspect_run(tenant_id, run_id)",
        run_semantics="Historical run_id reconstructed; no new run minted",
        attempt_semantics="No attempt transition; evidence only",
        root_execution_semantics="No new root execution; reconstruction only",
        child_execution_semantics="No child execution mint",
        lineage_semantics="Observed historical lineage only",
        authority_semantics="No execution authority; non-authoritative observation",
        side_effect_semantics="No external effects; no HostTaskExecutionPort / tool invoker",
        durability_semantics="Read-only against durable evidence stores",
    ),
    RecoveryOperationSemanticsRow(
        kind=RecoveryOperationKind.IDEMPOTENT_RESULT_REPLAY,
        supported=True,
        fork_status=None,
        canonical_owner="IdempotencyPreEffectCoordinator + IdempotencyStore",
        task_semantics="Bound to original effect scope/key",
        tenant_semantics="Tenant partition on idempotency key",
        run_semantics="Does not create new canonical run truth",
        attempt_semantics="No new attempt; returns prior completed disposition",
        root_execution_semantics="N/A — result reuse not new execution tree",
        child_execution_semantics="N/A",
        lineage_semantics="Reuses durable completed result record",
        authority_semantics="No new permission; existing effect semantics",
        side_effect_semantics="No second physical effect; handler not invoked again",
        durability_semantics="Requires durable idempotency record for cross-restart replay",
    ),
    RecoveryOperationSemanticsRow(
        kind=RecoveryOperationKind.PARTIAL_RECOVERY,
        supported=True,
        fork_status=None,
        canonical_owner="FanOutPartialRecoveryService + RecoveryAdmissionPort + TaskCheckpointPersistence",
        task_semantics="Same task as checkpoint topology",
        tenant_semantics="Checkpoint tenant binding enforced",
        run_semantics="Same logical run_id as checkpoint",
        attempt_semantics="source_attempt_id must match checkpoint.runtime.attempt_id",
        root_execution_semantics="root_execution_id + topology_execution_id + fan_out_id bound",
        child_execution_semantics="Only failed slot recovered; successful siblings preserved",
        lineage_semantics="Recovery evidence from checkpoint revision; not a topology fork",
        authority_semantics="RecoveryAdmission permit for specific operation only; no fork authority",
        side_effect_semantics="Selected slot work only; idempotency/authority as normal execution",
        durability_semantics="Checkpoint revision CAS; stale bindings fail closed",
    ),
    RecoveryOperationSemanticsRow(
        kind=RecoveryOperationKind.FIRST_CLASS_FORK,
        supported=False,
        fork_status=ForkSupportStatus.NOT_SUPPORTED,
        canonical_owner=None,
        task_semantics="NOT SUPPORTED — no sanctioned historical→independent-branch operation",
        tenant_semantics="NOT SUPPORTED",
        run_semantics="NOT SUPPORTED",
        attempt_semantics="NOT SUPPORTED",
        root_execution_semantics="NOT SUPPORTED",
        child_execution_semantics="NOT SUPPORTED",
        lineage_semantics="NOT SUPPORTED — no authority inheritance into independent branch",
        authority_semantics="NOT SUPPORTED — no historical authority inheritance contract",
        side_effect_semantics="NOT SUPPORTED",
        durability_semantics="NOT SUPPORTED",
    ),
)

RECOVERY_AUTHORITY_MATRIX: Final[tuple[RecoveryAuthorityMatrixRow, ...]] = (
    RecoveryAuthorityMatrixRow(
        RecoveryOperationKind.RESUME,
        historical_authority_reused=False,
        fresh_authority_required=True,
        can_widen_authority=False,
        notes="Checkpoint/historical state is binding evidence; Governance/restore path required",
    ),
    RecoveryAuthorityMatrixRow(
        RecoveryOperationKind.RETRY,
        historical_authority_reused=False,
        fresh_authority_required=False,
        can_widen_authority=False,
        notes="Existing bounded execution authority; transition does not widen",
    ),
    RecoveryAuthorityMatrixRow(
        RecoveryOperationKind.NEW_EXECUTION,
        historical_authority_reused=False,
        fresh_authority_required=True,
        can_widen_authority=False,
        notes="Sanctioned intake authority only",
    ),
    RecoveryAuthorityMatrixRow(
        RecoveryOperationKind.INSPECTION_REPLAY,
        historical_authority_reused=False,
        fresh_authority_required=False,
        can_widen_authority=False,
        notes="No execution authority",
    ),
    RecoveryAuthorityMatrixRow(
        RecoveryOperationKind.IDEMPOTENT_RESULT_REPLAY,
        historical_authority_reused=False,
        fresh_authority_required=False,
        can_widen_authority=False,
        notes="Reuses prior effect record",
    ),
    RecoveryAuthorityMatrixRow(
        RecoveryOperationKind.PARTIAL_RECOVERY,
        historical_authority_reused=False,
        fresh_authority_required=True,
        can_widen_authority=False,
        notes="RecoveryAdmission + Governance as applicable; scope-bound",
    ),
    RecoveryAuthorityMatrixRow(
        RecoveryOperationKind.FIRST_CLASS_FORK,
        historical_authority_reused=False,
        fresh_authority_required=False,
        can_widen_authority=False,
        notes="NOT SUPPORTED",
    ),
)

RECOVERY_SIDE_EFFECT_MATRIX: Final[tuple[RecoverySideEffectMatrixRow, ...]] = (
    RecoverySideEffectMatrixRow(
        RecoveryOperationKind.RESUME,
        may_execute_external_effects=True,
        can_replay_completed_effects=False,
        requires_idempotency=True,
        can_duplicate_physical_effect=False,
        notes="Completed nodes skipped; continuing nodes use normal idempotency",
    ),
    RecoverySideEffectMatrixRow(
        RecoveryOperationKind.RETRY,
        may_execute_external_effects=True,
        can_replay_completed_effects=False,
        requires_idempotency=True,
        can_duplicate_physical_effect=False,
        notes="Policy may block retry (budget_exhausted, deny kinds)",
    ),
    RecoverySideEffectMatrixRow(
        RecoveryOperationKind.NEW_EXECUTION,
        may_execute_external_effects=True,
        can_replay_completed_effects=False,
        requires_idempotency=True,
        can_duplicate_physical_effect=True,
        notes="Independent effect namespace from prior runs",
    ),
    RecoverySideEffectMatrixRow(
        RecoveryOperationKind.INSPECTION_REPLAY,
        may_execute_external_effects=False,
        can_replay_completed_effects=False,
        requires_idempotency=False,
        can_duplicate_physical_effect=False,
        notes="Read-only reconstruction",
    ),
    RecoverySideEffectMatrixRow(
        RecoveryOperationKind.IDEMPOTENT_RESULT_REPLAY,
        may_execute_external_effects=False,
        can_replay_completed_effects=True,
        requires_idempotency=True,
        can_duplicate_physical_effect=False,
        notes="Second invoke returns stored completed result",
    ),
    RecoverySideEffectMatrixRow(
        RecoveryOperationKind.PARTIAL_RECOVERY,
        may_execute_external_effects=True,
        can_replay_completed_effects=False,
        requires_idempotency=True,
        can_duplicate_physical_effect=False,
        notes="Only failed slot work",
    ),
    RecoverySideEffectMatrixRow(
        RecoveryOperationKind.FIRST_CLASS_FORK,
        may_execute_external_effects=False,
        can_replay_completed_effects=False,
        requires_idempotency=False,
        can_duplicate_physical_effect=False,
        notes="NOT SUPPORTED",
    ),
)

R6_FRZ_REC_05_BEHAVIORAL_EVIDENCE: Final[tuple[R6BehavioralEvidenceRef, ...]] = (
    R6BehavioralEvidenceRef(
        "resume_tree_plan",
        "tests.qualification.state_x._r6_recovery_branching_qualification_tests",
        "test_r6_q05_resume_preserves_logical_run_identity",
    ),
    R6BehavioralEvidenceRef(
        "retry_transition",
        "tests.qualification.state_x._r6_recovery_branching_qualification_tests",
        "test_r6_q10_retry_preserves_run_id",
    ),
    R6BehavioralEvidenceRef(
        "retry_stale",
        "tests.qualification.state_x._r6_recovery_branching_qualification_tests",
        "test_r6_q13_stale_retry_cannot_create_sibling_attempt",
    ),
    R6BehavioralEvidenceRef(
        "new_execution_mint",
        "tests.qualification.state_x._r6_recovery_branching_qualification_tests",
        "test_r6_q16_new_execution_mints_independent_identity",
    ),
    R6BehavioralEvidenceRef(
        "inspection_replay",
        "tests.qualification.state_x._r6_recovery_branching_qualification_tests",
        "test_r6_q18_inspection_replay_is_read_only",
    ),
    R6BehavioralEvidenceRef(
        "idempotent_replay",
        "tests.qualification.state_x._r3_r2_qualification_tests",
        "test_r3_r2_q23_retryable_redelivery_canonical_idempotency_replay",
    ),
    R6BehavioralEvidenceRef(
        "partial_recovery_stale_attempt",
        "tests.unit.runtime.architecture.test_npsc5e_r3_final_child_fanout_partial_recovery_qualification",
        "test_final_wrong_root_topology_slot_attempt_revision_blocked",
    ),
    R6BehavioralEvidenceRef(
        "durable_retry_gate",
        "tests.unit.runtime.execution.test_attempt_lifecycle_durability_gate",
        "test_composition_rejects_production_retry_with_in_memory_store",
    ),
)


def _matrix_by_kind(kind: RecoveryOperationKind) -> RecoveryOperationSemanticsRow:
    for row in RECOVERY_OPERATION_MATRIX:
        if row.kind is kind:
            return row
    raise KeyError(kind)


def scan_execution_fork_like_symbols() -> tuple[ForkLikeSymbolHit, ...]:
    hits: list[ForkLikeSymbolHit] = []
    for root_rel in _SCAN_ROOTS:
        root = _REPO_ROOT / root_rel
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = node.name
                    if _FORK_SYMBOL_RE.search(name):
                        kind = type(node).__name__
                        key = f"{rel}:{kind}:{name}"
                        if key not in _FORK_LIKE_CLASSIFIED_ALLOWLIST:
                            hits.append(
                                ForkLikeSymbolHit(
                                    rel_path=rel,
                                    symbol_kind=kind,
                                    name=name,
                                    lineno=node.lineno,
                                ),
                            )
    return tuple(hits)


def assert_frz_rec_05_r6_completeness() -> None:
    kinds = {row.kind for row in RECOVERY_OPERATION_MATRIX}
    assert kinds == set(RecoveryOperationKind)
    for row in RECOVERY_OPERATION_MATRIX:
        blob = " ".join(
            (
                row.task_semantics,
                row.tenant_semantics,
                row.run_semantics,
                row.attempt_semantics,
                row.root_execution_semantics,
                row.lineage_semantics,
                row.authority_semantics,
                row.side_effect_semantics,
                row.durability_semantics,
            ),
        ).upper()
        for token in _FORBIDDEN_SEMANTIC_TOKENS:
            assert token not in blob, f"{row.kind} contains forbidden token {token}"
        if row.supported:
            assert row.canonical_owner
            assert row.fork_status is None
        else:
            assert row.kind is RecoveryOperationKind.FIRST_CLASS_FORK
            assert row.fork_status is ForkSupportStatus.NOT_SUPPORTED
            assert row.canonical_owner is None
    fork_row = _matrix_by_kind(RecoveryOperationKind.FIRST_CLASS_FORK)
    assert fork_row.supported is False
    assert fork_row.fork_status is ForkSupportStatus.NOT_SUPPORTED
    assert len(RECOVERY_AUTHORITY_MATRIX) == len(RECOVERY_OPERATION_MATRIX)
    assert len(RECOVERY_SIDE_EFFECT_MATRIX) == len(RECOVERY_OPERATION_MATRIX)
    unclassified = scan_execution_fork_like_symbols()
    assert len(unclassified) == 0, unclassified


def assert_r6_behavioral_evidence_registered() -> None:
    for ref in R6_FRZ_REC_05_BEHAVIORAL_EVIDENCE:
        mod = importlib.import_module(ref.module)
        fn = getattr(mod, ref.test_name, None)
        assert callable(fn), f"missing behavioral evidence {ref.evidence_id}: {ref.test_name}"


@dataclass(frozen=True, slots=True)
class TenantIsolationAuditR6:
    tenant_scope_applicable: bool
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


TENANT_ISOLATION_AUDIT_R6: Final[TenantIsolationAuditR6] = TenantIsolationAuditR6(
    tenant_scope_applicable=True,
    canonical_tenant_identity="Task / Run / checkpoint / recovery request contracts (tenant_id)",
    tenant_owner="Execution + long-running + partial-recovery canonical owners",
    propagation_path=(
        "operation owner → identity validation → recovery/transition → execution or read-only result"
    ),
    state_isolation="Historical durable state remains tenant-bound",
    provider_config_isolation="No operation switches tenant via restored/replayed state",
    evidence_trace_isolation="Inspection replay tenant-scoped; non-authoritative",
    async_recovery_continuity="Resume/retry/partial recovery preserve tenant",
    cross_tenant_path="denied",
    fail_closed_behavior=True,
    adversarial_evidence=(
        "test_r6_q29_cross_tenant_resume_denied",
        "test_r6_q30_cross_tenant_retry_keying",
        "test_r6_q31_cross_tenant_partial_recovery_denied",
        "test_r3_r2_q12_idempotency_tenant_separation (replay)",
    ),
    result="PASS",
)
