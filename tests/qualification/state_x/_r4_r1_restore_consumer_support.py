# © Artur Czarnecki. All rights reserved.

"""STATE-X-R4-R1 — closed-world TaskCheckpoint consumer convergence support."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

STATE_X_R4_R1_PRE_AUDIT_HEAD: Final[str] = "6a8252548a8fb81830afac8534acbd5183f9df95"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INTERGRAX = _REPO_ROOT / "intergrax"

ConsumerCategory = Literal[
    "RESTORE_EXECUTION",
    "REENTRY",
    "STATE_PROJECTION",
    "GOVERNANCE_PROJECTION",
    "CONTROL_PLANE_EVIDENCE",
    "DIAGNOSTICS_ONLY",
    "PERSISTENCE_ONLY",
    "TEST/LAB_ONLY",
]


@dataclass(frozen=True, slots=True)
class TaskCheckpointConsumerRecord:
    file: str
    symbol: str
    category: ConsumerCategory
    canonical_validator: str
    mutates_live_state: bool
    can_influence_execution: bool
    can_influence_governance: bool
    tenant_binding: str
    authority_handling: str
    qualification_test: str


R4_R1_TASK_CHECKPOINT_CONSUMERS: Final[tuple[TaskCheckpointConsumerRecord, ...]] = (
    TaskCheckpointConsumerRecord(
        file="intergrax/runtime/long_running/coordinator.py",
        symbol="LongRunningCoordinator.restore_if_resuming",
        category="RESTORE_EXECUTION",
        canonical_validator="assert_checkpoint_resume_eligible",
        mutates_live_state=True,
        can_influence_execution=True,
        can_influence_governance=True,
        tenant_binding="target_tenant_id parameter",
        authority_handling="current_task + policy reconciliation",
        qualification_test="test_r4_r1_q20_execution_tree_identity_mismatch_rejected",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/runtime/long_running/resume_planner.py",
        symbol="build_checkpoint_resume_task / _base_resume_task",
        category="RESTORE_EXECUTION",
        canonical_validator="assert_checkpoint_resume_materialization_eligible",
        mutates_live_state=False,
        can_influence_execution=True,
        can_influence_governance=False,
        tenant_binding="target_tenant_id",
        authority_handling="materialization only; authority narrowed downstream",
        qualification_test="test_r4_q36_restore_semantics_not_sqlite_specific",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/runtime/long_running/scheduler.py",
        symbol="LongRunningScheduler._inspect_timeout",
        category="RESTORE_EXECUTION",
        canonical_validator="assert_checkpoint_resume_materialization_eligible via _can_materialize_checkpoint",
        mutates_live_state=False,
        can_influence_execution=True,
        can_influence_governance=False,
        tenant_binding="schedule entry tenant",
        authority_handling="WHEN-only; no authority minting",
        qualification_test="test_r4_r1_q24_scheduler_timeout_when_only_replay",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/applications/_shared/task_control.py",
        symbol="governed_resume_checkpoint_task",
        category="RESTORE_EXECUTION",
        canonical_validator="assert_checkpoint_resume_materialization_eligible",
        mutates_live_state=True,
        can_influence_execution=True,
        can_influence_governance=True,
        tenant_binding="request tenant_id",
        authority_handling="ControlPlane mutation + stale re-read",
        qualification_test="test_r4_r1_q26_post_authorization_stale_checkpoint_denied",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/runtime/task/nexus_worker_execution.py",
        symbol="NexusWorkerRuntime._reconcile_resume_identity",
        category="RESTORE_EXECUTION",
        canonical_validator="LongRunningCoordinator.restore_if_resuming",
        mutates_live_state=True,
        can_influence_execution=True,
        can_influence_governance=False,
        tenant_binding="execution_identity.tenant_id preserved",
        authority_handling="checkpoint run/attempt after canonical restore",
        qualification_test="test_r4_r1_q1_worker_recovery_positive_runtime_reconciles_identity",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/runtime/execution/suspended_operation/reentry_coordinator.py",
        symbol="_load_task_from_durable_checkpoint",
        category="REENTRY",
        canonical_validator="validated_task_snapshot_from_checkpoint",
        mutates_live_state=False,
        can_influence_execution=True,
        can_influence_governance=True,
        tenant_binding="decoded_payload tenant + task_id",
        authority_handling="governance grant evidence only",
        qualification_test="test_r4_r1_q12_suspended_reentry_uses_canonical_validation",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/runtime/human/agent_governance_pause_projection.py",
        symbol="TaskAgentGovernancePauseProjectionAdapter._apply_checkpoint_to_task",
        category="GOVERNANCE_PROJECTION",
        canonical_validator="validated_task_snapshot_from_checkpoint",
        mutates_live_state=True,
        can_influence_execution=False,
        can_influence_governance=True,
        tenant_binding="self._task.task_id / tenant_id",
        authority_handling="projection only; no execution admission",
        qualification_test="test_r4_r1_q05_pause_projection_invalid_snapshot_fails_before_mutation",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/runtime/human/agent_governance_grant_lifecycle.py",
        symbol="TaskAgentGovernanceGrantLifecycleAdapter._apply_checkpoint_to_task",
        category="GOVERNANCE_PROJECTION",
        canonical_validator="validated_task_snapshot_from_checkpoint",
        mutates_live_state=True,
        can_influence_execution=False,
        can_influence_governance=True,
        tenant_binding="self._task.task_id / tenant_id",
        authority_handling="lifecycle evidence only",
        qualification_test="test_r4_r1_q08_grant_lifecycle_invalid_snapshot_fails_before_mutation",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/applications/_shared/task_control_governance.py",
        symbol="_pause_id_from_checkpoint / task_checkpoint_stable_identity",
        category="CONTROL_PLANE_EVIDENCE",
        canonical_validator="validated_task_snapshot_from_checkpoint",
        mutates_live_state=False,
        can_influence_execution=False,
        can_influence_governance=True,
        tenant_binding="checkpoint.task_id / tenant_id",
        authority_handling="revision evidence; fail closed on invalid snapshot",
        qualification_test="test_r4_r1_q11_invalid_checkpoint_cannot_produce_mutation_revision",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/debug/hitl_service.py",
        symbol="DebugHitlResumeService.resume_with_human_response",
        category="RESTORE_EXECUTION",
        canonical_validator="assert_checkpoint_resume_materialization_eligible + validated_task_snapshot_from_checkpoint",
        mutates_live_state=False,
        can_influence_execution=True,
        can_influence_governance=True,
        tenant_binding="request tenant_id",
        authority_handling="lab path; full materialization before HostTaskExecutionPort",
        qualification_test="test_r4_r1_q18_debug_invalid_checkpoint_zero_execution",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/runtime/long_running/partial_results.py",
        symbol="build_partial_results_status",
        category="DIAGNOSTICS_ONLY",
        canonical_validator="none (non-authoritative; parse errors swallowed)",
        mutates_live_state=False,
        can_influence_execution=False,
        can_influence_governance=False,
        tenant_binding="request tenant_id",
        authority_handling="observational only",
        qualification_test="test_r4_r1_q19_diagnostics_only_consumer_classified",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/runtime/long_running/store.py",
        symbol="SQLiteTaskCheckpointStore",
        category="PERSISTENCE_ONLY",
        canonical_validator="n/a",
        mutates_live_state=False,
        can_influence_execution=False,
        can_influence_governance=False,
        tenant_binding="storage key tenant_id",
        authority_handling="none",
        qualification_test="test_r4_r1_q01_closed_world_task_checkpoint_snapshot_consumer_inventory",
    ),
    TaskCheckpointConsumerRecord(
        file="intergrax/runtime/long_running/checkpoint_resume_validation.py",
        symbol="validated_task_snapshot_from_checkpoint",
        category="RESTORE_EXECUTION",
        canonical_validator="canonical owner",
        mutates_live_state=False,
        can_influence_execution=False,
        can_influence_governance=False,
        tenant_binding="target_task_id / target_tenant_id",
        authority_handling="structural only",
        qualification_test="test_r4_r1_q03_canonical_structural_snapshot_reader_exactly_one",
    ),
)

# Production modules allowed to call Task.model_validate(...task_snapshot...) directly.
RAW_TASK_SNAPSHOT_PARSER_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "intergrax/runtime/long_running/checkpoint_resume_validation.py",
        "intergrax/runtime/long_running/coordinator.py",
        "intergrax/runtime/long_running/resume_planner.py",
        "intergrax/runtime/long_running/scheduler.py",
        "intergrax/runtime/long_running/partial_results.py",
    },
)


def _is_task_snapshot_model_validate(call: ast.Call) -> bool:
    func = call.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "model_validate"
        and isinstance(func.value, ast.Name)
        and func.value.id == "Task"
    ):
        return False
    if not call.args:
        return False
    arg = call.args[0]
    if isinstance(arg, ast.Attribute) and arg.attr == "task_snapshot":
        return True
    if isinstance(arg, ast.Subscript):
        base = arg.value
        return isinstance(base, ast.Attribute) and base.attr == "task_snapshot"
    return False


def find_raw_task_snapshot_parsers() -> list[tuple[str, int]]:
    violations: list[tuple[str, int]] = []
    for path in sorted(_INTERGRAX.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in RAW_TASK_SNAPSHOT_PARSER_ALLOWLIST:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_task_snapshot_model_validate(node):
                violations.append((rel, node.lineno))
    return violations
