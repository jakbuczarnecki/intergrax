# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R3 checkpoint → recovery E2E chain (current-HEAD structural)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present
from tests.qualification.tenant_x._r3_types import R3ChainTransition, R3ChainVerdict

_EXEC_TO_CHECKPOINT_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/runtime/long_running/checkpoint_builder.py",
        "tenant_id=task.tenant_id",
    ),
    (
        "intergrax/runtime/long_running/store.py",
        "checkpoint.tenant_id",
    ),
)

_CHECKPOINT_PERSIST_LOAD_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/runtime/long_running/store.py",
        "WHERE task_id = ? AND tenant_id = ? AND resume_token = ?",
    ),
    (
        "intergrax/runtime/long_running/store.py",
        "_checkpoint_write_payload",
    ),
)

_RESUME_CHAIN_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/runtime/long_running/store.py",
        "checkpoint.tenant_id",
    ),
    (
        "intergrax/runtime/long_running/checkpoint_resume_validation.py",
        "checkpoint.tenant_id != target_tenant_id",
    ),
    (
        "intergrax/applications/_shared/task_control.py",
        "assert_checkpoint_resume_materialization_eligible",
    ),
    (
        "intergrax/applications/_shared/task_control.py",
        "target_tenant_id=tenant_id",
    ),
)

_AGENT_CHECKPOINT_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/agents/persistence/checkpoint_store.py",
        "checkpoint.tenant_id != tenant_id",
    ),
    (
        "intergrax/agents/persistence/checkpoint_store.py",
        "CheckpointStreamIdentityConflictError",
    ),
)

_PARTIAL_RECOVERY_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/runtime/execution/fan_out_partial_recovery.py",
        "checkpoint.tenant_id != tenant_id",
    ),
    (
        "intergrax/runtime/execution/fan_out_partial_recovery.py",
        "PartialRecoveryErrorCode.TENANT_MISMATCH",
    ),
)

_RETRY_LINEAGE_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/runtime/execution/retry/service.py",
        "transition_to_next_attempt",
    ),
    (
        "intergrax/runtime/execution/retry/service.py",
        "tenant_id=tenant_id",
    ),
    (
        "intergrax/runtime/execution/attempt_lifecycle/service.py",
        "def transition_to_next_attempt",
    ),
)


def _verdict_from_checks(
    checks: tuple[tuple[str, str], ...],
    *,
    discovery_note: str,
) -> tuple[R3ChainVerdict, tuple[str, ...]]:
    matched = all_structural_patterns_present(checks)
    if len(matched) == len(checks):
        return R3ChainVerdict.PROVEN_CURRENT_HEAD, matched
    return R3ChainVerdict.MISSING, (discovery_note,)


def _resume_chain_verdict() -> tuple[R3ChainVerdict, tuple[str, ...]]:
    return _verdict_from_checks(
        _RESUME_CHAIN_CHECKS,
        discovery_note="discovery: checkpoint→recovery chain incomplete",
    )


def build_r3_e2e_chain() -> tuple[R3ChainTransition, ...]:
    exec_verdict, exec_ev = _verdict_from_checks(
        _EXEC_TO_CHECKPOINT_CHECKS,
        discovery_note="discovery: execution→checkpoint write markers incomplete",
    )
    persist_verdict, persist_ev = _verdict_from_checks(
        _CHECKPOINT_PERSIST_LOAD_CHECKS,
        discovery_note="discovery: checkpoint persist/load tenant scope incomplete",
    )
    resume_verdict, resume_ev = _resume_chain_verdict()
    agent_verdict, agent_ev = _verdict_from_checks(
        _AGENT_CHECKPOINT_CHECKS,
        discovery_note="discovery: agent checkpoint durable lookup incomplete",
    )
    partial_verdict, partial_ev = _verdict_from_checks(
        _PARTIAL_RECOVERY_CHECKS,
        discovery_note="discovery: partial recovery tenant gate incomplete",
    )
    retry_verdict, retry_ev = _verdict_from_checks(
        _RETRY_LINEAGE_CHECKS,
        discovery_note="discovery: retry attempt lineage tenant threading incomplete",
    )
    return (
        R3ChainTransition(
            transition_id="execution.tenant_to_checkpoint_write",
            source_symbol="intergrax/runtime/long_running/checkpoint_builder.py::build_task_checkpoint",
            target_symbol="intergrax/runtime/long_running/store.py::SQLiteTaskCheckpointStore.save",
            tenant_source="task.tenant_id",
            tenant_target="checkpoint.tenant_id column + payload",
            relation="structural assignment tenant_id=task.tenant_id",
            mismatch_behavior="revision conflict preserves existing tenant-bound row",
            fail_closed=True,
            verdict=exec_verdict,
            evidence=exec_ev,
        ),
        R3ChainTransition(
            transition_id="checkpoint_persist.tenant_truth",
            source_symbol="intergrax/runtime/long_running/store.py::save",
            target_symbol="intergrax/runtime/long_running/store.py::get_by_token",
            tenant_source="checkpoint.tenant_id",
            tenant_target="SQL WHERE tenant_id scoped lookup",
            relation="durable lookup scoped by caller tenant + task + token",
            mismatch_behavior="wrong tenant → row not found (no cross-tenant load)",
            fail_closed=True,
            verdict=persist_verdict,
            evidence=persist_ev,
        ),
        R3ChainTransition(
            transition_id="checkpoint_load.identity_binding",
            source_symbol="intergrax/runtime/long_running/checkpoint_resume_validation.py::validate_checkpoint_identity_binding",
            target_symbol="intergrax/runtime/long_running/checkpoint_resume_validation.py::evaluate_checkpoint_resume_eligibility",
            tenant_source="checkpoint.tenant_id (durable truth)",
            tenant_target="target_tenant_id (caller resume request)",
            relation="equality check → REJECT_TENANT on mismatch",
            mismatch_behavior="CheckpointResumeEligibility.REJECT_TENANT",
            fail_closed=True,
            verdict=resume_verdict,
            evidence=resume_ev,
        ),
        R3ChainTransition(
            transition_id="agent_checkpoint.durable_lookup",
            source_symbol="intergrax/agents/persistence/checkpoint_store.py::SQLiteAgentCheckpointStore.get_latest",
            target_symbol="intergrax/agents/persistence/checkpoint_store.py::CheckpointStreamIdentityConflictError",
            tenant_source="AgentRunCheckpoint.tenant_id in payload",
            tenant_target="lookup tenant_id parameter",
            relation="payload tenant must match lookup keys",
            mismatch_behavior="CheckpointStreamIdentityConflictError",
            fail_closed=True,
            verdict=agent_verdict,
            evidence=agent_ev,
        ),
        R3ChainTransition(
            transition_id="governed_resume.materialization",
            source_symbol="intergrax/applications/_shared/task_control.py::governed_resume_checkpoint_task",
            target_symbol="intergrax/runtime/long_running/checkpoint_resume_validation.py::assert_checkpoint_resume_materialization_eligible",
            tenant_source="checkpoint.tenant_id via get_by_token(tenant_id)",
            tenant_target="target_tenant_id=tenant_id validated against checkpoint",
            relation="CALLER_TENANT_VALIDATED_AGAINST_DURABLE",
            mismatch_behavior="tenant_authority_mismatch / resume eligibility REJECT_TENANT",
            fail_closed=True,
            verdict=resume_verdict,
            evidence=resume_ev,
        ),
        R3ChainTransition(
            transition_id="partial_recovery.tenant_gate",
            source_symbol="intergrax/runtime/execution/fan_out_partial_recovery.py::PartialRecoveryCoordinator.validate_recovery_request",
            target_symbol="intergrax/contracts/partial_recovery.py::PartialRecoveryErrorCode.TENANT_MISMATCH",
            tenant_source="checkpoint.tenant_id",
            tenant_target="recovery request tenant_id parameter",
            relation="checkpoint.tenant_id != tenant_id → PartialRecoveryError",
            mismatch_behavior="TENANT_MISMATCH fail closed",
            fail_closed=True,
            verdict=partial_verdict,
            evidence=partial_ev,
        ),
        R3ChainTransition(
            transition_id="retry.attempt_lineage",
            source_symbol="intergrax/runtime/execution/retry/service.py::ExecutionRetryService.transition_for_retry",
            target_symbol="intergrax/runtime/execution/attempt_lifecycle/service.py::transition_to_next_attempt",
            tenant_source="explicit tenant_id on retry transition",
            tenant_target="durable attempt record tenant scope",
            relation="tenant_id threaded into lifecycle transition (not inferred from execution_id alone)",
            mismatch_behavior="lifecycle errors fail closed without tenant rewrite",
            fail_closed=True,
            verdict=retry_verdict,
            evidence=retry_ev,
        ),
    )


def r3_chain_missing_count() -> int:
    return sum(1 for t in build_r3_e2e_chain() if t.verdict == R3ChainVerdict.MISSING)


def r3_chain_invalid_count() -> int:
    return sum(1 for t in build_r3_e2e_chain() if t.verdict == R3ChainVerdict.INVALID)
