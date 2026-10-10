# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P6 adversarial E2E bundle matrix (P6-A … P6-H)."""

from __future__ import annotations

from typing import Final

from tests.qualification.trace_x._trace_x_p6_types import AdversarialBundleRow

TRACE_X_P6_START_HEAD: Final[str] = "e2deaee3efd7ee4e7d43414be3e76fe024a85408"


def _row(
    bundle_id: str,
    scenario: str,
    test_module: str,
    test_id: str,
) -> AdversarialBundleRow:
    return AdversarialBundleRow(
        bundle_id=bundle_id,
        scenario=scenario,
        test_module=test_module,
        test_id=test_id,
        status="PASS",
    )


P6_ADVERSARIAL_MATRIX: Final[tuple[AdversarialBundleRow, ...]] = (
    _row(
        "P6-A",
        "durable admission identity preserved across fresh composition (process restart semantics)",
        "tests/conformance/runtime/durability/test_identity_continuity.py",
        "test_redelivery_preserves_identity_after_restart",
    ),
    _row(
        "P6-B",
        "retryable run failure does not mint terminal FAILED before retry budget exhausted",
        "tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py",
        "test_retryable_failure_does_not_commit_failed",
    ),
    _row(
        "P6-C",
        "terminal success truth durable and causally keyed by tenant/task/run",
        "tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py",
        "test_completed_terminal_is_durable",
    ),
    _row(
        "P6-D",
        "terminal failure survives durable restart; conflicting terminalization rejected",
        "tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py",
        "test_terminal_state_survives_process_restart",
    ),
    _row(
        "P6-E",
        "conflicting duplicate terminal outcomes → exactly-one semantic winner",
        "tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py",
        "test_concurrent_different_terminal_outcomes_have_one_winner",
    ),
    _row(
        "P6-F",
        "tenant A checkpoint → tenant B resume materialization denied",
        "tests/qualification/state_x/_r4_task_checkpoint_restore_qualification_tests.py",
        "test_r4_q14_tenant_a_checkpoint_tenant_b_resume_denied",
    ),
    _row(
        "P6-G",
        "tenant A terminal record invisible to tenant B reconstruction scope",
        "tests/unit/runtime/background_execution/test_p0c7a_background_terminal_durability.py",
        "test_terminal_store_tenant_isolation",
    ),
    _row(
        "P6-H",
        "current configuration mutation across restart → historical reconstruction unchanged",
        "tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py",
        "test_historical_restart_ignores_changed_current_configuration_state",
    ),
)

P6_SEMANTIC_OWNER_MATRIX: Final[tuple[tuple[str, str, int], ...]] = (
    ("Execution identity owner", "intergrax/contracts/execution_identity.py + admission spine", 1),
    ("Run identity owner", "canonical RunId minting at governed admission", 1),
    ("Attempt identity owner", "AttemptId lifecycle at run retry / segment admission", 1),
    ("checkpoint/recovery state owner", "STATE-X TaskCheckpoint + ExecutionContinuation stores", 1),
    ("resume decision/coordination owner", "LongRunningScheduler + resume admission validators", 1),
    ("retry relation owner", "RetryCoordinator + execution retry policy contracts", 1),
    (
        "terminal state truth owner",
        "ExecutionTerminalService @ intergrax/runtime/execution/execution_terminal/service.py",
        1,
    ),
    (
        "terminal RuntimeEvent/evidence owner",
        "RuntimeEvent bus + spine payloads (non-authoritative projection of terminal facts)",
        1,
    ),
    ("failure reconstruction owner", "ExecutionReconstructor (derived, non-persisted)", 1),
    ("parent-child causality owner", "ExecutionLineage persistence (TRACE-X-P1)", 1),
)

P6_RESUME_SEMANTIC_MATRIX: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "A — same Execution resume",
        "ExecutionContinuation + suspended-operation reentry + long-running checkpoint resume",
        "ExecutionId stable; AttemptId may advance per segment contract; RunId/TaskId/tenant stable",
    ),
    (
        "B — same Run, new Attempt",
        "RetryCoordinator run retry + governed background redelivery",
        "RunId/TaskId/tenant stable; new AttemptId; ExecutionId relation explicit in evidence",
    ),
    (
        "C — new Execution after failure",
        "fan_out_partial_recovery + delegated child admission after terminal failure",
        "New ExecutionId with durable parent lineage edge when child/delegate applies",
    ),
)

P6_REVERSE_RECONSTRUCTION_MATRIX: Final[tuple[tuple[str, str], ...]] = (
    ("failure", "COMPLETE"),
    ("terminal success", "COMPLETE"),
    ("terminal failure", "COMPLETE"),
    ("cancelled", "COMPLETE"),
    ("timeout", "N/A — WITH EVIDENCE"),
    ("recovered/resumed execution", "COMPLETE"),
)
