# © Artur Czarnecki. All rights reserved.

"""Deterministic restart/resume path classification for TRACE-X-P6 (fail-closed)."""

from __future__ import annotations

from tests.qualification.trace_x._trace_x_p6_discovery import discover_restart_resume_path_keys
from tests.qualification.trace_x._trace_x_p6_types import (
    RegisteredRestartResumePath,
    RestartResumeParityResult,
    RestartResumePathClass,
)

_CLASSIFICATION_OVERRIDES: dict[str, RestartResumePathClass] = {
    "applications/governed_contractor_application/host/offline_demo.py": (
        RestartResumePathClass.E_TEST_LAB_REFERENCE
    ),
    "applications/governed_contractor_application/host/orchestrator.py": (
        RestartResumePathClass.E_TEST_LAB_REFERENCE
    ),
}

# Prefix families sharing one restart/resume semantic responsibility (most specific first).
_PREFIX_CLASSIFICATION: tuple[tuple[str, RestartResumePathClass], ...] = (
    ("intergrax/runtime/execution/debug_lab_nexus_loop.py", RestartResumePathClass.E_TEST_LAB_REFERENCE),
    ("applications/local_workspace_application/scripts/run-lkw-slack-ask-workflow-proof.py", RestartResumePathClass.E_TEST_LAB_REFERENCE),
    ("applications/lab_application/", RestartResumePathClass.E_TEST_LAB_REFERENCE),
    ("applications/poc_template_application/", RestartResumePathClass.E_TEST_LAB_REFERENCE),
    ("intergrax/debug/", RestartResumePathClass.E_TEST_LAB_REFERENCE),
    ("intergrax/eval/", RestartResumePathClass.E_TEST_LAB_REFERENCE),
    ("intergrax/experiments/", RestartResumePathClass.E_TEST_LAB_REFERENCE),
    ("intergrax/runtime/nexus/retry/", RestartResumePathClass.B_CANONICAL_RETRY),
    ("intergrax/runtime/execution/retry/", RestartResumePathClass.B_CANONICAL_RETRY),
    ("intergrax/contracts/execution_retry.py", RestartResumePathClass.B_CANONICAL_RETRY),
    ("intergrax/runtime/resilience/local_provider_retry_budget.py", RestartResumePathClass.B_CANONICAL_RETRY),
    ("intergrax/runtime/execution/fan_out_partial_recovery.py", RestartResumePathClass.C_NEW_EXECUTION_AFTER_FAILURE),
    ("intergrax/runtime/execution/child.py", RestartResumePathClass.C_NEW_EXECUTION_AFTER_FAILURE),
    ("intergrax/runtime/execution/decision_recovery.py", RestartResumePathClass.D_SANCTIONED_NON_RESUMABLE),
    ("intergrax/runtime/notifications/formatters.py", RestartResumePathClass.D_SANCTIONED_NON_RESUMABLE),
    ("intergrax/runtime/long_running/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/execution/continuation/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/execution/suspended_operation/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/contracts/execution/suspended_operation/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/contracts/execution_continuation", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/contracts/delegated_execution_continuation.py", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/execution/delegated_execution/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/execution/active_execution_continuation_store.py", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/background_execution/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/cancellation/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/human/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/autonomous_work/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/applications/_shared/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/agents/persistence/checkpoint_wiring.py", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/task/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/execution/execution_terminal/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/nexus/orchestration/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/nexus/tools/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/nexus/nexus_loop.py", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/events/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/persistence/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/policy/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/runtime_inspection/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/interactions/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/queueing/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/integrations/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/contracts/autonomous_work/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/contracts/runtime_inspection/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/contracts/decision/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/contracts/agent_run.py", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/contracts/task_metadata_keys.py", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/background_tasks/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/notifications/templates/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/execution/budget/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("intergrax/runtime/execution/", RestartResumePathClass.A_CANONICAL_RESUME),
    ("applications/governed_contractor_application/", RestartResumePathClass.A_CANONICAL_RESUME),
)


def classify_restart_resume_path(path: str) -> RestartResumePathClass:
    """Positive classification only; unknown surfaces must not default to canonical resume."""
    if path in _CLASSIFICATION_OVERRIDES:
        return _CLASSIFICATION_OVERRIDES[path]
    lowered = path.lower()
    if any(
        token in lowered
        for token in (
            "/debug/",
            "/eval/",
            "/experiments/",
            "/scripts/",
        )
    ):
        return RestartResumePathClass.E_TEST_LAB_REFERENCE
    for prefix, classification in _PREFIX_CLASSIFICATION:
        if path == prefix or path.startswith(prefix):
            return classification
    return RestartResumePathClass.G_UNCLEAR


def _summary_for(path: str, classification: RestartResumePathClass) -> str:
    return f"{classification.value}: {path}"


def build_restart_resume_registry() -> tuple[RegisteredRestartResumePath, ...]:
    rows: list[RegisteredRestartResumePath] = []
    for path, surface_id in sorted(discover_restart_resume_path_keys()):
        classification = classify_restart_resume_path(path)
        rows.append(
            RegisteredRestartResumePath(
                path=path,
                surface_id=surface_id,
                classification=classification,
                summary=_summary_for(path, classification),
            ),
        )
    return tuple(rows)


RESTART_RESUME_REGISTRY: tuple[RegisteredRestartResumePath, ...] = build_restart_resume_registry()


def compare_restart_resume_registry(
    discovered: frozenset[tuple[str, str]],
    registry: tuple[RegisteredRestartResumePath, ...],
) -> RestartResumeParityResult:
    reg_keys = frozenset((row.path, row.surface_id) for row in registry)
    unknown = discovered - reg_keys
    orphan = reg_keys - discovered
    bypass = frozenset(
        (row.path, row.surface_id)
        for row in registry
        if row.classification is RestartResumePathClass.F_PRODUCTION_BYPASS
    )
    unclassified = frozenset(
        (row.path, row.surface_id)
        for row in registry
        if row.classification is RestartResumePathClass.G_UNCLEAR
    )
    ok = not unknown and not orphan and not bypass and not unclassified
    return RestartResumeParityResult(
        unknown=unknown,
        orphan=orphan,
        production_bypass=bypass,
        unclassified=unclassified,
        ok=ok,
    )
