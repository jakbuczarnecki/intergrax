# © Artur Czarnecki. All rights reserved.

"""Deterministic restart/resume path classification for TRACE-X-P6."""

from __future__ import annotations

from tests.qualification.trace_x._trace_x_p6_discovery import discover_restart_resume_path_keys
from tests.qualification.trace_x._trace_x_p6_types import (
    RegisteredRestartResumePath,
    RestartResumeParityResult,
    RestartResumePathClass,
)


def _path_class(path: str) -> RestartResumePathClass:
    lowered = path.lower()
    if path in _CLASSIFICATION_OVERRIDES:
        return _CLASSIFICATION_OVERRIDES[path]
    if any(
        token in lowered
        for token in (
            "/debug/",
            "/eval/",
            "/experiments/",
            "/scripts/",
            "run-lkw-slack-ask-workflow-proof.py",
        )
    ):
        return RestartResumePathClass.E_TEST_LAB_REFERENCE
    if "intergrax/runtime/nexus/retry/" in path or path.endswith(
        "intergrax/runtime/nexus/retry/coordinator.py"
    ):
        return RestartResumePathClass.B_CANONICAL_RETRY
    if path.endswith("intergrax/runtime/execution/retry/policy.py"):
        return RestartResumePathClass.B_CANONICAL_RETRY
    if path.endswith("intergrax/runtime/resilience/local_provider_retry_budget.py"):
        return RestartResumePathClass.B_CANONICAL_RETRY
    if path.endswith("intergrax/runtime/execution/fan_out_partial_recovery.py"):
        return RestartResumePathClass.C_NEW_EXECUTION_AFTER_FAILURE
    if path.endswith("intergrax/runtime/execution/child.py"):
        return RestartResumePathClass.C_NEW_EXECUTION_AFTER_FAILURE
    if path.endswith("intergrax/contracts/execution_retry.py"):
        return RestartResumePathClass.B_CANONICAL_RETRY
    if path.endswith("intergrax/runtime/execution/decision_recovery.py"):
        return RestartResumePathClass.D_SANCTIONED_NON_RESUMABLE
    if path.endswith("intergrax/runtime/execution/debug_lab_nexus_loop.py"):
        return RestartResumePathClass.E_TEST_LAB_REFERENCE
    if path.endswith("intergrax/runtime/notifications/formatters.py"):
        return RestartResumePathClass.D_SANCTIONED_NON_RESUMABLE
    return RestartResumePathClass.A_CANONICAL_RESUME


_CLASSIFICATION_OVERRIDES: dict[str, RestartResumePathClass] = {
    "applications/governed_contractor_application/host/offline_demo.py": (
        RestartResumePathClass.E_TEST_LAB_REFERENCE
    ),
    "applications/governed_contractor_application/host/orchestrator.py": (
        RestartResumePathClass.E_TEST_LAB_REFERENCE
    ),
}


def _summary_for(path: str, classification: RestartResumePathClass) -> str:
    return f"{classification.value}: {path}"


def build_restart_resume_registry() -> tuple[RegisteredRestartResumePath, ...]:
    rows: list[RegisteredRestartResumePath] = []
    for path, surface_id in sorted(discover_restart_resume_path_keys()):
        classification = _path_class(path)
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
