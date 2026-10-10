# © Artur Czarnecki. All rights reserved.

"""Deterministic terminal outcome producer classification for TRACE-X-P6 (fail-closed)."""

from __future__ import annotations

from tests.qualification.trace_x._trace_x_p6_discovery import discover_terminal_producer_keys
from tests.qualification.trace_x._trace_x_p6_module_evidence import terminal_evidence
from tests.qualification.trace_x._trace_x_p6_types import (
    RegisteredTerminalProducer,
    TerminalProducerParityResult,
    TerminalProducerRole,
)

_CANONICAL_TRUTH_PATH = "intergrax/runtime/execution/execution_terminal/service.py"

_ROLE_OVERRIDES: dict[str, TerminalProducerRole] = {
    "intergrax/runtime/cancellation/resume_admission.py": TerminalProducerRole.COMPATIBILITY_ADAPTER,
}


def _terminal_role_from_evidence(path: str, evidence: frozenset[str]) -> TerminalProducerRole:
    if path in _ROLE_OVERRIDES:
        return _ROLE_OVERRIDES[path]
    if path == _CANONICAL_TRUTH_PATH or "defines_execution_terminal_service" in evidence:
        return TerminalProducerRole.CANONICAL_TERMINAL_TRUTH
    if path.startswith("intergrax/runtime/diagnostics/"):
        return TerminalProducerRole.DIAGNOSTIC_PROJECTION
    if path.startswith("intergrax/runtime/adaptive/"):
        return TerminalProducerRole.OBSERVABILITY_PROJECTION
    if path.startswith("intergrax/contracts/execution_terminal.py"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/contracts/execution/execution_terminal"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.endswith("intergrax/runtime/execution/execution_terminal/persistence.py"):
        return TerminalProducerRole.CANONICAL_TERMINAL_DELEGATE
    if path.endswith("intergrax/runtime/events/trace_bridge.py"):
        return TerminalProducerRole.RUNTIME_EVENT_EVIDENCE
    if path.startswith("intergrax/runtime/task/task_state.py"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/eval/") or path.startswith("intergrax/experiments/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if "calls_commit_terminal_outcome" in evidence or "calls_record_cancellation" in evidence:
        return TerminalProducerRole.CANONICAL_TERMINAL_DELEGATE
    if path.endswith("intergrax/runtime/nexus/nexus_loop.py"):
        return TerminalProducerRole.CANONICAL_TERMINAL_DELEGATE
    if path.endswith("intergrax/runtime/nexus/orchestration/graph_runner.py"):
        return TerminalProducerRole.CANONICAL_TERMINAL_DELEGATE
    if path.startswith("intergrax/contracts/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/applications/_shared/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/runtime/long_running/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/runtime/execution/execution_terminal/"):
        return TerminalProducerRole.CANONICAL_TERMINAL_DELEGATE
    if path.startswith("intergrax/runtime/execution/lineage/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/runtime/execution/retry/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/runtime/execution/suspended_operation/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/runtime/background_execution/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/queueing/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/autonomous_work/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/hosting/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/background_tasks/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/runtime/nexus/orchestration/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/runtime/task/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if path.startswith("intergrax/runtime/execution/"):
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    if "reconcile_task_state_with_terminal" in evidence:
        return TerminalProducerRole.CANONICAL_TERMINAL_DELEGATE
    if evidence <= frozenset(
        {
            "execution_terminal_outcome_type",
            "execution_terminal_record_type",
            "execution_terminal_conflict_type",
            "references_commit_terminal_outcome",
            "references_record_cancellation",
        },
    ) and evidence:
        return TerminalProducerRole.COMPATIBILITY_ADAPTER
    return TerminalProducerRole.UNCLEAR


def classify_terminal_producer(path: str) -> TerminalProducerRole:
    evidence = terminal_evidence(path)
    return _terminal_role_from_evidence(path, evidence)


def build_terminal_producer_registry() -> tuple[RegisteredTerminalProducer, ...]:
    rows: list[RegisteredTerminalProducer] = []
    for path, surface_id in sorted(discover_terminal_producer_keys()):
        role = classify_terminal_producer(path)
        evidence = terminal_evidence(path)
        marker_summary = ",".join(sorted(evidence)) if evidence else "marker_only"
        rows.append(
            RegisteredTerminalProducer(
                path=path,
                surface_id=surface_id,
                role=role,
                summary=f"{role.value}: {path} [{marker_summary}]",
            ),
        )
    return tuple(rows)


TERMINAL_PRODUCER_REGISTRY: tuple[RegisteredTerminalProducer, ...] = (
    build_terminal_producer_registry()
)


def compare_terminal_producer_registry(
    discovered: frozenset[tuple[str, str]],
    registry: tuple[RegisteredTerminalProducer, ...],
) -> TerminalProducerParityResult:
    reg_keys = frozenset((row.path, row.surface_id) for row in registry)
    unknown = discovered - reg_keys
    orphan = reg_keys - discovered
    forbidden = frozenset(
        (row.path, row.surface_id)
        for row in registry
        if row.role is TerminalProducerRole.FORBIDDEN_BYPASS
    )
    unclassified = frozenset(
        (row.path, row.surface_id)
        for row in registry
        if row.role is TerminalProducerRole.UNCLEAR
    )
    ok = not unknown and not orphan and not forbidden and not unclassified
    return TerminalProducerParityResult(
        unknown=unknown,
        orphan=orphan,
        forbidden_bypass=forbidden,
        unclassified=unclassified,
        ok=ok,
    )
