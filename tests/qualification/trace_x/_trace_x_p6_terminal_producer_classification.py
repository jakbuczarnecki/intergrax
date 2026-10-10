# © Artur Czarnecki. All rights reserved.

"""Deterministic terminal outcome producer classification for TRACE-X-P6."""

from __future__ import annotations

from tests.qualification.trace_x._trace_x_p6_discovery import discover_terminal_producer_keys
from tests.qualification.trace_x._trace_x_p6_types import (
    RegisteredTerminalProducer,
    TerminalProducerParityResult,
    TerminalProducerRole,
)

_CANONICAL_TRUTH_PATH = "intergrax/runtime/execution/execution_terminal/service.py"


def _terminal_role(path: str) -> TerminalProducerRole:
    if path in _ROLE_OVERRIDES:
        return _ROLE_OVERRIDES[path]
    if path == _CANONICAL_TRUTH_PATH:
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
    if "commit_terminal_outcome" in path or "ExecutionTerminalService" in path:
        return TerminalProducerRole.CANONICAL_TERMINAL_DELEGATE
    if "record_cancellation" in path:
        return TerminalProducerRole.CANONICAL_TERMINAL_DELEGATE
    return TerminalProducerRole.CANONICAL_TERMINAL_DELEGATE


_ROLE_OVERRIDES: dict[str, TerminalProducerRole] = {
    "intergrax/runtime/cancellation/resume_admission.py": (
        TerminalProducerRole.COMPATIBILITY_ADAPTER
    ),
}


def build_terminal_producer_registry() -> tuple[RegisteredTerminalProducer, ...]:
    rows: list[RegisteredTerminalProducer] = []
    for path, surface_id in sorted(discover_terminal_producer_keys()):
        role = _terminal_role(path)
        rows.append(
            RegisteredTerminalProducer(
                path=path,
                surface_id=surface_id,
                role=role,
                summary=f"{role.value}: {path}",
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
