# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""EE-B1.1 — ExecutionRuntime graceful shutdown phase contract."""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "EXECUTION_RUNTIME_SHUTDOWN_PHASE_ORDER",
    "ExecutionRuntimeShutdownPhase",
]


class ExecutionRuntimeShutdownPhase(StrEnum):
    """Ordered shutdown phases — no kill-without-cleanup, no sealed-attempt reopen."""

    STOP_ACCEPTING_NEW_WORK = "stop_accepting_new_work"
    DRAIN_ACTIVE_EXECUTIONS = "drain_active_executions"
    FLUSH_REQUIRED_EVIDENCE = "flush_required_evidence"
    PERSIST_FINAL_STATE = "persist_final_state"
    TERMINATE_WORKERS = "terminate_workers"


EXECUTION_RUNTIME_SHUTDOWN_PHASE_ORDER: tuple[ExecutionRuntimeShutdownPhase, ...] = (
    ExecutionRuntimeShutdownPhase.STOP_ACCEPTING_NEW_WORK,
    ExecutionRuntimeShutdownPhase.DRAIN_ACTIVE_EXECUTIONS,
    ExecutionRuntimeShutdownPhase.FLUSH_REQUIRED_EVIDENCE,
    ExecutionRuntimeShutdownPhase.PERSIST_FINAL_STATE,
    ExecutionRuntimeShutdownPhase.TERMINATE_WORKERS,
)
