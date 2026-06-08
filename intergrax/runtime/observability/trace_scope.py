# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Causal trace scope for Harness Observability Spine (OBS-BUS-2)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from intergrax.runtime.observability.emitter import ObservabilityEmitter

_trace_scope_var: ContextVar[Optional["TraceScopeState"]] = ContextVar(
    "intergrax_hos_trace_scope",
    default=None,
)


@dataclass(frozen=True)
class TraceScopeState:
    """Active correlation context for nested spine emissions."""

    run_id: str
    task_id: str
    tenant_id: str
    correlation_id: str
    parent_event_id: Optional[str] = None
    step_id: Optional[str] = None
    node_id: Optional[str] = None
    agent_id: Optional[str] = None


def current_trace_scope() -> Optional[TraceScopeState]:
    return _trace_scope_var.get()


def current_parent_event_id() -> Optional[str]:
    scope = _trace_scope_var.get()
    if scope is None:
        return None
    return scope.parent_event_id


def bind_parent_event_id(parent_event_id: str) -> None:
    """Update the active scope parent (used after anchor events such as STEP_STARTED)."""
    scope = _trace_scope_var.get()
    if scope is None:
        return
    _trace_scope_var.set(
        TraceScopeState(
            run_id=scope.run_id,
            task_id=scope.task_id,
            tenant_id=scope.tenant_id,
            correlation_id=scope.correlation_id,
            parent_event_id=parent_event_id,
            step_id=scope.step_id,
            node_id=scope.node_id,
            agent_id=scope.agent_id,
        )
    )


class TraceScope:
    """
    Context manager establishing run/task correlation and optional causal parent.

    Nested scopes inherit ``correlation_id`` and may override ``parent_event_id``.
    """

    def __init__(
        self,
        emitter: ObservabilityEmitter,
        *,
        run_id: str,
        task_id: str,
        tenant_id: str,
        correlation_id: Optional[str] = None,
        parent_event_id: Optional[str] = None,
        step_id: Optional[str] = None,
        node_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        self._emitter = emitter
        self._state = TraceScopeState(
            run_id=run_id,
            task_id=task_id,
            tenant_id=tenant_id,
            correlation_id=correlation_id or task_id,
            parent_event_id=parent_event_id or current_parent_event_id(),
            step_id=step_id,
            node_id=node_id,
            agent_id=agent_id,
        )
        self._token: Optional[Token[Optional[TraceScopeState]]] = None
        self._last_runtime_event_id: Optional[str] = None

    @property
    def state(self) -> TraceScopeState:
        return self._state

    @property
    def current(self) -> Optional[str]:
        """Runtime ``event_id`` of the last emission in this scope (for child binding)."""
        return self._last_runtime_event_id

    def register_runtime_event(self, event_id: str) -> None:
        self._last_runtime_event_id = event_id
        bind_parent_event_id(event_id)

    def __enter__(self) -> TraceScope:
        self._token = _trace_scope_var.set(self._state)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _trace_scope_var.reset(self._token)

    @contextmanager
    def step(
        self,
        step_id: str,
        *,
        node_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Iterator[TraceScope]:
        """
        Emit ``STEP_STARTED``, bind children to that event, then complete or fail.
        """
        from intergrax.contracts.event_severity import EventSeverity
        from intergrax.contracts.execution_phase import ExecutionPhase
        from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

        started = self._emitter.emit_runtime(
            RuntimeEvent(
                tenant_id=self._state.tenant_id,
                task_id=self._state.task_id,
                run_id=self._state.run_id,
                agent_id=agent_id or self._state.agent_id,
                step_id=step_id,
                node_id=node_id or self._state.node_id,
                event_type=RuntimeEventType.STEP_STARTED,
                phase=ExecutionPhase.STEP_EXECUTION,
                severity=EventSeverity.INFO,
                correlation_id=self._state.correlation_id,
            )
        )
        child_scope = TraceScope(
            self._emitter,
            run_id=self._state.run_id,
            task_id=self._state.task_id,
            tenant_id=self._state.tenant_id,
            correlation_id=self._state.correlation_id,
            parent_event_id=started.event_id,
            step_id=step_id,
            node_id=node_id or self._state.node_id,
            agent_id=agent_id or self._state.agent_id,
        )
        try:
            with child_scope:
                yield child_scope
        except Exception:
            self._emitter.emit_runtime(
                RuntimeEvent(
                    tenant_id=self._state.tenant_id,
                    task_id=self._state.task_id,
                    run_id=self._state.run_id,
                    agent_id=agent_id or self._state.agent_id,
                    step_id=step_id,
                    node_id=node_id or self._state.node_id,
                    event_type=RuntimeEventType.STEP_FAILED,
                    phase=ExecutionPhase.STEP_EXECUTION,
                    severity=EventSeverity.ERROR,
                    correlation_id=self._state.correlation_id,
                    parent_event_id=started.event_id,
                )
            )
            raise
        else:
            self._emitter.emit_runtime(
                RuntimeEvent(
                    tenant_id=self._state.tenant_id,
                    task_id=self._state.task_id,
                    run_id=self._state.run_id,
                    agent_id=agent_id or self._state.agent_id,
                    step_id=step_id,
                    node_id=node_id or self._state.node_id,
                    event_type=RuntimeEventType.STEP_COMPLETED,
                    phase=ExecutionPhase.STEP_EXECUTION,
                    severity=EventSeverity.INFO,
                    correlation_id=self._state.correlation_id,
                    parent_event_id=started.event_id,
                )
            )
