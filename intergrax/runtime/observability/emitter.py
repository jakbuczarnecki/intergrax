# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified observability emit API for Harness Observability Spine (OBS-BUS-2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Protocol

from intergrax.contracts.execution_identity import (
    ExecutionId,
    peek_active_execution_identity,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.payload_registry import RuntimeEventPayload, runtime_event_with_payload
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.runtime.nexus.tracing.trace_models import (
    DEFAULT_REDACTED_TEXT,
    DiagnosticPayload,
    TraceComponent,
    TraceEvent,
    TraceLevel,
    utc_now_iso,
)
from intergrax.runtime.observability.trace_scope import (
    TraceScope,
    TraceScopeState,
    current_parent_event_id,
    current_trace_scope,
)

if TYPE_CHECKING:
    from intergrax.runtime.nexus.artifacts.models import ArtifactRef


class TraceSeqProvider(Protocol):
    def __call__(self) -> int: ...


@dataclass(frozen=True)
class EmittedDiagnostic:
    """Result of a diagnostic emission on trace + bus planes."""

    trace: TraceEvent
    runtime: RuntimeEvent


@dataclass
class ObservabilityEmitter:
    """
    Single developer-facing emit facade for trace + runtime event planes.

    Writes to ``RunTraceWriter``, optional in-memory trace sink, and ``RuntimeEventBus``.
    Applies active ``TraceScope`` ``parent_event_id`` on bus emissions.
    """

    run_id: str
    task_id: str
    tenant_id: str
    agent_id: str = ""
    attempt_id: str = ""
    execution_id: str = ""
    trace_writer: Optional[RunTraceWriter] = None
    event_bus: Optional[RuntimeEventBus] = None
    trace_events: Optional[List[TraceEvent]] = None
    production_mode: bool = False
    next_seq: Optional[TraceSeqProvider] = None
    _seq_counter: int = field(default=0, init=False, repr=False)

    def _allocate_seq(self) -> int:
        if self.next_seq is not None:
            return self.next_seq()
        self._seq_counter += 1
        return self._seq_counter

    def scope(
        self,
        *,
        correlation_id: Optional[str] = None,
        parent_event_id: Optional[str] = None,
        step_id: Optional[str] = None,
        node_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> TraceScope:
        return TraceScope(
            self,
            run_id=self.run_id,
            task_id=self.task_id,
            tenant_id=self.tenant_id,
            correlation_id=correlation_id or self.task_id,
            parent_event_id=parent_event_id,
            step_id=step_id,
            node_id=node_id,
            agent_id=agent_id or self.agent_id or None,
        )

    def run(
        self,
        *,
        correlation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> TraceScope:
        """Root scope for a single execution attempt."""
        return self.scope(
            correlation_id=correlation_id,
            agent_id=agent_id or self.agent_id or None,
        )

    def emit_diagnostic(
        self,
        *,
        component: TraceComponent,
        step: str,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        payload: Optional[DiagnosticPayload] = None,
        artifact_refs: Optional[List[ArtifactRef]] = None,
        tags: Optional[dict[str, object]] = None,
    ) -> EmittedDiagnostic:
        if not self.run_id:
            raise RuntimeError("ObservabilityEmitter.run_id must be provided (got empty).")

        if payload is not None and not isinstance(payload, DiagnosticPayload):
            raise TypeError("Trace payload must implement DiagnosticPayload.")
        if payload is not None:
            payload = payload.redact()

        safe_message = DEFAULT_REDACTED_TEXT if self.production_mode else message
        scope = current_trace_scope()
        merged_tags = {
            "tenant_id": self.tenant_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
        }
        if tags:
            merged_tags.update(tags)
        merged_tags["task_id"] = self.task_id
        if scope is not None:
            if scope.step_id:
                merged_tags.setdefault("step_id", scope.step_id)
            if scope.node_id:
                merged_tags.setdefault("node_id", scope.node_id)

        trace = TraceEvent(
            event_id=TraceEvent.new_id(),
            run_id=self.run_id,
            seq=self._allocate_seq(),
            ts_utc=utc_now_iso(),
            level=level,
            component=component,
            step=step,
            message=safe_message,
            payload=payload,
            tags=merged_tags,
            artifact_refs=tuple(artifact_refs) if artifact_refs else (),
        )

        if self.trace_events is not None:
            self.trace_events.append(trace)
        if self.trace_writer is not None:
            self.trace_writer.append_event(trace)

        runtime = self._bridge_trace_event(trace, scope=scope)
        return EmittedDiagnostic(trace=trace, runtime=runtime)

    def emit_runtime(
        self,
        event: RuntimeEvent,
        *,
        typed_payload: Optional[RuntimeEventPayload] = None,
        promote_fields: Optional[dict[str, object]] = None,
    ) -> RuntimeEvent:
        scoped = self._apply_scope(event, current_trace_scope())
        if typed_payload is not None:
            scoped = runtime_event_with_payload(
                scoped,
                typed_payload,
                promote_fields=promote_fields,
            )
        if self.event_bus is not None:
            self.event_bus.record(scoped, tenant_id=self.tenant_id)
        return scoped

    def _bridge_trace_event(
        self,
        trace: TraceEvent,
        *,
        scope: Optional[TraceScopeState],
    ) -> RuntimeEvent:
        from intergrax.runtime.events.trace_bridge import (
            trace_bridge_subject_from_tags,
            trace_event_to_runtime_event,
        )

        subject = trace_bridge_subject_from_tags(
            tenant_id=self.tenant_id,
            task_id=self.task_id,
            agent_id=self.agent_id,
        )
        correlation_id = scope.correlation_id if scope is not None else self.task_id
        bridge_execution_id: ExecutionId | None = None
        if self.execution_id:
            bridge_execution_id = validate_execution_id(self.execution_id)
        active = peek_active_execution_identity()
        if active is not None:
            bridge_run_id, bridge_attempt_id = active
            emitter_run_id = validate_run_id(self.run_id)
            if emitter_run_id != bridge_run_id:
                raise RuntimeError("emitter run_id conflicts with active execution identity")
        elif self.attempt_id:
            bridge_run_id = validate_run_id(self.run_id)
            bridge_attempt_id = validate_attempt_id(self.attempt_id)
        else:
            raise RuntimeError(
                "active execution identity or emitter attempt_id required for trace bridge",
            )
        runtime = trace_event_to_runtime_event(
            trace,
            subject,
            run_id=bridge_run_id,
            attempt_id=bridge_attempt_id,
            execution_id=bridge_execution_id,
            correlation_id=correlation_id,
        )
        runtime = self._apply_scope(runtime, scope)
        if scope is not None:
            if scope.step_id and runtime.step_id is None:
                runtime = runtime.model_copy(update={"step_id": scope.step_id})
            if scope.node_id and runtime.node_id is None:
                runtime = runtime.model_copy(update={"node_id": scope.node_id})
            if scope.agent_id and runtime.agent_id is None:
                runtime = runtime.model_copy(update={"agent_id": scope.agent_id})
        if self.event_bus is not None:
            self.event_bus.record(runtime, tenant_id=self.tenant_id)
        return runtime

    @staticmethod
    def _apply_scope(
        event: RuntimeEvent,
        scope: Optional[TraceScopeState],
    ) -> RuntimeEvent:
        parent_id = current_parent_event_id()
        updates: dict[str, object] = {}
        if parent_id and event.parent_event_id is None:
            updates["parent_event_id"] = parent_id
        if scope is not None:
            if event.correlation_id in ("", event.task_id) and scope.correlation_id:
                updates["correlation_id"] = scope.correlation_id
            if event.tenant_id is None:
                updates["tenant_id"] = scope.tenant_id
        if not updates:
            return event
        return event.model_copy(update=updates)
