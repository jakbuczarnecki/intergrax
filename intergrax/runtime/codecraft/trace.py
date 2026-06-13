# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraftTraceEmitter — CODECRAFT_* diagnostic trace steps (ECC-1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.runtime.nexus.tracing.trace_models import (
    DiagnosticPayload,
    TraceComponent,
    TraceEvent,
    TraceLevel,
    utc_now_iso,
)

CODECRAFT_STEP_SESSION_OPENED = "codecraft.session_opened"
CODECRAFT_STEP_STATIC_GATE = "codecraft.static_gate"
CODECRAFT_STEP_EXEC = "codecraft.exec"
CODECRAFT_STEP_DISPOSED = "codecraft.disposed"


@dataclass(frozen=True)
class CodeCraftDiagV1(DiagnosticPayload):
    """Typed payload for code craft trace steps."""

    craft_id: str
    event: str
    mode: str
    passed: bool | None = None
    rule_ids: tuple[str, ...] = ()
    sandbox_session_id: str | None = None
    exit_code: int | None = None
    duration_ms: float | None = None

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.codecraft.event"

    def to_dict(self) -> dict[str, Any]:
        return {
            "craft_id": self.craft_id,
            "event": self.event,
            "mode": self.mode,
            "passed": self.passed,
            "rule_ids": list(self.rule_ids),
            "sandbox_session_id": self.sandbox_session_id,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
        }

    def redact(self) -> CodeCraftDiagV1:
        return self


class CodeCraftTraceEmitter:
    """Emits ``codecraft.*`` trace steps correlated with craft_id and sandbox session."""

    def __init__(
        self,
        *,
        run_id: str,
        trace_writer: RunTraceWriter | None = None,
        event_bus: RuntimeEventBus | None = None,
        seq_offset: int = 0,
    ) -> None:
        self._run_id = run_id
        self._seq = seq_offset
        self._trace_writer = trace_writer
        self._event_bus = event_bus
        self.events: list[TraceEvent] = []

    def session_opened(
        self,
        *,
        craft_id: str,
        mode: str,
        tenant_id: str,
        task_id: str,
        agent_id: str = "",
    ) -> TraceEvent:
        return self._emit(
            step=CODECRAFT_STEP_SESSION_OPENED,
            message=f"codecraft session opened ({mode})",
            level=TraceLevel.INFO,
            payload=CodeCraftDiagV1(
                craft_id=craft_id,
                event="CODECRAFT_SESSION_OPENED",
                mode=mode,
                passed=None,
            ),
            tags={
                "tenant_id": tenant_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "craft_id": craft_id,
            },
        )

    def static_gate(
        self,
        *,
        craft_id: str,
        mode: str,
        passed: bool,
        rule_ids: tuple[str, ...],
        tenant_id: str,
        task_id: str,
        agent_id: str = "",
    ) -> TraceEvent:
        return self._emit(
            step=CODECRAFT_STEP_STATIC_GATE,
            message=f"codecraft static gate {'passed' if passed else 'failed'}",
            level=TraceLevel.INFO if passed else TraceLevel.ERROR,
            payload=CodeCraftDiagV1(
                craft_id=craft_id,
                event="CODECRAFT_STATIC_GATE",
                mode=mode,
                passed=passed,
                rule_ids=rule_ids,
            ),
            tags={
                "tenant_id": tenant_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "craft_id": craft_id,
            },
        )

    def exec_completed(
        self,
        *,
        craft_id: str,
        mode: str,
        sandbox_session_id: str | None,
        exit_code: int | None,
        success: bool,
        tenant_id: str,
        task_id: str,
        agent_id: str = "",
        duration_ms: float | None = None,
    ) -> TraceEvent:
        return self._emit(
            step=CODECRAFT_STEP_EXEC,
            message=f"codecraft exec {'ok' if success else 'failed'}",
            level=TraceLevel.INFO if success else TraceLevel.ERROR,
            payload=CodeCraftDiagV1(
                craft_id=craft_id,
                event="CODECRAFT_EXEC",
                mode=mode,
                passed=success,
                sandbox_session_id=sandbox_session_id,
                exit_code=exit_code,
                duration_ms=duration_ms,
            ),
            tags={
                "tenant_id": tenant_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "craft_id": craft_id,
            },
        )

    def disposed(
        self,
        *,
        craft_id: str,
        mode: str,
        tenant_id: str,
        task_id: str,
        agent_id: str = "",
    ) -> TraceEvent:
        return self._emit(
            step=CODECRAFT_STEP_DISPOSED,
            message="codecraft session disposed",
            level=TraceLevel.INFO,
            payload=CodeCraftDiagV1(
                craft_id=craft_id,
                event="CODECRAFT_DISPOSED",
                mode=mode,
                passed=True,
            ),
            tags={
                "tenant_id": tenant_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "craft_id": craft_id,
            },
        )

    def _emit(
        self,
        *,
        step: str,
        message: str,
        level: TraceLevel,
        payload: CodeCraftDiagV1,
        tags: dict[str, Any],
    ) -> TraceEvent:
        self._seq += 1
        evt = TraceEvent(
            event_id=TraceEvent.new_id(),
            run_id=self._run_id,
            seq=self._seq,
            ts_utc=utc_now_iso(),
            level=level,
            component=TraceComponent.CODECRAFT,
            step=step,
            message=message,
            payload=payload,
            tags=tags,
        )
        self.events.append(evt)
        if self._trace_writer is not None:
            self._trace_writer.append_event(evt)
        if self._event_bus is not None:
            from intergrax.runtime.events.trace_bridge import (
                trace_bridge_subject_from_tags,
                trace_event_to_runtime_event,
            )

            subject = trace_bridge_subject_from_tags(
                tenant_id=str(tags.get("tenant_id", "default")),
                task_id=str(tags.get("task_id", self._run_id)),
                agent_id=str(tags.get("agent_id", "")),
            )
            self._event_bus.record(trace_event_to_runtime_event(evt, subject))
        return evt
