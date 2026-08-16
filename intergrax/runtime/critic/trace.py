# © Artur Czarnecki. All rights reserved.

"""Critic trace emission — `critic.*` steps for lab trace API (Phase CRIT-V-3.6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.runtime.critic.contracts import (
    CriticLayer,
    CriticRequest,
    CriticScope,
    CriticVerdict,
    LayerVerdict,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.runtime.nexus.tracing.trace_models import (
    DiagnosticPayload,
    TraceComponent,
    TraceEvent,
    TraceLevel,
    utc_now_iso,
)

from intergrax.runtime.critic.trace_steps import (
    CRITIC_STEP_EVALUATOR_LOOP,
    CRITIC_STEP_FINAL_VERDICT,
    CRITIC_STEP_L0_FAILED,
    CRITIC_STEP_L1_JUDGE,
    CRITIC_STEP_TRAJECTORY,
)


@dataclass(frozen=True)
class CriticVerdictDiagV1(DiagnosticPayload):
    """Typed payload for critic trace steps."""

    scope: str
    passed: bool
    recommended_action: str
    layer: str
    score: float | None
    failure_reasons: tuple[str, ...] = ()
    agent_id: str = ""
    node_id: str | None = None

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.critic.verdict"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "recommended_action": self.recommended_action,
            "layer": self.layer,
            "score": self.score,
            "failure_reasons": list(self.failure_reasons),
            "agent_id": self.agent_id,
            "node_id": self.node_id,
        }

    def redact(self) -> CriticVerdictDiagV1:
        return self


class CriticTraceEmitter:
    """
    Emits ``critic.*`` trace steps for partial and final verification.

    Writes to an optional ``RunTraceWriter`` and/or ``RuntimeEventBus`` (via trace bridge).
    """

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

    def emit_verdict(
        self,
        request: CriticRequest,
        verdict: CriticVerdict,
        *,
        tenant_id: str,
        task_id: str,
        agent_id: str,
        node_id: str | None = None,
    ) -> list[TraceEvent]:
        emitted: list[TraceEvent] = []
        for layer in verdict.layers:
            layer_evt = self._emit_layer(
                request,
                verdict,
                layer,
                tenant_id=tenant_id,
                task_id=task_id,
                agent_id=agent_id,
                node_id=node_id,
            )
            if layer_evt is not None:
                emitted.append(layer_evt)
        if verdict.scope is CriticScope.GRAPH_FINAL:
            emitted.append(
                self._emit_final(
                    request,
                    verdict,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    agent_id=agent_id,
                )
            )
        return emitted

    def emit_evaluator_loop(
        self,
        *,
        tenant_id: str,
        task_id: str,
        agent_id: str,
        iteration: int,
        passed: bool,
        node_id: str | None = None,
    ) -> TraceEvent:
        payload = CriticVerdictDiagV1(
            scope=CriticScope.NODE_PARTIAL.value,
            passed=passed,
            recommended_action="continue" if passed else "revise",
            layer="evaluator_loop",
            score=None,
            failure_reasons=(),
            agent_id=agent_id,
            node_id=node_id,
        )
        return self._emit(
            step=CRITIC_STEP_EVALUATOR_LOOP,
            message=f"critic evaluator loop iteration {iteration} passed={passed}",
            level=TraceLevel.INFO if passed else TraceLevel.WARNING,
            payload=payload,
            tags={
                "tenant_id": tenant_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "iteration": iteration,
                **({"node_id": node_id} if node_id else {}),
            },
        )

    def _emit_layer(
        self,
        request: CriticRequest,
        verdict: CriticVerdict,
        layer: LayerVerdict,
        *,
        tenant_id: str,
        task_id: str,
        agent_id: str,
        node_id: str | None,
    ) -> TraceEvent | None:
        step_level = _step_and_level_for_layer(layer)
        if step_level is None:
            return None
        step, level = step_level
        payload = CriticVerdictDiagV1(
            scope=verdict.scope.value,
            passed=layer.passed,
            recommended_action=verdict.recommended_action.value,
            layer=layer.layer.value,
            score=layer.score,
            failure_reasons=tuple(layer.errors),
            agent_id=agent_id,
            node_id=node_id,
        )
        scope_label = request.scope.value.replace("_", " ")
        status = "passed" if layer.passed else "failed"
        return self._emit(
            step=step,
            message=f"critic {scope_label} {layer.layer.value} {status}",
            level=level,
            payload=payload,
            tags={
                "tenant_id": tenant_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "critic_scope": request.scope.value,
                **({"node_id": node_id} if node_id else {}),
            },
        )

    def _emit_final(
        self,
        request: CriticRequest,
        verdict: CriticVerdict,
        *,
        tenant_id: str,
        task_id: str,
        agent_id: str,
    ) -> TraceEvent:
        payload = CriticVerdictDiagV1(
            scope=verdict.scope.value,
            passed=verdict.passed,
            recommended_action=verdict.recommended_action.value,
            layer="final",
            score=_aggregate_score(verdict),
            failure_reasons=tuple(verdict.failure_reasons),
            agent_id=agent_id,
            node_id=None,
        )
        status = "passed" if verdict.passed else "failed"
        return self._emit(
            step=CRITIC_STEP_FINAL_VERDICT,
            message=f"critic final verdict {status}",
            level=TraceLevel.INFO if verdict.passed else TraceLevel.ERROR,
            payload=payload,
            tags={
                "tenant_id": tenant_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "critic_scope": request.scope.value,
            },
        )

    def _emit(
        self,
        *,
        step: str,
        message: str,
        level: TraceLevel,
        payload: CriticVerdictDiagV1,
        tags: dict[str, Any],
    ) -> TraceEvent:
        self._seq += 1
        evt = TraceEvent(
            event_id=TraceEvent.new_id(),
            run_id=self._run_id,
            seq=self._seq,
            ts_utc=utc_now_iso(),
            level=level,
            component=TraceComponent.CRITIC,
            step=step,
            message=message,
            payload=payload,
            tags=tags,
        )
        self.events.append(evt)
        if self._trace_writer is not None:
            self._trace_writer.append_event(evt)
        if self._event_bus is not None:
            from intergrax.contracts.execution_identity import (
                require_active_execution_identity,
                validate_task_id,
            )
            from intergrax.runtime.events.trace_bridge import (
                trace_bridge_subject_from_tags,
                trace_event_to_runtime_event,
            )

            resolved_task_id = validate_task_id(tags.get("task_id"))
            active_run_id, attempt_id = require_active_execution_identity()
            subject = trace_bridge_subject_from_tags(
                tenant_id=str(tags.get("tenant_id", "default")),
                task_id=resolved_task_id,
                agent_id=str(tags.get("agent_id", "")),
            )
            self._event_bus.record(
                trace_event_to_runtime_event(
                    evt,
                    subject,
                    run_id=active_run_id,
                    attempt_id=attempt_id,
                )
            )
        return evt


def _step_and_level_for_layer(layer: LayerVerdict) -> tuple[str, TraceLevel] | None:
    if layer.layer is CriticLayer.L0_DETERMINISTIC:
        if layer.passed:
            return None
        return CRITIC_STEP_L0_FAILED, TraceLevel.ERROR
    if layer.layer is CriticLayer.L1_SEMANTIC:
        level = TraceLevel.INFO if layer.passed else TraceLevel.WARNING
        return CRITIC_STEP_L1_JUDGE, level
    if layer.layer is CriticLayer.L1_TRAJECTORY:
        level = TraceLevel.INFO if layer.passed else TraceLevel.WARNING
        return CRITIC_STEP_TRAJECTORY, level
    level = TraceLevel.ERROR if not layer.passed else TraceLevel.INFO
    return CRITIC_STEP_L0_FAILED, level


def _aggregate_score(verdict: CriticVerdict) -> float | None:
    scores = [layer.score for layer in verdict.layers if layer.score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def build_critic_trace_emitter(
    *,
    run_id: str,
    trace_writer: RunTraceWriter | None = None,
    event_bus: RuntimeEventBus | None = None,
    seq_offset: int = 0,
) -> CriticTraceEmitter:
    """Factory for graph-scoped critic trace emission."""
    return CriticTraceEmitter(
        run_id=run_id,
        trace_writer=trace_writer,
        event_bus=event_bus,
        seq_offset=seq_offset,
    )
