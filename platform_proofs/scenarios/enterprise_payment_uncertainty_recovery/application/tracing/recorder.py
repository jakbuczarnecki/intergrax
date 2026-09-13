"""Records canonical platform TraceEvent rows for scenario qualification runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from intergrax.contracts.execution_identity import ExecutionId, RunId, mint_execution_id, mint_run_id
from intergrax.runtime.events.w3c_trace_context import generate_trace_id
from intergrax.runtime.nexus.tracing.trace_models import (
    TraceComponent,
    TraceEvent,
    TraceLevel,
    utc_now_iso,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.tracing.diagnostics import (
    ErlQual004LifecycleStepDiagV1,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.tracing.port import (
    ScenarioExecutionTraceStepId,
)

_COMPONENT = TraceComponent.RUNTIME


@dataclass(frozen=True, slots=True)
class ScenarioExecutionTraceScope:
    trace_id: str
    correlation_id: str
    execution_id: ExecutionId
    scenario_id: str
    variant_id: str
    run_id: RunId

    @staticmethod
    def mint(
        *,
        correlation_id: str,
        scenario_id: str,
        variant_id: str,
    ) -> ScenarioExecutionTraceScope:
        return ScenarioExecutionTraceScope(
            trace_id=generate_trace_id(),
            correlation_id=correlation_id,
            execution_id=mint_execution_id(),
            scenario_id=scenario_id,
            variant_id=variant_id,
            run_id=mint_run_id(),
        )


@dataclass
class RecordingScenarioExecutionTrace:
    """In-memory TraceEvent recorder for lab execution and unit tests."""

    scope: ScenarioExecutionTraceScope
    _events: list[TraceEvent] = field(default_factory=list)
    _seq: int = 0

    def begin_execution(
        self,
        *,
        correlation_id: str,
        scenario_id: str,
        variant_id: str,
    ) -> None:
        self.scope = ScenarioExecutionTraceScope.mint(
            correlation_id=correlation_id,
            scenario_id=scenario_id,
            variant_id=variant_id,
        )
        self._events.clear()
        self._seq = 0

    def emit_lifecycle_step(
        self,
        step_id: ScenarioExecutionTraceStepId,
        *,
        outcome: str,
        component_identity: str,
        business_detail: dict[str, Any] | None = None,
    ) -> None:
        self._seq += 1
        detail = business_detail or {}
        payload = ErlQual004LifecycleStepDiagV1(
            trace_id=self.scope.trace_id,
            correlation_id=self.scope.correlation_id,
            execution_id=str(self.scope.execution_id),
            scenario_id=self.scope.scenario_id,
            step_id=step_id.value,
            outcome=outcome,
            variant_id=self.scope.variant_id,
            component_identity=component_identity,
            business_detail=detail,
        )
        event = TraceEvent(
            event_id=TraceEvent.new_id(),
            run_id=str(self.scope.run_id),
            seq=self._seq,
            ts_utc=utc_now_iso(),
            level=TraceLevel.INFO,
            component=_COMPONENT,
            step=f"erl_qual_004.{step_id.value}",
            message=step_id.value.replace("_", " "),
            payload=payload,
            tags={
                "trace_id": self.scope.trace_id,
                "correlation_id": self.scope.correlation_id,
                "execution_id": str(self.scope.execution_id),
                "scenario_id": self.scope.scenario_id,
                "step_id": step_id.value,
                "outcome": outcome,
                "component_identity": component_identity,
            },
        )
        self._events.append(event)

    def snapshot(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)


class NullScenarioExecutionTrace:
    """No-op trace port when observability is not required."""

    def emit_lifecycle_step(
        self,
        step_id: ScenarioExecutionTraceStepId,
        *,
        outcome: str,
        component_identity: str,
        business_detail: dict[str, Any] | None = None,
    ) -> None:
        return None

    def snapshot(self) -> tuple[TraceEvent, ...]:
        return ()
