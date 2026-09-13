# © Artur Czarnecki. All rights reserved.

"""ERL-QUAL-004 execution tracing — lifecycle visibility and correlation."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.runtime.nexus.tracing.trace_models import TraceEvent
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution import (
    EnterprisePaymentScenarioExecutionRequest,
    ScenarioLifecycleOutcome,
    build_lab_execution_composition,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.tracing.diagnostics import (
    ErlQual004LifecycleStepDiagV1,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.tracing.port import (
    ScenarioExecutionTraceStepId,
    TraceBusinessDetail,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.tracing.recorder import (
    NullScenarioExecutionTrace,
    RecordingScenarioExecutionTrace,
    ScenarioExecutionTraceScope,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCENARIO_ROOT = _REPO_ROOT / "platform_proofs/scenarios/enterprise_payment_uncertainty_recovery"
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"

_FULL_LIFECYCLE_STEPS = frozenset(ScenarioExecutionTraceStepId)


def _run(variant_id: str):
    executor = build_lab_execution_composition()
    return executor.execute(
        EnterprisePaymentScenarioExecutionRequest(
            variant_id=variant_id,
            run_id=f"trace-{variant_id}",
        ),
    )


def _step_ids(events: tuple[TraceEvent, ...]) -> list[str]:
    ids: list[str] = []
    for event in events:
        payload = event.payload
        if isinstance(payload, ErlQual004LifecycleStepDiagV1):
            ids.append(payload.step_id)
        elif event.tags.get("step_id"):
            ids.append(str(event.tags["step_id"]))
    return ids


def _correlation_ids(events: tuple[TraceEvent, ...]) -> set[str]:
    values: set[str] = set()
    for event in events:
        payload = event.payload
        if isinstance(payload, ErlQual004LifecycleStepDiagV1):
            values.add(payload.correlation_id)
        tag = event.tags.get("correlation_id")
        if tag:
            values.add(str(tag))
    return values


def test_variant_a_success_trace_has_full_lifecycle() -> None:
    result = _run("payment_completed_after_unknown")
    assert result.lifecycle_outcome is ScenarioLifecycleOutcome.RECOVERY_CONTINUATION
    steps = set(_step_ids(result.execution_trace_events))
    assert _FULL_LIFECYCLE_STEPS.issubset(steps)


def test_variant_b_failed_payment_trace_shows_controlled_stop() -> None:
    result = _run("payment_failed_after_unknown")
    assert result.lifecycle_outcome is ScenarioLifecycleOutcome.CONTROLLED_STOP
    steps = _step_ids(result.execution_trace_events)
    resolution_events = [
        event
        for event in result.execution_trace_events
        if isinstance(event.payload, ErlQual004LifecycleStepDiagV1)
        and event.payload.step_id == ScenarioExecutionTraceStepId.RESOLUTION_DECIDED.value
    ]
    assert resolution_events
    resolution_payload = resolution_events[0].payload
    assert isinstance(resolution_payload, ErlQual004LifecycleStepDiagV1)
    assert resolution_payload.outcome == "stop"
    assert ScenarioExecutionTraceStepId.RECOVERY_EXECUTED.value in steps


def test_variant_c_unavailable_truth_trace_shows_escalation() -> None:
    result = _run("payment_truth_unavailable")
    assert result.lifecycle_outcome is ScenarioLifecycleOutcome.SAFE_ESCALATION_OR_WAIT
    recovery_events = [
        event
        for event in result.execution_trace_events
        if isinstance(event.payload, ErlQual004LifecycleStepDiagV1)
        and event.payload.step_id == ScenarioExecutionTraceStepId.RECOVERY_EXECUTED.value
    ]
    assert recovery_events
    recovery_payload = recovery_events[0].payload
    assert isinstance(recovery_payload, ErlQual004LifecycleStepDiagV1)
    assert recovery_payload.outcome == "escalate"
    unknown_steps = [
        event
        for event in result.execution_trace_events
        if isinstance(event.payload, ErlQual004LifecycleStepDiagV1)
        and event.payload.step_id == ScenarioExecutionTraceStepId.UNKNOWN_DETECTED.value
    ]
    assert unknown_steps


def test_trace_correlation_id_stable_across_payment_erl_and_recovery() -> None:
    result = _run("payment_completed_after_unknown")
    assert len(result.execution_trace_events) >= len(_FULL_LIFECYCLE_STEPS)
    correlation_values = _correlation_ids(result.execution_trace_events)
    assert len(correlation_values) == 1
    assert correlation_values.pop() == result.correlation_id
    trace_ids = {
        event.payload.trace_id
        for event in result.execution_trace_events
        if isinstance(event.payload, ErlQual004LifecycleStepDiagV1)
    }
    assert len(trace_ids) == 1
    execution_ids = {
        event.payload.execution_id
        for event in result.execution_trace_events
        if isinstance(event.payload, ErlQual004LifecycleStepDiagV1)
    }
    assert len(execution_ids) == 1


def test_tracing_architecture_no_duplicate_framework() -> None:
    forbidden_names = frozenset(
        {
            "PaymentTrace",
            "ScenarioTrace",
            "CustomEventBus",
            "CustomTelemetry",
        }
    )
    violations: list[str] = []
    for path in _SCENARIO_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in forbidden_names:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{node.name}")
    assert not violations, violations

    needle = "enterprise_payment_uncertainty_recovery"
    intergrax_hits: list[str] = []
    for path in _INTERGRAX_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if needle in text:
            intergrax_hits.append(str(path.relative_to(_REPO_ROOT)))
    assert not intergrax_hits


_RECORDER_ADAPTER_SYMBOL = "RecordingScenarioExecutionTrace"
_ALLOWED_CONCRETE_RECORDER_IMPORTS = frozenset(
    {
        "application/execution/composition.py",
        "application/tracing/__init__.py",
        "application/tracing/recorder.py",
    }
)
_ORCHESTRATION_SCAN_ROOTS = (
    _SCENARIO_ROOT / "application" / "execution",
    _SCENARIO_ROOT / "application" / "services",
    _SCENARIO_ROOT / "external_payment" / "adapters",
)


def _imports_recording_trace_adapter(module_path: Path) -> bool:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if not node.module.endswith(".tracing.recorder"):
                continue
            for alias in node.names:
                if alias.name == _RECORDER_ADAPTER_SYMBOL:
                    return True
    return False


def test_execution_orchestration_does_not_import_concrete_trace_recorder() -> None:
    violations: list[str] = []
    for root in _ORCHESTRATION_SCAN_ROOTS:
        for path in root.rglob("*.py"):
            rel = path.relative_to(_SCENARIO_ROOT).as_posix()
            if rel in _ALLOWED_CONCRETE_RECORDER_IMPORTS:
                continue
            if _imports_recording_trace_adapter(path):
                violations.append(rel)
    assert not violations, violations


@dataclass
class _SubstituteScenarioExecutionTrace:
    """Test double — not derived from RecordingScenarioExecutionTrace."""

    begin_calls: list[tuple[str, str, str]] = field(default_factory=list)
    emitted_steps: list[ScenarioExecutionTraceStepId] = field(default_factory=list)

    def begin_execution(
        self,
        *,
        correlation_id: str,
        scenario_id: str,
        variant_id: str,
    ) -> None:
        self.begin_calls.append((correlation_id, scenario_id, variant_id))

    def emit_lifecycle_step(
        self,
        step_id: ScenarioExecutionTraceStepId,
        *,
        outcome: str,
        component_identity: str,
        business_detail: TraceBusinessDetail | None = None,
    ) -> None:
        _ = (outcome, component_identity, business_detail)
        self.emitted_steps.append(step_id)

    def snapshot(self) -> tuple[TraceEvent, ...]:
        return ()


def test_executor_operates_with_substitute_trace_port_not_recording() -> None:
    substitute = _SubstituteScenarioExecutionTrace()
    executor = build_lab_execution_composition(execution_trace=substitute)
    result = executor.execute(
        EnterprisePaymentScenarioExecutionRequest(
            variant_id="payment_completed_after_unknown",
            run_id="substitute-trace-port",
        ),
    )
    assert result.lifecycle_outcome is ScenarioLifecycleOutcome.RECOVERY_CONTINUATION
    assert len(substitute.begin_calls) == 1
    assert substitute.begin_calls[0][1] == "ERL-QUAL-004"
    assert ScenarioExecutionTraceStepId.SCENARIO_EXECUTION_STARTED in substitute.emitted_steps
    assert ScenarioExecutionTraceStepId.SCENARIO_COMPLETED in substitute.emitted_steps
    assert result.execution_trace_events == ()


def test_null_trace_adapter_satisfies_full_port_lifecycle() -> None:
    trace = NullScenarioExecutionTrace()
    trace.begin_execution(
        correlation_id="corr-null",
        scenario_id="ERL-QUAL-004",
        variant_id="payment_completed_after_unknown",
    )
    for step_id in ScenarioExecutionTraceStepId:
        trace.emit_lifecycle_step(
            step_id,
            outcome="noop",
            component_identity="test.null_trace",
        )
    assert trace.snapshot() == ()


def test_recording_trace_adapter_satisfies_full_port_lifecycle() -> None:
    trace = RecordingScenarioExecutionTrace(
        scope=ScenarioExecutionTraceScope.mint(
            correlation_id="corr-rec",
            scenario_id="ERL-QUAL-004",
            variant_id="payment_completed_after_unknown",
        ),
    )
    trace.begin_execution(
        correlation_id="corr-rec-run",
        scenario_id="ERL-QUAL-004",
        variant_id="payment_failed_after_unknown",
    )
    trace.emit_lifecycle_step(
        ScenarioExecutionTraceStepId.SCENARIO_EXECUTION_STARTED,
        outcome="started",
        component_identity="test.recording_trace",
        business_detail={"run_id": "r1"},
    )
    events = trace.snapshot()
    assert len(events) == 1
    payload = events[0].payload
    assert isinstance(payload, ErlQual004LifecycleStepDiagV1)
    assert payload.correlation_id == "corr-rec-run"
    assert payload.trace_id
    assert payload.execution_id
