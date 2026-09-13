# © Artur Czarnecki. All rights reserved.

"""ERL-QUAL-004 execution tracing — lifecycle visibility and correlation."""

from __future__ import annotations

import ast
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
