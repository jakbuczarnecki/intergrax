# © Artur Czarnecki. All rights reserved.

import inspect

import pytest

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_catalog import phase_for_event
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.runtime.events.schema_guard import RuntimeEventSchemaError, assert_runtime_event_schema
from intergrax.runtime.nexus.orchestration import planning_runner

pytestmark = pytest.mark.gate


def _event(**overrides) -> RuntimeEvent:
    base = runtime_event_test_identity()
    base.update(
        {
            "tenant_id": "tenant_1",
            "event_type": RuntimeEventType.TASK_CREATED,
            "phase": ExecutionPhase.INTAKE,
            "severity": EventSeverity.INFO,
        }
    )
    base.update(overrides)
    return RuntimeEvent(**base)


def test_decision_emitted_catalog_phase_is_step_execution() -> None:
    assert phase_for_event(RuntimeEventType.DECISION_EMITTED) is ExecutionPhase.STEP_EXECUTION


def test_decision_emitted_step_execution_passes_schema() -> None:
    assert_runtime_event_schema(
        _event(
            event_type=RuntimeEventType.DECISION_EMITTED,
            phase=ExecutionPhase.STEP_EXECUTION,
            payload={
                "step_id": "step_1",
                "decision": "continue",
                "policy_action": "allow",
            },
        )
    )


def test_decision_emitted_planning_phase_rejected() -> None:
    with pytest.raises(RuntimeEventSchemaError, match="phase mismatch for decision_emitted"):
        assert_runtime_event_schema(
            _event(
                event_type=RuntimeEventType.DECISION_EMITTED,
                phase=ExecutionPhase.PLANNING,
                payload={"decision_type": "nexus_planning"},
            )
        )


def test_plan_created_planning_phase_passes_schema_with_decision_record() -> None:
    assert_runtime_event_schema(
        _event(
            event_type=RuntimeEventType.PLAN_CREATED,
            phase=ExecutionPhase.PLANNING,
            payload={
                "plan_id": "plan_1",
                "step_count": 1,
                "task_state": "planned",
                "decision_record": {
                    "decision_type": "nexus_planning",
                    "policy_action": "allow",
                    "metadata": {"classification": "standard", "planner_source": "default"},
                },
            },
        )
    )


def test_planning_runner_does_not_emit_decision_emitted() -> None:
    source = inspect.getsource(planning_runner.NexusPlanningRunner.run)
    assert "RuntimeEventType.DECISION_EMITTED" not in source
