# © Artur Czarnecki. All rights reserved.

"""JSON serialization for DG-005 scenarios (worker IPC)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
)

from testing_support.obs_distributed_topology.models import (
    Dg005Scenario,
    PlannedRuntimeEvent,
    SqliteEvidenceProviderConfig,
)


def _planned_to_dict(event: PlannedRuntimeEvent) -> dict[str, str]:
    return {
        "event_id": str(event.event_id),
        "tenant_id": event.tenant_id,
        "task_id": str(event.task_id),
        "run_id": str(event.run_id),
        "attempt_id": str(event.attempt_id),
        "execution_id": str(event.execution_id),
        "event_type": event.event_type,
        "timestamp_iso": event.timestamp_iso,
    }


def _planned_from_dict(payload: dict[str, str]) -> PlannedRuntimeEvent:
    return PlannedRuntimeEvent(
        event_id=EventId(payload["event_id"]),
        tenant_id=payload["tenant_id"],
        task_id=TaskId(payload["task_id"]),
        run_id=RunId(payload["run_id"]),
        attempt_id=AttemptId(payload["attempt_id"]),
        execution_id=ExecutionId(payload["execution_id"]),
        event_type=payload["event_type"],
        timestamp_iso=payload["timestamp_iso"],
    )


def scenario_to_dict(scenario: Dg005Scenario) -> dict[str, Any]:
    return {
        "qualification_sha": scenario.qualification_sha,
        "provider": asdict(scenario.provider),
        "primary_tenant": scenario.primary_tenant,
        "foreign_tenant": scenario.foreign_tenant,
        "primary_task_id": str(scenario.primary_task_id),
        "primary_run_id": str(scenario.primary_run_id),
        "isolated_run_id": str(scenario.isolated_run_id),
        "foreign_run_id": str(scenario.foreign_run_id),
        "primary_attempt_id": str(scenario.primary_attempt_id),
        "primary_execution_id": str(scenario.primary_execution_id),
        "primary_events": [_planned_to_dict(e) for e in scenario.primary_events],
        "isolated_run_events": [_planned_to_dict(e) for e in scenario.isolated_run_events],
        "foreign_tenant_events": [_planned_to_dict(e) for e in scenario.foreign_tenant_events],
        "diagnostics_task_id": str(scenario.diagnostics_task_id),
        "diagnostics_run_id": str(scenario.diagnostics_run_id),
        "diagnostics_attempt_id": str(scenario.diagnostics_attempt_id),
        "idempotent_event": _planned_to_dict(scenario.idempotent_event),
        "reconstruction_initial_limit": scenario.reconstruction_initial_limit,
        "as_of_position_index": scenario.as_of_position_index,
    }


def scenario_from_dict(payload: dict[str, Any]) -> Dg005Scenario:
    return Dg005Scenario(
        qualification_sha=payload["qualification_sha"],
        provider=SqliteEvidenceProviderConfig(**payload["provider"]),
        primary_tenant=payload["primary_tenant"],
        foreign_tenant=payload["foreign_tenant"],
        primary_task_id=TaskId(payload["primary_task_id"]),
        primary_run_id=RunId(payload["primary_run_id"]),
        isolated_run_id=RunId(payload["isolated_run_id"]),
        foreign_run_id=RunId(payload["foreign_run_id"]),
        primary_attempt_id=AttemptId(payload["primary_attempt_id"]),
        primary_execution_id=ExecutionId(payload["primary_execution_id"]),
        primary_events=tuple(_planned_from_dict(e) for e in payload["primary_events"]),
        isolated_run_events=tuple(
            _planned_from_dict(e) for e in payload["isolated_run_events"]
        ),
        foreign_tenant_events=tuple(
            _planned_from_dict(e) for e in payload["foreign_tenant_events"]
        ),
        diagnostics_task_id=TaskId(payload["diagnostics_task_id"]),
        diagnostics_run_id=RunId(payload["diagnostics_run_id"]),
        diagnostics_attempt_id=AttemptId(payload["diagnostics_attempt_id"]),
        idempotent_event=_planned_from_dict(payload["idempotent_event"]),
        reconstruction_initial_limit=int(payload["reconstruction_initial_limit"]),
        as_of_position_index=int(payload["as_of_position_index"]),
    )


def write_scenario(path: Path, scenario: Dg005Scenario) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scenario_to_dict(scenario), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_scenario(path: Path) -> Dg005Scenario:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return scenario_from_dict(payload)
