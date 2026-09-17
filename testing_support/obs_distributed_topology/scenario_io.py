# © Artur Czarnecki. All rights reserved.

"""JSON serialization for DG-005 scenarios (worker IPC)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.knowledge.contracts.validation import JsonObject, JsonValue

from testing_support.obs_distributed_topology.models import (
    Dg005Scenario,
    PlannedRuntimeEvent,
)
from testing_support.obs_distributed_topology.provider_contract import (
    EvidenceProviderDescriptor,
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


def _provider_to_dict(descriptor: EvidenceProviderDescriptor) -> JsonObject:
    return {
        "provider_id": descriptor.provider_id,
        "config": dict(descriptor.config),
    }


def _provider_from_dict(payload: JsonObject) -> EvidenceProviderDescriptor:
    if "provider_id" not in payload:
        raise ValueError("scenario provider requires provider_id")
    provider_id = payload["provider_id"]
    if not isinstance(provider_id, str) or not provider_id.strip():
        raise ValueError("scenario provider provider_id must be a non-empty str")
    if "config" not in payload:
        raise ValueError("scenario provider requires config")
    raw_config = payload["config"]
    if not isinstance(raw_config, dict):
        raise ValueError("scenario provider config must be a JSON object")
    config: JsonObject = {}
    for key, value in raw_config.items():
        if not isinstance(key, str):
            raise ValueError("scenario provider config keys must be strings")
        config[key] = cast(JsonValue, value)
    return EvidenceProviderDescriptor(provider_id=provider_id, config=config)


def scenario_to_dict(scenario: Dg005Scenario) -> JsonObject:
    payload: JsonObject = {
        "qualification_sha": scenario.qualification_sha,
        "provider": _provider_to_dict(scenario.provider),
        "primary_tenant": scenario.primary_tenant,
        "foreign_tenant": scenario.foreign_tenant,
        "primary_task_id": str(scenario.primary_task_id),
        "primary_run_id": str(scenario.primary_run_id),
        "isolated_run_id": str(scenario.isolated_run_id),
        "foreign_run_id": str(scenario.foreign_run_id),
        "primary_attempt_id": str(scenario.primary_attempt_id),
        "primary_execution_id": str(scenario.primary_execution_id),
        "primary_events": cast(
            JsonValue,
            [_planned_to_dict(e) for e in scenario.primary_events],
        ),
        "isolated_run_events": cast(
            JsonValue,
            [_planned_to_dict(e) for e in scenario.isolated_run_events],
        ),
        "foreign_tenant_events": cast(
            JsonValue,
            [_planned_to_dict(e) for e in scenario.foreign_tenant_events],
        ),
        "diagnostics_task_id": str(scenario.diagnostics_task_id),
        "diagnostics_run_id": str(scenario.diagnostics_run_id),
        "diagnostics_attempt_id": str(scenario.diagnostics_attempt_id),
        "idempotent_event": cast(
            JsonValue,
            _planned_to_dict(scenario.idempotent_event),
        ),
        "reconstruction_initial_limit": scenario.reconstruction_initial_limit,
        "as_of_position_index": scenario.as_of_position_index,
    }
    return payload


def scenario_from_dict(payload: JsonObject) -> Dg005Scenario:
    return Dg005Scenario(
        qualification_sha=str(payload["qualification_sha"]),
        provider=_provider_from_dict(cast(JsonObject, payload["provider"])),
        primary_tenant=str(payload["primary_tenant"]),
        foreign_tenant=str(payload["foreign_tenant"]),
        primary_task_id=TaskId(str(payload["primary_task_id"])),
        primary_run_id=RunId(str(payload["primary_run_id"])),
        isolated_run_id=RunId(str(payload["isolated_run_id"])),
        foreign_run_id=RunId(str(payload["foreign_run_id"])),
        primary_attempt_id=AttemptId(str(payload["primary_attempt_id"])),
        primary_execution_id=ExecutionId(str(payload["primary_execution_id"])),
        primary_events=tuple(
            _planned_from_dict(cast(dict[str, str], e))
            for e in cast(list[object], payload["primary_events"])
        ),
        isolated_run_events=tuple(
            _planned_from_dict(cast(dict[str, str], e))
            for e in cast(list[object], payload["isolated_run_events"])
        ),
        foreign_tenant_events=tuple(
            _planned_from_dict(cast(dict[str, str], e))
            for e in cast(list[object], payload["foreign_tenant_events"])
        ),
        diagnostics_task_id=TaskId(str(payload["diagnostics_task_id"])),
        diagnostics_run_id=RunId(str(payload["diagnostics_run_id"])),
        diagnostics_attempt_id=AttemptId(str(payload["diagnostics_attempt_id"])),
        idempotent_event=_planned_from_dict(
            cast(dict[str, str], payload["idempotent_event"]),
        ),
        reconstruction_initial_limit=int(
            cast(int | str | float, payload["reconstruction_initial_limit"]),
        ),
        as_of_position_index=int(
            cast(int | str | float, payload["as_of_position_index"]),
        ),
    )


def write_scenario(path: Path, scenario: Dg005Scenario) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scenario_to_dict(scenario), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_scenario(path: Path) -> Dg005Scenario:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("scenario JSON root must be an object")
    return scenario_from_dict(cast(JsonObject, loaded))
