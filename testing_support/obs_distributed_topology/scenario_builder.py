# © Artur Czarnecki. All rights reserved.

"""Build deterministic DG-005 qualification scenarios."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType

from testing_support.obs_distributed_topology.models import (
    Dg005Scenario,
    PlannedRuntimeEvent,
)
from testing_support.obs_distributed_topology.providers.sqlite_file import (
    sqlite_file_descriptor,
)

_PRIMARY_EVENT_COUNT = 12
_RECONSTRUCTION_PAGE_LIMIT = 5


def build_dg005_scenario(
    *,
    qualification_sha: str,
    sqlite_db_path: Path,
) -> Dg005Scenario:
    primary_tenant = "tenant-dg005-primary"
    foreign_tenant = "tenant-dg005-foreign"
    primary_task_id = mint_task_id()
    primary_run_id = mint_run_id()
    isolated_run_id = mint_run_id()
    foreign_run_id = mint_run_id()
    primary_attempt_id = mint_attempt_id()
    primary_execution_id = mint_execution_id()

    base_time = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
    primary_events: list[PlannedRuntimeEvent] = []
    for index in range(_PRIMARY_EVENT_COUNT):
        event_id = mint_event_id()
        timestamp = base_time - timedelta(hours=index)
        event_type = (
            RuntimeEventType.STEP_STARTED.value
            if index % 2 == 0
            else RuntimeEventType.STEP_COMPLETED.value
        )
        primary_events.append(
            PlannedRuntimeEvent(
                event_id=event_id,
                tenant_id=primary_tenant,
                task_id=primary_task_id,
                run_id=primary_run_id,
                attempt_id=primary_attempt_id,
                execution_id=primary_execution_id,
                event_type=event_type,
                timestamp_iso=timestamp.isoformat(),
            )
        )

    isolated_run_events = (
        PlannedRuntimeEvent(
            event_id=mint_event_id(),
            tenant_id=primary_tenant,
            task_id=primary_task_id,
            run_id=isolated_run_id,
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            event_type=RuntimeEventType.TASK_CREATED.value,
            timestamp_iso=base_time.isoformat(),
        ),
        PlannedRuntimeEvent(
            event_id=mint_event_id(),
            tenant_id=primary_tenant,
            task_id=primary_task_id,
            run_id=isolated_run_id,
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            event_type=RuntimeEventType.TASK_COMPLETED.value,
            timestamp_iso=(base_time + timedelta(minutes=1)).isoformat(),
        ),
    )

    foreign_tenant_events = (
        PlannedRuntimeEvent(
            event_id=mint_event_id(),
            tenant_id=foreign_tenant,
            task_id=mint_task_id(),
            run_id=foreign_run_id,
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            event_type=RuntimeEventType.STEP_STARTED.value,
            timestamp_iso=base_time.isoformat(),
        ),
    )

    diagnostics_task_id = mint_task_id()
    diagnostics_run_id = mint_run_id()
    diagnostics_attempt_id = mint_attempt_id()
    idempotent_event = PlannedRuntimeEvent(
        event_id=mint_event_id(),
        tenant_id=primary_tenant,
        task_id=primary_task_id,
        run_id=primary_run_id,
        attempt_id=primary_attempt_id,
        execution_id=primary_execution_id,
        event_type=RuntimeEventType.STEP_STARTED.value,
        timestamp_iso=base_time.isoformat(),
    )

    return Dg005Scenario(
        qualification_sha=qualification_sha,
        provider=sqlite_file_descriptor(db_path=sqlite_db_path),
        primary_tenant=primary_tenant,
        foreign_tenant=foreign_tenant,
        primary_task_id=primary_task_id,
        primary_run_id=primary_run_id,
        isolated_run_id=isolated_run_id,
        foreign_run_id=foreign_run_id,
        primary_attempt_id=primary_attempt_id,
        primary_execution_id=primary_execution_id,
        primary_events=tuple(primary_events),
        isolated_run_events=isolated_run_events,
        foreign_tenant_events=foreign_tenant_events,
        diagnostics_task_id=diagnostics_task_id,
        diagnostics_run_id=diagnostics_run_id,
        diagnostics_attempt_id=diagnostics_attempt_id,
        idempotent_event=idempotent_event,
        reconstruction_initial_limit=_RECONSTRUCTION_PAGE_LIMIT,
        as_of_position_index=6,
    )
