# © Artur Czarnecki. All rights reserved.

"""Minimal canonical harness task route mounting for task-control unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI

from intergrax.applications._shared.harness_task_routes import mount_canonical_harness_task_routes
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.interactions.task_executor import HostTaskExecutionExecutor
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence


def mount_canonical_harness_task_routes_for_tests(
    app: FastAPI,
    *,
    host_execution: HostTaskExecutionPort | None = None,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
) -> HostTaskExecutionPort:
    resolved_host = host_execution if host_execution is not None else AsyncMock()
    mount_canonical_harness_task_routes(
        app,
        task_executor=HostTaskExecutionExecutor(resolved_host),
        host_execution=resolved_host,
        checkpoint_store=checkpoint_store,
        mutation_boundary=mutation_boundary,
    )
    return resolved_host
