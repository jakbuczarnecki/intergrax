# © Artur Czarnecki. All rights reserved.

"""Tier-3 harness task control surfaces (H-APP-WIRING, FLOW-CTL, REL-ADV)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

from intergrax.applications._shared.async_task_index_resolver import resolve_async_task_index
from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
from intergrax.applications._shared.reliability_wiring import apply_reliability_task_defaults
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

TaskEnricher = Callable[[Task], Task]


def build_reliability_task_enricher(
    env: ApplicationEnvironmentProfile,
    *,
    extra: TaskEnricher | None = None,
) -> TaskEnricher:
    """Apply REL-ADV defaults on every task before Nexus execution."""

    def enricher(task: Task) -> Task:
        enriched = apply_reliability_task_defaults(task, env)
        if extra is not None:
            enriched = extra(enriched)
        return enriched

    return enricher


def wire_harness_task_control(
    app: FastAPI,
    *,
    enabled: bool,
    task_runner: UnifiedTaskRunner,
    env: ApplicationEnvironmentProfile,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    task_route_prefix: str = "/v1/tasks",
    extra_enricher: TaskEnricher | None = None,
) -> TaskEnricher:
    """
    Mount optional harness task HTTP routes and return a task enricher for intake/MCP.

    When ``enabled`` is False, only the enricher is returned (reliability defaults still apply).
    """
    enricher = build_reliability_task_enricher(env, extra=extra_enricher)
    if enabled:
        async_index = resolve_async_task_index(env)
        mount_harness_task_routes(
            app,
            task_runner=task_runner,
            checkpoint_store=checkpoint_store,
            prefix=task_route_prefix,
            task_enricher=enricher,
            async_index=async_index,
        )
    return enricher


def build_task_runner_with_enricher(
    nexus_loop: Any,
    enricher: TaskEnricher | None,
) -> UnifiedTaskRunner:
    """UnifiedTaskRunner that applies enricher on every ``run_task`` / ``run_runtime_request``."""
    return UnifiedTaskRunner(nexus_loop, task_enricher=enricher)
