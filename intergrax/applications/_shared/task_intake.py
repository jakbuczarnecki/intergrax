# © Artur Czarnecki. All rights reserved.

"""Typed Tier-3 task intake helpers (Phase Q+-M.2)."""

from __future__ import annotations

from typing import Optional

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.task.task import Task


def apply_orchestration_graph_id(task: Task, graph_id: Optional[str]) -> Task:
    """Set ``TaskRuntimeState.orchestration.graph_id`` (canonical) instead of flat metadata keys."""
    normalized = (graph_id or "").strip()
    if not normalized:
        return task
    updated = task.model_copy(
        update={
            "runtime": task.runtime.model_copy(
                update={
                    "orchestration": task.runtime.orchestration.model_copy(
                        update={"graph_id": normalized},
                    ),
                },
            ),
        },
    )
    updated.sync_metadata()
    return updated


def apply_long_running_enabled(
    task: Task,
    *,
    enabled: bool,
    checkpoint_on_pause: bool = True,
) -> Task:
    """Enable durable execution via ``TaskExecutionOptions.long_running`` only."""
    if not enabled:
        return task
    long_running = task.options.long_running.model_copy(
        update={
            "enabled": True,
            "checkpoint_on_pause": checkpoint_on_pause,
        },
        deep=True,
    )
    updated = task.model_copy(
        update={"options": task.options.model_copy(update={"long_running": long_running}, deep=True)},
    )
    updated.sync_metadata()
    return updated


def apply_long_running_from_profile(
    task: Task,
    env: ApplicationEnvironmentProfile,
    *,
    enabled: bool | None = None,
    checkpoint_on_pause: bool = True,
) -> Task:
    """
    Enable long-running execution when the host profile opts in (ORCH-CONFIG.6).

    ``enabled`` overrides profile when set; otherwise uses
    ``orchestration_profile.long_running_enabled``.
    """
    profile_enabled = env.orchestration_profile.long_running_enabled
    should_enable = profile_enabled if enabled is None else enabled
    return apply_long_running_enabled(
        task,
        enabled=should_enable,
        checkpoint_on_pause=checkpoint_on_pause,
    )


def apply_orchestration_task_defaults(
    task: Task,
    env: ApplicationEnvironmentProfile,
    *,
    long_running: bool | None = None,
) -> Task:
    """Apply Tier-3 orchestration intake defaults from ``ApplicationEnvironmentProfile``."""
    return apply_long_running_from_profile(task, env, enabled=long_running)
