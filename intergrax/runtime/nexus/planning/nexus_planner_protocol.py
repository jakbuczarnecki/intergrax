# © Artur Czarnecki. All rights reserved.

"""Nexus task planner protocol (Phase ORCH-1)."""

from __future__ import annotations

from typing import Protocol

from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task


class NexusTaskPlannerProtocol(Protocol):
    """Structured Nexus plan before ``ExecutionGraph`` materialization."""

    def plan(self, task: Task, registry: AgentRegistry) -> NexusPlan:
        """Produce a task-level plan for the given registry and task."""
