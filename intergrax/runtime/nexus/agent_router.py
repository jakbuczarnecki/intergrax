# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Callable, Optional

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext


class AgentRouter:
    """
    Select agents from registry using explicit agent_id or capability matching (§10.4, §16).
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def route(self, task: Task) -> Agent:
        if task.agent_id and self._registry.has(task.agent_id):
            return self._registry.get(task.agent_id)

        if task.context.capability:
            matches = self._registry.find_by_capability(task.context.capability)
            if matches:
                best = self._best_capability_match(task.context, matches)
                if best is not None:
                    return best

        match = self._registry.find_best_match(task.context)
        if match is not None:
            return match

        ids = self._registry.list_agent_ids()
        if not ids:
            raise RuntimeError("AgentRegistry is empty.")
        return self._registry.get(ids[0])

    def _best_capability_match(
        self,
        context: TaskContext,
        candidates: list[Agent],
    ) -> Optional[Agent]:
        best: Optional[tuple[float, Agent]] = None
        for agent in candidates:
            result = agent.can_handle(context)
            if not result.matched:
                continue
            if best is None or result.score > best[0]:
                best = (result.score, agent)
        return best[1] if best else candidates[0]
