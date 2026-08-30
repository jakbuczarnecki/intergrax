# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.execution_identity import (
    require_active_execution_id,
    require_active_execution_identity,
    validate_run_id,
)
from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads import AgentSelectionPayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.capability_routing import (
    select_best_capability_match,
    validate_task_for_capability_routing,
)
from intergrax.runtime.task.task import Task, TaskContext


@dataclass(frozen=True)
class AgentRouteSelection:
    requested_agent_id: str
    selected_agent_id: str
    capability: str = ""
    match_score: float | None = None
    selection_reason: str = ""
    fallback_used: bool = False


class AgentRouter:
    """
    Select agents from registry using explicit agent_id or capability matching (§10.4, §16).
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        production_mode: bool = False,
        event_bus: Optional[RuntimeEventBus] = None,
    ) -> None:
        self._registry = registry
        self._production_mode = production_mode
        self._event_bus = event_bus

    def route(
        self,
        task: Task,
        *,
        run_id: str | None = None,
        node_id: str | None = None,
    ) -> Agent:
        validate_task_for_capability_routing(task)
        requested = task.agent_id or ""
        capability = task.context.capability or ""
        selection: AgentRouteSelection

        if task.agent_id and self._registry.has(task.agent_id):
            if not self._registry.is_routable(
                task.agent_id,
                production_mode=self._production_mode,
            ):
                raise RuntimeError(
                    f"Requested agent is not routable: {task.agent_id}"
                )
            agent = self._registry.get(task.agent_id)
            selection = AgentRouteSelection(
                requested_agent_id=requested,
                selected_agent_id=agent.get_contract().id,
                capability=capability,
                selection_reason="explicit_agent_id",
            )
        elif task.context.capability:
            route = select_best_capability_match(
                self._registry,
                task,
                task.context.capability,
                production_mode=self._production_mode,
            )
            if route.selected is not None:
                agent = route.selected
                score: float | None = None
                match = agent.can_handle(task.context)
                if match.matched:
                    score = match.score
                selection = AgentRouteSelection(
                    requested_agent_id=requested,
                    selected_agent_id=agent.get_contract().id,
                    capability=capability,
                    match_score=score,
                    selection_reason=route.selection_reason,
                )
            else:
                agent, selection = self._route_best_or_fallback(task, requested, capability)
        else:
            agent, selection = self._route_best_or_fallback(task, requested, capability)

        self._emit_agent_selected(
            task,
            selection,
            run_id=run_id,
            node_id=node_id,
        )
        return agent

    def _route_best_or_fallback(
        self,
        task: Task,
        requested: str,
        capability: str,
    ) -> tuple[Agent, AgentRouteSelection]:
        match = self._registry.find_best_match(
            task.context,
            production_mode=self._production_mode,
        )
        if match is not None:
            return match, AgentRouteSelection(
                requested_agent_id=requested,
                selected_agent_id=match.get_contract().id,
                capability=capability,
                selection_reason="registry_best_match",
            )

        ids = self._registry.list_routable_agent_ids(production_mode=self._production_mode)
        if not ids:
            raise RuntimeError("AgentRegistry has no routable agents.")
        agent = self._registry.get(ids[0])
        return agent, AgentRouteSelection(
            requested_agent_id=requested,
            selected_agent_id=agent.get_contract().id,
            capability=capability,
            selection_reason="fallback_first_routable",
            fallback_used=True,
        )

    def _best_capability_match(
        self,
        context: TaskContext,
        candidates: list[Agent],
    ) -> tuple[Agent, float | None]:
        best: Optional[tuple[float, Agent]] = None
        for agent in candidates:
            result = agent.can_handle(context)
            if not result.matched:
                continue
            if best is None or result.score > best[0]:
                best = (result.score, agent)
        if best is not None:
            return best[1], best[0]
        return candidates[0], None

    def _emit_agent_selected(
        self,
        task: Task,
        selection: AgentRouteSelection,
        *,
        run_id: str,
        node_id: str | None,
    ) -> None:
        if self._event_bus is None:
            return
        active_run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        resolved_run_id = validate_run_id(run_id) if run_id else active_run_id
        if resolved_run_id != active_run_id:
            raise RuntimeError("run_id conflicts with active execution identity")
        base = RuntimeEvent(
            tenant_id=task.tenant_id,
            task_id=task.task_id,
            run_id=resolved_run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            agent_id=selection.selected_agent_id,
            node_id=node_id,
            event_type=RuntimeEventType.AGENT_SELECTED,
            phase=ExecutionPhase.AGENT_SELECTION,
            severity=EventSeverity.INFO,
            correlation_id=task.task_id,
        )
        typed = AgentSelectionPayloadV1(
            requested_agent_id=selection.requested_agent_id,
            selected_agent_id=selection.selected_agent_id,
            capability=selection.capability,
            match_score=selection.match_score,
            selection_reason=selection.selection_reason,
            fallback_used=selection.fallback_used,
        )
        event = runtime_event_with_payload(
            base,
            typed,
            promote_fields={
                "selected_agent_id": selection.selected_agent_id,
                "selection_reason": selection.selection_reason,
            },
        )
        self._event_bus.record(event, tenant_id=task.tenant_id)
