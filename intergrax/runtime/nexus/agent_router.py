# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import (
    require_active_execution_id,
    require_active_execution_identity,
    validate_run_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.tier2_agent import Tier2Agent
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads import AgentSelectionPayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.registry.capability_routing import (
    select_best_capability_match,
    validate_task_for_capability_routing,
)
from intergrax.runtime.task.agent_capability_intake import task_envelope_for_agent_capability_match
from intergrax.runtime.task.task import Task


@dataclass(frozen=True)
class AgentRouteSelection:
    requested_agent_id: str
    selected_agent_id: str
    capability: str = ""
    match_score: float | None = None
    selection_reason: str = ""
    fallback_used: bool = False


class AgentRoutingError(RuntimeError):
    """No agent may be selected for the task under the router's posture."""


class AgentRouter:
    """
    Select agents from registry using explicit agent_id or capability matching (§10.4, §16).

    ``production_mode`` has no default: every construction states its posture. The
    first-routable fallback is default-deny; only ``lab_first_routable_fallback=True`` (LAB,
    never together with ``production_mode=True``) keeps it, and then the selection event
    carries ``fallback_used=True``.
    """

    def __init__(
        self,
        registry: AgentRegistryRead,
        *,
        production_mode: bool,
        event_bus: RuntimeEventBus | None = None,
        lab_first_routable_fallback: bool = False,
    ) -> None:
        if production_mode and lab_first_routable_fallback:
            raise ValueError(
                "lab_first_routable_fallback is a LAB opt-in and cannot be combined with production_mode=True",
            )
        self._registry = registry
        self._production_mode = production_mode
        self._event_bus = event_bus
        self._lab_first_routable_fallback = lab_first_routable_fallback

    @property
    def production_mode(self) -> bool:
        return self._production_mode

    def route(
        self,
        task: Task,
        *,
        run_id: str | None = None,
        node_id: str | None = None,
    ) -> Tier2Agent:
        validate_task_for_capability_routing(task)
        requested = task.agent_id or ""
        capability = task.context.capability or ""
        selection: AgentRouteSelection

        if task.agent_id and self._registry.has(task.agent_id):
            if not self._registry.is_routable(
                task.agent_id,
                production_mode=self._production_mode,
            ):
                raise RuntimeError(f"Requested agent is not routable: {task.agent_id}")
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
                if route.selected_match is not None and route.selected_match.matched:
                    score = route.selected_match.score
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
    ) -> tuple[Tier2Agent, AgentRouteSelection]:
        match = self._registry.find_best_match(
            task_envelope_for_agent_capability_match(task),
            production_mode=self._production_mode,
        )
        if match is not None:
            return match, AgentRouteSelection(
                requested_agent_id=requested,
                selected_agent_id=match.get_contract().id,
                capability=capability,
                selection_reason="registry_best_match",
            )

        if not self._lab_first_routable_fallback:
            raise AgentRoutingError(
                "No agent matches the task and the first-routable fallback is not enabled for this router.",
            )
        ids = self._registry.list_routable_agent_ids(production_mode=self._production_mode)
        if not ids:
            raise AgentRoutingError("AgentRegistry has no routable agents.")
        agent = self._registry.get(ids[0])
        return agent, AgentRouteSelection(
            requested_agent_id=requested,
            selected_agent_id=agent.get_contract().id,
            capability=capability,
            selection_reason="fallback_first_routable",
            fallback_used=True,
        )

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
