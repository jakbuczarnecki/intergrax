# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus handoff validation and graph mutation (§42.15, Phase I.4)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from intergrax.contracts.agent_handoff import AgentHandoff
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.task import Task, TaskContext


class HandoffValidationError(ValueError):
    """Raised when a handoff cannot be validated or routed."""


class HandoffValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    resolved_agent_id: Optional[str] = None


class HandoffCoordinator:
    """
    Validates handoff requests and inserts handoff nodes into the execution graph.

    All cross-agent transfers go through this coordinator — agents never call each other.
    """

    def __init__(self, registry: AgentRegistryRead) -> None:
        self._registry = registry

    def validate(
        self,
        handoff: AgentHandoff,
        *,
        from_agent_id: str,
        task: Optional[Task] = None,
    ) -> HandoffValidationResult:
        errors: list[str] = []
        if handoff.from_agent_id != from_agent_id:
            errors.append(
                f"handoff.from_agent_id mismatch: expected {from_agent_id}, got {handoff.from_agent_id}"
            )

        resolved_agent_id: Optional[str] = None
        try:
            resolved_agent_id = self.resolve_target_agent_id(handoff)
        except HandoffValidationError as exc:
            errors.append(str(exc))

        if task is not None and handoff.artifacts:
            shared = task.metadata.get("shared_task_context") or {}
            artifacts = shared.get("artifacts") if isinstance(shared, dict) else {}
            for label in handoff.artifacts:
                if not isinstance(artifacts, dict) or label not in artifacts:
                    errors.append(f"missing shared artifact: {label}")

        return HandoffValidationResult(
            valid=not errors,
            errors=errors,
            resolved_agent_id=resolved_agent_id,
        )

    def resolve_target_agent_id(self, handoff: AgentHandoff) -> str:
        if handoff.to_agent_id:
            if not self._registry.has(handoff.to_agent_id):
                raise HandoffValidationError(f"unknown target agent: {handoff.to_agent_id}")
            return handoff.to_agent_id

        if handoff.to_capability:
            matches = self._registry.find_by_capability(handoff.to_capability)
            if not matches:
                raise HandoffValidationError(
                    f"no agent registered for capability: {handoff.to_capability}"
                )
            context = TaskContext(capability=handoff.to_capability)
            best = self._best_capability_match(context, matches)
            return best.get_contract().id

        raise HandoffValidationError("handoff requires to_agent_id or to_capability")

    def apply_to_graph(
        self,
        graph: ExecutionGraph,
        handoff: AgentHandoff,
        *,
        from_node_id: str,
        resolved_agent_id: str,
    ) -> ExecutionNode:
        node_id = f"handoff_{handoff.handoff_id}"
        if any(node.node_id == node_id for node in graph.nodes):
            raise HandoffValidationError(f"handoff node already exists: {node_id}")

        node = ExecutionNode(
            node_id=node_id,
            agent_id=resolved_agent_id,
            capability=handoff.to_capability,
            description=handoff.reason or f"handoff from {handoff.from_agent_id}",
            depends_on=[from_node_id],
            metadata={
                "handoff_id": handoff.handoff_id,
                "handoff_from_agent": handoff.from_agent_id,
                "handoff_payload": dict(handoff.payload),
                "required_validation": list(handoff.required_validation),
            },
        )
        graph.nodes.append(node)
        return node

    @staticmethod
    def _best_capability_match(context: TaskContext, candidates: list) -> object:
        best = None
        best_score = -1.0
        for agent in candidates:
            result = agent.can_handle(context)
            if not result.matched:
                continue
            if result.score > best_score:
                best_score = result.score
                best = agent
        return best or candidates[0]
