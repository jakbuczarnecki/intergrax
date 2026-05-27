# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.task.task import Task


class AgentContextBundle(BaseModel):
    """Bounded context passed to an agent for a graph node (§28)."""

    message: str
    prior_outputs: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextManager:
    """
    Builds per-node agent context from task state and prior graph outputs (Phase C.4).
    """

    def __init__(self, *, max_prior_chars: int = 4000) -> None:
        self._max_prior_chars = max_prior_chars

    def build_agent_context(
        self,
        task: Task,
        node: ExecutionNode,
        prior_outputs: Dict[str, AgentExecutionResult],
    ) -> AgentContextBundle:
        evidence: List[str] = []
        structured: Dict[str, Any] = {}

        for dep_id in node.depends_on:
            prior = prior_outputs.get(dep_id)
            if prior is None:
                continue
            if prior.summary:
                evidence.append(prior.summary)
            structured[dep_id] = {
                "agent_id": prior.agent_id,
                "summary": prior.summary,
                "structured_data": prior.structured_data,
            }

        prior_text = "\n\n".join(evidence)
        if len(prior_text) > self._max_prior_chars:
            prior_text = prior_text[: self._max_prior_chars] + "\n...[truncated]"

        message = task.message or ""
        if prior_text:
            message = (
                f"{message}\n\n--- prior agent outputs ---\n{prior_text}"
                if message.strip()
                else prior_text
            )

        return AgentContextBundle(
            message=message,
            prior_outputs=structured,
            evidence=evidence,
            metadata={
                "node_id": node.node_id,
                "capability": node.capability,
                "depends_on": list(node.depends_on),
            },
        )

    def apply_to_task(self, task: Task, bundle: AgentContextBundle) -> Task:
        """Return task copy with bounded message for agent execution."""
        return task.model_copy(
            update={
                "message": bundle.message,
                "metadata": {
                    **task.metadata,
                    "agent_context": bundle.metadata,
                    "prior_agent_outputs": bundle.prior_outputs,
                },
            }
        )
