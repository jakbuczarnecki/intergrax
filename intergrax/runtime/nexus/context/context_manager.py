# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.nexus.artifacts.models import ArtifactRef
from intergrax.runtime.nexus.context.shared_task_context import (
    SharedArtifactEntry,
    SharedContextConflictError,
    SharedTaskContext,
    get_or_create_shared_task_context,
    load_shared_task_context,
    save_shared_task_context,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.task.task import Task


class AgentContextBundle(BaseModel):
    """Bounded context passed to an agent for a graph node (§28, §42.14)."""

    message: str
    prior_outputs: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    shared_context: Optional[SharedTaskContext] = None


class ContextManager:
    """
    Builds per-node agent context from task state, shared context, and prior graph outputs.
    """

    def __init__(self, *, max_prior_chars: int = 4000) -> None:
        self._max_prior_chars = max_prior_chars

    def get_shared_context(self, task: Task) -> Optional[SharedTaskContext]:
        return load_shared_task_context(task)

    def ensure_shared_context(self, task: Task) -> SharedTaskContext:
        shared = get_or_create_shared_task_context(task, task_id=task.task_id)
        save_shared_task_context(task, shared)
        return shared

    def build_agent_context(
        self,
        task: Task,
        node: ExecutionNode,
        prior_outputs: Dict[str, AgentExecutionResult],
    ) -> AgentContextBundle:
        shared = self.ensure_shared_context(task)
        evidence: List[str] = []
        structured: Dict[str, Any] = dict(shared.structured_outputs)

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
            shared_context=shared,
            metadata={
                "node_id": node.node_id,
                "capability": node.capability,
                "depends_on": list(node.depends_on),
                "shared_context_version": shared.version,
            },
        )

    def apply_to_task(self, task: Task, bundle: AgentContextBundle) -> Task:
        """Return task copy with bounded message and shared context for agent execution."""
        shared = bundle.shared_context or self.ensure_shared_context(task)
        return task.model_copy(
            update={
                "message": bundle.message,
                "metadata": {
                    **task.metadata,
                    "agent_context": bundle.metadata,
                    "prior_agent_outputs": bundle.prior_outputs,
                    "shared_task_context": shared.model_dump(mode="json"),
                },
            }
        )

    def record_node_output(
        self,
        task: Task,
        node: ExecutionNode,
        execution: AgentExecutionResult,
    ) -> SharedTaskContext:
        """Merge a completed node result into the task-level shared context."""
        shared = self.ensure_shared_context(task)
        shared.structured_outputs[node.node_id] = {
            "agent_id": execution.agent_id,
            "node_id": node.node_id,
            "capability": node.capability,
            "summary": execution.summary,
            "structured_data": dict(execution.structured_data or {}),
            "status": execution.status.value,
        }
        shared.version += 1
        save_shared_task_context(task, shared)
        return shared

    def put_structured_output(
        self,
        task: Task,
        *,
        key: str,
        payload: Dict[str, Any],
        expected_version: Optional[int] = None,
    ) -> SharedTaskContext:
        """Explicit shared-context write (Tier-1 API for orchestrators / handoff)."""
        shared = self.ensure_shared_context(task)
        if expected_version is not None and shared.version != expected_version:
            raise SharedContextConflictError(
                f"shared context version mismatch: expected {expected_version}, got {shared.version}"
            )
        shared.structured_outputs[key.strip()] = dict(payload)
        shared.version += 1
        save_shared_task_context(task, shared)
        return shared

    def put_artifact(
        self,
        task: Task,
        *,
        label: str,
        artifact: ArtifactRef,
        expected_version: Optional[int] = None,
    ) -> SharedTaskContext:
        shared = self.ensure_shared_context(task)
        if expected_version is not None and shared.version != expected_version:
            raise SharedContextConflictError(
                f"shared context version mismatch: expected {expected_version}, got {shared.version}"
            )
        shared.artifacts[label.strip()] = SharedArtifactEntry.from_ref(artifact)
        shared.version += 1
        save_shared_task_context(task, shared)
        return shared
