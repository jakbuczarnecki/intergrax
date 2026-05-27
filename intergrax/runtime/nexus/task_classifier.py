# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task


class TaskClassification(str, Enum):
    """Nexus task classification labels (§10.2)."""

    SINGLE_AGENT_DEFAULT = "single_agent_default"
    SINGLE_AGENT_EXPLICIT = "single_agent_explicit"
    CAPABILITY_ROUTED = "capability_routed"
    MULTI_AGENT = "multi_agent"
    UNSUPPORTED = "unsupported"
    HIGH_RISK = "high_risk"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"


class TaskClassifier:
    """
    Task classifier v2 (Phase B.1).

    Enriches task metadata; does not mutate Task.state (TaskLifecycle owns that).
    """

    def classify(self, task: Task) -> Task:
        capability = task.context.capability
        if capability:
            task.metadata["requested_capability"] = capability

        if task.metadata.get("require_human_approval"):
            task.metadata["classification"] = TaskClassification.HUMAN_APPROVAL_REQUIRED.value
            return task

        if capability and not self._has_capability_support(task, capability):
            task.metadata["classification"] = TaskClassification.UNSUPPORTED.value
            task.metadata["unsupported_reason"] = f"no agent for capability {capability!r}"
            return task

        if capability and self._is_multi_agent_capability(task, capability):
            task.metadata["classification"] = TaskClassification.MULTI_AGENT.value
        elif task.agent_id:
            task.metadata["classification"] = TaskClassification.SINGLE_AGENT_EXPLICIT.value
        elif capability:
            task.metadata["classification"] = TaskClassification.CAPABILITY_ROUTED.value
        else:
            task.metadata["classification"] = TaskClassification.SINGLE_AGENT_DEFAULT.value

        if self._is_high_risk(task):
            task.metadata["risk_level"] = AgentRiskLevel.HIGH.value
            if task.metadata["classification"] != TaskClassification.UNSUPPORTED.value:
                task.metadata["classification"] = TaskClassification.HIGH_RISK.value

        return task

    def _has_capability_support(self, task: Task, capability: str) -> bool:
        registry: Optional[AgentRegistry] = task.metadata.get("_registry")  # type: ignore[assignment]
        if registry is None:
            return True
        return len(registry.find_by_capability(capability)) > 0

    def _is_multi_agent_capability(self, task: Task, capability: str) -> bool:
        registry: Optional[AgentRegistry] = task.metadata.get("_registry")  # type: ignore[assignment]
        if registry is None:
            return False
        return len(registry.find_by_capability(capability)) > 1

    def _is_high_risk(self, task: Task) -> bool:
        registry: Optional[AgentRegistry] = task.metadata.get("_registry")  # type: ignore[assignment]
        if registry is None:
            return bool(task.metadata.get("high_risk"))
        agent_id = task.agent_id
        if agent_id and registry.has(agent_id):
            return registry.get_contract(agent_id).risk_level == AgentRiskLevel.HIGH
        capability = task.context.capability
        if capability:
            agents = registry.find_by_capability(capability)
            return any(a.get_contract().risk_level == AgentRiskLevel.HIGH for a in agents)
        return bool(task.metadata.get("high_risk"))


class ClassifyingTaskClassifier(TaskClassifier):
    """TaskClassifier with registry injected for capability checks."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def classify(self, task: Task) -> Task:
        task.metadata["_registry"] = self._registry
        result = super().classify(task)
        result.metadata.pop("_registry", None)
        return result
