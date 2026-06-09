# © Artur Czarnecki. All rights reserved.

"""Rules-backed task classifier — keyword intent routes with deterministic fallback (ORCH-CONFIG.1)."""

from __future__ import annotations

from intergrax.applications.contracts.intent_route import IntentRoute
from intergrax.runtime.nexus.intent_routing import apply_intent_routes
from intergrax.runtime.nexus.task_classifier import ClassifyingTaskClassifier
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task


class RulesTaskClassifier:
    """
    Applies ``IntentRoute`` rules, then delegates to ``ClassifyingTaskClassifier``.

    LLM classifier (COG-3.2) will extend this path; rules remain the fail-safe fallback.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        intent_routes: list[IntentRoute] | None = None,
    ) -> None:
        self._inner = ClassifyingTaskClassifier(registry)
        self._routes = list(intent_routes or [])

    def classify(self, task: Task) -> Task:
        routed = apply_intent_routes(task, self._routes)
        return self._inner.classify(routed)
