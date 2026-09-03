# © Artur Czarnecki. All rights reserved.

"""Rules-backed task classifier — keyword intent routes with deterministic fallback (ORCH-CONFIG.1)."""

from __future__ import annotations

from intergrax.contracts.intent_route import IntentRoute
from intergrax.runtime.nexus.intent_routing import apply_intent_routes
from intergrax.runtime.nexus.task_classifier import ClassifyingTaskClassifier
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.task import Task


class RulesTaskClassifier:
    """
    Applies ``IntentRoute`` rules, then delegates to ``ClassifyingTaskClassifier``.

    LLM classifier (COG-3.2) will extend this path; rules remain the fail-safe fallback.
    """

    def __init__(
        self,
        registry: AgentRegistryRead,
        *,
        intent_routes: list[IntentRoute] | None = None,
        orchestration_trigger_capabilities: frozenset[str] | None = None,
        pipeline_capability_suffix: str = ".pipeline",
    ) -> None:
        self._inner = ClassifyingTaskClassifier(
            registry,
            orchestration_trigger_capabilities=orchestration_trigger_capabilities,
            pipeline_capability_suffix=pipeline_capability_suffix,
        )
        self._routes = list(intent_routes or [])

    def classify(self, task: Task) -> Task:
        before = (task.context.capability or "").strip()
        routed = apply_intent_routes(task, self._routes)
        after = (routed.context.capability or "").strip()
        if not before and after:
            cls = routed.runtime.classification
            cls.classifier_source = "rules"
            cls.confidence = 1.0
            cls.rationale = f"intent_route:{after}"
        classified = self._inner.classify(routed)
        classified.sync_metadata()
        return classified
