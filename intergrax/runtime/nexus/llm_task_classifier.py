# © Artur Czarnecki. All rights reserved.

"""LLM-backed task classifier with deterministic rules fallback (COG-3.2 / ORCH-CONFIG.1)."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.intent_route import IntentRoute
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.rules_task_classifier import RulesTaskClassifier
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext


class _LlmClassificationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class LlmTaskClassifier:
    """
    Infer ``task.context.capability`` via LLM when unset, then classify.

    Falls back to :class:`RulesTaskClassifier` on parse failure or unknown capability.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        llm_adapter: LLMAdapter,
        *,
        intent_routes: list[IntentRoute] | None = None,
        orchestration_trigger_capabilities: frozenset[str] | None = None,
        pipeline_capability_suffix: str = ".pipeline",
    ) -> None:
        self._llm_adapter = llm_adapter
        self._routes = list(intent_routes or [])
        self._rules = RulesTaskClassifier(
            registry,
            intent_routes=intent_routes,
            orchestration_trigger_capabilities=orchestration_trigger_capabilities,
            pipeline_capability_suffix=pipeline_capability_suffix,
        )
        self._allowed_capabilities = tuple(
            dict.fromkeys(route.capability.strip() for route in self._routes if route.capability.strip())
        )

    def classify(self, task: Task) -> Task:
        existing = (task.context.capability or "").strip()
        if existing:
            return self._rules.classify(task)

        inferred = self._infer_capability(task)
        if inferred is not None:
            capability, confidence, rationale = inferred
            task.context = TaskContext(
                capability=capability,
                intent=task.context.intent,
                metadata=dict(task.context.metadata),
            )
            task.runtime.classification.classifier_source = "llm"
            task.runtime.classification.confidence = confidence
            task.runtime.classification.rationale = rationale

        classified = self._rules.classify(task)
        if not existing and classified.context.capability:
            cls = classified.runtime.classification
            if cls.classifier_source is None:
                cls.classifier_source = "rules"
            if cls.confidence is None:
                cls.confidence = 1.0
            if not cls.rationale:
                cls.rationale = f"intent_route:{classified.context.capability}"
            classified.sync_metadata()
        return classified

    def _infer_capability(self, task: Task) -> tuple[str, float, str] | None:
        if not self._allowed_capabilities:
            return None
        message = (task.message or "").strip()
        if not message:
            return None

        options = ", ".join(repr(cap) for cap in self._allowed_capabilities)
        prompt = (
            "You are a Nexus task router. Return JSON only with shape "
            '{"capability":"...","confidence":0.0-1.0,"rationale":"..."} '
            f"Choose exactly one capability from: {options}. "
            f"User message: {message!r}"
        )
        response = self._llm_adapter.generate_messages(
            [ChatMessage(role="user", content=prompt)],
            run_id=task.task_id,
        )
        payload = _extract_json_object(response.content.strip())
        if payload is None:
            return None
        try:
            parsed = _LlmClassificationPayload.model_validate(payload)
        except Exception:
            return None
        capability = parsed.capability.strip()
        if capability not in self._allowed_capabilities:
            return None
        return capability, parsed.confidence, parsed.rationale or "llm_classification"


def _extract_json_object(raw: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return None
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None
