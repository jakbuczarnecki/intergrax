# © Artur Czarnecki. All rights reserved.

"""Shared qualification payload types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class QualificationRecommendation:
    """Structured single-model / council output for DS-E2E.

    Plain class (not Pydantic) so ``type(cls) is type`` holds for deliberation
    contracts while still supporting Ollama structured generation.
    """

    __slots__ = ("recommendation", "confidence", "rationale_summary")

    def __init__(
        self,
        recommendation: str,
        *,
        confidence: str = "medium",
        rationale_summary: str = "",
    ) -> None:
        self.recommendation = recommendation
        self.confidence = confidence
        self.rationale_summary = rationale_summary

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "recommendation": {"type": "string"},
                "confidence": {"type": "string"},
                "rationale_summary": {"type": "string"},
            },
            "required": ["recommendation"],
            "additionalProperties": False,
        }

    @classmethod
    def model_validate(cls, data: object) -> QualificationRecommendation:
        if not isinstance(data, Mapping):
            raise TypeError("QualificationRecommendation.model_validate expects mapping")
        recommendation = data.get("recommendation")
        if not isinstance(recommendation, str) or not recommendation:
            raise ValueError("recommendation must be a non-empty string")
        confidence = data.get("confidence", "medium")
        rationale_summary = data.get("rationale_summary", "")
        if not isinstance(confidence, str):
            raise TypeError("confidence must be str")
        if not isinstance(rationale_summary, str):
            raise TypeError("rationale_summary must be str")
        return cls(
            recommendation=recommendation,
            confidence=confidence,
            rationale_summary=rationale_summary,
        )


@dataclass(frozen=True, slots=True)
class QualificationSemanticContent:
    """Semantic verification payload."""

    text: str


@dataclass(frozen=True, slots=True)
class SandboxSideEffectRecord:
    """Durable sandbox side-effect marker for governance proofs."""

    tenant_id: str
    decision_id: str
    decision_version: str
    action_kind: str
    executed: bool
