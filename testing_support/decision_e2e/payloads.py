# © Artur Czarnecki. All rights reserved.

"""Shared qualification payload types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QualificationRecommendation:
    """Structured single-model / council output for DS-E2E."""

    recommendation: str
    confidence: str = "medium"
    rationale_summary: str = ""


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
