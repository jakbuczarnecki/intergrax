# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""EE-B1.1 — provider-neutral execution failure classification contract."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_retry import ExecutionFailureClassification
from intergrax.contracts.resilience_policy import FailureClass

_MAX_REASON = 512


class ExecutionFailureSemanticCategory(StrEnum):
    """Enterprise reliability taxonomy (EE-B1.1) — orthogonal to retry projection."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    POLICY_BLOCKED = "policy_blocked"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    DEPENDENCY_FAILURE = "dependency_failure"
    UNKNOWN = "unknown"


class ExecutionFailureContext(BaseModel):
    """Normalized failure inputs for classification — no provider or LLM details."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: str = Field(default="", max_length=_MAX_REASON)
    failure_class: FailureClass | None = None
    timeout: bool = False
    resource_exhausted: bool = False
    policy_blocked: bool = False
    dependency_unavailable: bool = False
    has_unknown_side_effect: bool = False


class ExecutionFailureDecision(BaseModel):
    """Typed classification outcome with optional retry-plane projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: ExecutionFailureSemanticCategory
    reason: str = Field(default="", max_length=_MAX_REASON)
    retry_projection: ExecutionFailureClassification | None = None


class ExecutionFailureClassifier(Protocol):
    """Pluggable, testable failure classifier (provider-neutral)."""

    def classify(
        self, failure_context: ExecutionFailureContext
    ) -> ExecutionFailureDecision:
        """Map a normalized failure context to a semantic category and retry projection."""
        ...


__all__ = [
    "ExecutionFailureClassifier",
    "ExecutionFailureContext",
    "ExecutionFailureDecision",
    "ExecutionFailureSemanticCategory",
]
