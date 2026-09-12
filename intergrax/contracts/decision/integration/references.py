# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration-boundary reference contracts (not L1–L18 matrix types)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage

REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE = "reference.decision.lifecycle_record.v1"


class ReferenceEnterpriseLifecycleState(StrEnum):
    """Reference-architecture lifecycle vocabulary exposed at the integration boundary."""

    CREATED = "created"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


def _require_non_empty(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    return value


@dataclass(frozen=True, slots=True)
class ReferenceDecisionLifecycleReference:
    """Input contract for reference decision lifecycle integration."""

    source_type: str
    decision_id: str
    lifecycle_state: ReferenceEnterpriseLifecycleState | None
    decision_type: str
    created_at_iso: str | None
    mapping_version: str = "1"

    def __post_init__(self) -> None:
        _require_non_empty(self.source_type, "source_type")
        _require_non_empty(self.decision_id, "decision_id")
        _require_non_empty(self.decision_type, "decision_type")
        _require_non_empty(self.mapping_version, "mapping_version")
        if (
            self.lifecycle_state is not None
            and type(self.lifecycle_state) is not ReferenceEnterpriseLifecycleState
        ):
            raise TypeError(
                "lifecycle_state must be ReferenceEnterpriseLifecycleState or None",
            )


@dataclass(frozen=True, slots=True)
class PlatformDecisionLifecycleReference:
    """Output contract pointing at platform decision lifecycle semantics."""

    reference_decision_id: str
    stage: DecisionLifecycleStage
    transition_index: int
    mapping_version: str

    def __post_init__(self) -> None:
        _require_non_empty(self.reference_decision_id, "reference_decision_id")
        _require_non_empty(self.mapping_version, "mapping_version")
        if type(self.stage) is not DecisionLifecycleStage:
            raise TypeError("stage must be DecisionLifecycleStage")
        if type(self.transition_index) is not int or isinstance(
            self.transition_index, bool
        ):
            raise TypeError("transition_index must be int")
        if self.transition_index < 0:
            raise ValueError("transition_index must be >= 0")


__all__ = [
    "REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE",
    "PlatformDecisionLifecycleReference",
    "ReferenceDecisionLifecycleReference",
    "ReferenceEnterpriseLifecycleState",
]
