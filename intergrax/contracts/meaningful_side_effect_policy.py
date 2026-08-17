# © Artur Czarnecki. All rights reserved.

"""Typed runtime policy rules for meaningful side-effect evaluation (GEC-5 / G1B-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.runtime_policy import PolicyAction


def _normalize_required_id(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _normalize_optional_action(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError("action must be non-empty when provided")
    return normalized


@dataclass(frozen=True, slots=True)
class MeaningfulSideEffectPolicyRule:
    """Immutable runtime rule for ``evaluate_meaningful_side_effect``."""

    rule_id: str
    decision: PolicyAction
    action: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rule_id",
            _normalize_required_id(self.rule_id, field_name="rule_id"),
        )
        object.__setattr__(self, "action", _normalize_optional_action(self.action))
        if not isinstance(self.decision, PolicyAction):
            raise TypeError("decision must be PolicyAction")
