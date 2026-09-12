# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing decision — proposal only, never execution (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4


def mint_self_healing_decision_id() -> str:
    return f"sh_dec_{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class SelfHealingProposedAction:
    """WHAT should happen — translated later by ``SelfHealingActionProvider``."""

    action_type: str
    target_resource: str
    operation_kind: str
    provider_id: str
    rationale: str

    def __post_init__(self) -> None:
        if not self.action_type.strip():
            raise ValueError("action_type required")
        if not self.target_resource.strip():
            raise ValueError("target_resource required")
        if not self.operation_kind.strip():
            raise ValueError("operation_kind required")
        if not self.provider_id.strip():
            raise ValueError("provider_id required")
        if not self.rationale.strip():
            raise ValueError("rationale required")


@dataclass(frozen=True, slots=True)
class SelfHealingDecision:
    """
    Strategy output — governance and safety decide whether actions may execute.

    Invariant: no ``execute()`` surface; confidence is not authorization.
    """

    decision_id: str
    strategy_id: str
    confidence: float
    proposed_actions: tuple[SelfHealingProposedAction, ...]
    evidence_refs: tuple[str, ...]
    justification: str
    required_approval: bool

    def __post_init__(self) -> None:
        if not self.decision_id.startswith("sh_dec_"):
            raise ValueError("decision_id must be sh_dec_*")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        if not self.justification.strip():
            raise ValueError("justification required")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must be non-empty")


__all__ = [
    "SelfHealingDecision",
    "SelfHealingProposedAction",
    "mint_self_healing_decision_id",
]
