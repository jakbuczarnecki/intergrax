# © Artur Czarnecki. All rights reserved.

"""HITL ↔ PolicyDecision correlation metadata (IDEAL-5.3)."""

from __future__ import annotations

HITL_POLICY_DECISION_ID_KEY = "policy_decision_id"
HITL_INTERRUPT_ID_KEY = "interrupt_id"


def correlate_hitl_with_policy(
    metadata: dict[str, object],
    *,
    policy_decision_id: str,
    interrupt_id: str,
) -> None:
    metadata[HITL_POLICY_DECISION_ID_KEY] = policy_decision_id
    metadata[HITL_INTERRUPT_ID_KEY] = interrupt_id
