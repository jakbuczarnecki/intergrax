# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Shared helpers for platform default self-healing strategies."""

from __future__ import annotations

from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.decision import (
    SelfHealingDecision,
    SelfHealingProposedAction,
    mint_self_healing_decision_id,
)


def _first_matching_operation(context: SelfHealingContext, operation_kind: str):
    for operation in context.available_operations:
        if operation.operation_kind == operation_kind:
            return operation
    return None


def mint_default_decision(
    *,
    strategy_id: str,
    context: SelfHealingContext,
    action_type: str,
    operation_kind: str,
    confidence: float,
    justification: str,
    required_approval: bool,
) -> SelfHealingDecision | None:
    operation = _first_matching_operation(context, operation_kind)
    if operation is None:
        return None
    evidence = context.diagnostic_investigation.evidence_refs
    if not evidence:
        return None
    action = SelfHealingProposedAction(
        action_type=action_type,
        target_resource=operation.target_resource,
        operation_kind=operation.operation_kind,
        provider_id=operation.provider_id,
        rationale=justification,
    )
    return SelfHealingDecision(
        decision_id=mint_self_healing_decision_id(),
        strategy_id=strategy_id,
        confidence=confidence,
        proposed_actions=(action,),
        evidence_refs=evidence,
        justification=justification,
        required_approval=required_approval,
    )


__all__ = ["mint_default_decision"]
