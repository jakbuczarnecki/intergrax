# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy resolution pipeline — enterprise ordering (SELF-HEALING R1)."""

from __future__ import annotations

from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.strategy import SelfHealingStrategy


def _capability_match(strategy: SelfHealingStrategy, context: SelfHealingContext) -> bool:
    required = set(strategy.descriptor.capabilities)
    if not required:
        return False
    tags = set(context.diagnostic_investigation.capability_tags)
    signal_types = {signal.signal_type for signal in context.predictive_signals}
    return bool(required & (tags | signal_types))


def _evidence_compatible(strategy: SelfHealingStrategy, context: SelfHealingContext) -> bool:
    if not context.diagnostic_investigation.evidence_refs:
        return False
    return True


def _tenant_allowed(strategy: SelfHealingStrategy, tenant_id: str) -> bool:
    scope = strategy.descriptor.tenant_scope
    if scope is None:
        return True
    return tenant_id in scope


def _resolution_sort_key(strategy: SelfHealingStrategy) -> tuple[int, int, int, str]:
    descriptor = strategy.descriptor
    tenant_specific = 1 if descriptor.tenant_scope is not None else 0
    return (
        tenant_specific,
        descriptor.specificity,
        descriptor.priority,
        descriptor.strategy_id,
    )


def resolve_strategies_for_context(
    strategies: tuple[SelfHealingStrategy, ...],
    context: SelfHealingContext,
) -> tuple[SelfHealingStrategy, ...]:
    """
    Resolution pipeline:

    tenant scope → capability match → evidence compatibility → priority/specificity.
    """
    candidates: list[SelfHealingStrategy] = []
    for strategy in strategies:
        if not _tenant_allowed(strategy, context.tenant_id):
            continue
        if not _capability_match(strategy, context):
            continue
        if not _evidence_compatible(strategy, context):
            continue
        candidates.append(strategy)
    return tuple(sorted(candidates, key=_resolution_sort_key, reverse=True))


__all__ = ["resolve_strategies_for_context"]
