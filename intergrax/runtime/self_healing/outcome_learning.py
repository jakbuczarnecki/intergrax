# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing outcome learning loop (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.self_healing.audit import SelfHealingAuditRecord
from intergrax.contracts.self_healing.quality import SelfHealingStrategyQualityProfile
from intergrax.contracts.self_healing.result import SelfHealingExecutionOutcome


@dataclass
class InMemorySelfHealingStrategyQualityStore:
    _profiles: dict[tuple[str, str], SelfHealingStrategyQualityProfile] = field(
        default_factory=dict,
    )

    def get(self, *, strategy_id: str, tenant_id: str) -> SelfHealingStrategyQualityProfile | None:
        return self._profiles.get((strategy_id, tenant_id))

    def apply_audit_record(self, record: SelfHealingAuditRecord) -> SelfHealingStrategyQualityProfile:
        key = (record.strategy_id, record.tenant_id)
        current = self._profiles.get(
            key,
            SelfHealingStrategyQualityProfile(
                strategy_id=record.strategy_id,
                tenant_id=record.tenant_id,
            ),
        )
        successful = current.successful_preventions
        failed = current.failed_actions
        if record.outcome is SelfHealingExecutionOutcome.SUCCEEDED:
            successful += 1
        elif record.outcome in {
            SelfHealingExecutionOutcome.FAILED,
            SelfHealingExecutionOutcome.DENIED,
            SelfHealingExecutionOutcome.REJECTED,
        }:
            failed += 1
        updated = SelfHealingStrategyQualityProfile(
            strategy_id=record.strategy_id,
            tenant_id=record.tenant_id,
            successful_preventions=successful,
            failed_actions=failed,
            rollback_rate=current.rollback_rate,
            false_positive_rate=current.false_positive_rate,
        )
        self._profiles[key] = updated
        return updated


class SelfHealingOutcomeEngine:
    def __init__(self, store: InMemorySelfHealingStrategyQualityStore) -> None:
        self._store = store

    def record(self, audit: SelfHealingAuditRecord) -> SelfHealingStrategyQualityProfile:
        return self._store.apply_audit_record(audit)


__all__ = [
    "InMemorySelfHealingStrategyQualityStore",
    "SelfHealingOutcomeEngine",
]
