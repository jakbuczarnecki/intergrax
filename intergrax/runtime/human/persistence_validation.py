# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Persistence acceptance boundary for human decision records (SX-F12)."""

from __future__ import annotations

from intergrax.runtime.human.models import HumanDecisionRecord
from intergrax.runtime.human.persistence_errors import HumanDecisionPersistenceValidationError

__all__ = ["validate_human_decision_for_persistence"]


def validate_human_decision_for_persistence(
    record: HumanDecisionRecord,
) -> HumanDecisionRecord:
    """
    Trust-boundary validation before any durable write.

    Complements model construction validators: Pydantic instances may be copied or
    updated without re-running field validators.
    """
    if not isinstance(record, HumanDecisionRecord):
        raise HumanDecisionPersistenceValidationError(
            "human decision record must be HumanDecisionRecord",
            decision_id=getattr(record, "decision_id", ""),
            tenant_id=getattr(record, "tenant_id", ""),
        )
    decision_id = str(record.decision_id or "").strip()
    if not decision_id:
        raise HumanDecisionPersistenceValidationError(
            "human decision decision_id must be nonblank",
            decision_id=record.decision_id,
            tenant_id=record.tenant_id,
        )
    tenant_id = str(record.tenant_id or "").strip()
    if not tenant_id:
        raise HumanDecisionPersistenceValidationError(
            "human decision tenant_id must be nonblank",
            decision_id=decision_id,
            tenant_id=record.tenant_id,
        )
    approver = record.approver
    if approver is None:
        raise HumanDecisionPersistenceValidationError(
            "human decision approver evidence required for persistence",
            decision_id=decision_id,
            tenant_id=tenant_id,
        )
    if approver.tenant_id != tenant_id:
        raise HumanDecisionPersistenceValidationError(
            "human decision approver tenant must match decision tenant_id",
            decision_id=decision_id,
            tenant_id=tenant_id,
        )
    return record
