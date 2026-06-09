# © Artur Czarnecki. All rights reserved.

"""Impersonation rationale logging (IDEAL-4.6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class ImpersonationAuditRecord:
    actor_id: str
    impersonated_user_id: str
    rationale: str
    recorded_at: datetime


def record_impersonation(
    *,
    actor_id: str,
    impersonated_user_id: str,
    rationale: str,
) -> ImpersonationAuditRecord:
    if not rationale.strip():
        raise ValueError("impersonation rationale is required")
    return ImpersonationAuditRecord(
        actor_id=actor_id,
        impersonated_user_id=impersonated_user_id,
        rationale=rationale.strip(),
        recorded_at=datetime.now(timezone.utc),
    )
