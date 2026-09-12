# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing audit trail (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.self_healing.result import SelfHealingExecutionOutcome


@dataclass(frozen=True, slots=True)
class SelfHealingAuditRecord:
    decision_id: str
    strategy_id: str
    tenant_id: str
    evidence_refs: tuple[str, ...]
    approval_refs: tuple[str, ...]
    external_operation_id: str | None
    execution_id: str | None
    outcome: SelfHealingExecutionOutcome
    recorded_at: datetime
    action_type: str = ""
    admission_reason: str = ""

    def __post_init__(self) -> None:
        if not self.decision_id.startswith("sh_dec_"):
            raise ValueError("decision_id must be sh_dec_*")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.evidence_refs:
            raise ValueError("evidence_refs required")


__all__ = ["SelfHealingAuditRecord"]
