# © Artur Czarnecki. All rights reserved.

"""Prompt approval workflow beyond registry metadata (AUDIT-IDEAL-17.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(slots=True)
class PromptApprovalRecord:
    prompt_id: str
    version: int
    approver_id: str
    approved_at_utc: str
    change_ticket_ref: str = ""


@dataclass
class PromptApprovalQueue:
    """In-memory approval store for managed prompt versions."""

    _records: dict[tuple[str, int], PromptApprovalRecord] = field(default_factory=dict)

    def approve(
        self,
        *,
        prompt_id: str,
        version: int,
        approver_id: str,
        change_ticket_ref: str = "",
    ) -> PromptApprovalRecord:
        record = PromptApprovalRecord(
            prompt_id=prompt_id,
            version=version,
            approver_id=approver_id,
            approved_at_utc=datetime.now(timezone.utc).isoformat(),
            change_ticket_ref=change_ticket_ref,
        )
        self._records[(prompt_id, version)] = record
        return record

    def is_approved(self, prompt_id: str, version: int) -> bool:
        return (prompt_id, version) in self._records

    def get(self, prompt_id: str, version: int) -> PromptApprovalRecord | None:
        return self._records.get((prompt_id, version))
