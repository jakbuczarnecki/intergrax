# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Delivery receipts and dead-letter ledger (Appendix B.13)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4


@dataclass(slots=True)
class DeliveryReceipt:
    delivery_id: str
    destination: str
    task_id: str
    channel: str
    status: str
    attempts: int
    delivered_at_utc: str
    last_error: Optional[str] = None
    payload_summary: Dict[str, str] = field(default_factory=dict)


class InMemoryDeliveryLedger:
    """Process-local ledger for tests and lab defaults."""

    def __init__(self) -> None:
        self._receipts: List[DeliveryReceipt] = []
        self._dead_letters: List[DeliveryReceipt] = []

    def record_success(
        self,
        *,
        destination: str,
        task_id: str,
        channel: str,
        attempts: int,
        payload_summary: Optional[Dict[str, str]] = None,
    ) -> DeliveryReceipt:
        receipt = DeliveryReceipt(
            delivery_id=f"dlv_{uuid4().hex}",
            destination=destination,
            task_id=task_id,
            channel=channel,
            status="delivered",
            attempts=attempts,
            delivered_at_utc=_utc_now(),
            payload_summary=dict(payload_summary or {}),
        )
        self._receipts.append(receipt)
        return receipt

    def record_dead_letter(
        self,
        *,
        destination: str,
        task_id: str,
        channel: str,
        attempts: int,
        last_error: str,
        payload_summary: Optional[Dict[str, str]] = None,
    ) -> DeliveryReceipt:
        receipt = DeliveryReceipt(
            delivery_id=f"dlv_{uuid4().hex}",
            destination=destination,
            task_id=task_id,
            channel=channel,
            status="dead_letter",
            attempts=attempts,
            delivered_at_utc=_utc_now(),
            last_error=last_error,
            payload_summary=dict(payload_summary or {}),
        )
        self._dead_letters.append(receipt)
        return receipt

    def list_receipts(self, *, limit: int = 100) -> List[DeliveryReceipt]:
        return self._receipts[-limit:]

    def list_dead_letters(self, *, limit: int = 100) -> List[DeliveryReceipt]:
        return self._dead_letters[-limit:]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
