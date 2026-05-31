# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Delivery ledger contract (Appendix B.13)."""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol

from intergrax.runtime.notifications.deliveries.delivery_ledger import DeliveryReceipt


class DeliveryLedger(Protocol):
    def record_success(
        self,
        *,
        destination: str,
        task_id: str,
        channel: str,
        attempts: int,
        payload_summary: Optional[Dict[str, str]] = None,
    ) -> DeliveryReceipt: ...

    def record_dead_letter(
        self,
        *,
        destination: str,
        task_id: str,
        channel: str,
        attempts: int,
        last_error: str,
        payload_summary: Optional[Dict[str, str]] = None,
    ) -> DeliveryReceipt: ...

    def list_receipts(self, *, limit: int = 100) -> List[DeliveryReceipt]: ...

    def list_dead_letters(self, *, limit: int = 100) -> List[DeliveryReceipt]: ...
