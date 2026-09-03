# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared wake-up receipt claim resolution for repository adapters (AW-4A)."""

from __future__ import annotations

from intergrax.autonomous_work.repository import (
    WorkerWakeUpReceiptClaim,
    WorkerWakeUpReceiptClaimStatus,
)
from intergrax.contracts.autonomous_work.wake_up import (
    WorkerWakeUpReceipt,
    wake_up_signals_logically_equivalent,
)


def resolve_wake_up_receipt_claim(
    incoming: WorkerWakeUpReceipt,
    stored: WorkerWakeUpReceipt | None,
) -> WorkerWakeUpReceiptClaim:
    """Resolve a claim against optional stored canonical receipt."""
    if stored is None:
        return WorkerWakeUpReceiptClaim(
            status=WorkerWakeUpReceiptClaimStatus.CLAIMED,
            receipt=incoming,
        )
    if wake_up_signals_logically_equivalent(incoming, stored):
        return WorkerWakeUpReceiptClaim(
            status=WorkerWakeUpReceiptClaimStatus.DUPLICATE,
            receipt=stored,
        )
    return WorkerWakeUpReceiptClaim(
        status=WorkerWakeUpReceiptClaimStatus.CONFLICT,
        receipt=stored,
    )
