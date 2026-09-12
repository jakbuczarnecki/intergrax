"""Inbound capture contract — commerce application toward external acquirer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class PaymentCaptureCommand:
    """Funds capture instruction crossing the enterprise integration boundary."""

    correlation_id: str
    external_business_reference: str
    idempotency_key: str
    merchant_order_reference: str
    amount: Decimal
    currency: str
    request_timestamp: datetime
