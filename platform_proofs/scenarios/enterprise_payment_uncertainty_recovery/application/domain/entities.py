"""Enterprise business entities for the payment-uncertainty lab workflow."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class EnterpriseOrganization:
    """Buyer organization for the scenario purchase order."""

    organization_key: str
    legal_name: str
    account_reference: str


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    """Order state visible to the scenario application at workflow entry."""

    logical_order_id: str
    order_number: str
    organization: EnterpriseOrganization
    amount: Decimal
    currency: str
    business_status: str


@dataclass(frozen=True, slots=True)
class PaymentCaptureRequest:
    """Payment capture intent issued by the application toward external integration."""

    intent_reference: str
    idempotency_key: str
    related_order_number: str
    amount: Decimal
    currency: str
