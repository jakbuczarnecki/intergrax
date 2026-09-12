"""Production-like lab business references — maps from logical dataset IDs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabBusinessReferences:
    """Canonical enterprise-facing identifiers for ERL-QUAL-004 materialization."""

    logical_order_id: str = "erl-qual-004-ord-0001"
    order_number: str = "PO-2026-004872"
    organization_key: str = "ORG-NIC-PL-004872"
    organization_legal_name: str = "Nordic Industrial Components Sp. z o.o."
    account_reference: str = "ACCT-EU-B2B-88421"
    payment_intent_reference: str = "PAY-20260912-8F31A"
    payment_idempotency_key: str = "idem-capture-po-2026-004872-v1"
    order_amount: str = "12500.00"
    currency: str = "EUR"
    order_business_status: str = "AWAITING_PAYMENT_CONFIRMATION"
