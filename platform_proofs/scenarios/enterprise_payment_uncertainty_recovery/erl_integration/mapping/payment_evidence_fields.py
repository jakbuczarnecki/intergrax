"""Map variant dataset slices to payment reconciliation evidence records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_reconciliation_evidence import (
    PaymentReconciliationEvidence,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.reference.dataset_manifest import (
    InvalidDatasetError,
)

_RECON_AVAILABLE = "available"
_RECON_UNAVAILABLE = "unavailable"

_SETTLEMENT_SETTLED = "SETTLED"
_SETTLEMENT_NOT_SETTLED = "NOT_SETTLED"
_SETTLEMENT_UNKNOWN = "UNKNOWN"

_TIER_AUTHORITATIVE = "AUTHORITATIVE"
_TIER_UNAVAILABLE = "UNAVAILABLE"

_TRUTH_AVAILABLE = "AVAILABLE"


def resolve_payment_reconciliation_evidence(
    variant_document: dict[str, Any],
    *,
    correlation_id: str,
    external_effect_reference: str,
    sor_transaction_ref: str | None,
    funds_captured: bool,
    truth_availability_state: str,
    observed_at: datetime | None = None,
) -> PaymentReconciliationEvidence:
    """Build payment evidence from variant reconciliation metadata and SoR fields."""
    reconciliation = variant_document.get("reconciliation")
    if not isinstance(reconciliation, dict):
        raise InvalidDatasetError("reconciliation must be an object")

    availability = reconciliation.get("availability")
    if not isinstance(availability, str):
        raise InvalidDatasetError("reconciliation.availability must be a string")

    discoverable = reconciliation.get("discoverable_outcome")
    discoverable_outcome = discoverable if isinstance(discoverable, str) else None

    if availability == _RECON_UNAVAILABLE or truth_availability_state != _TRUTH_AVAILABLE:
        return PaymentReconciliationEvidence(
            correlation_id=correlation_id,
            external_effect_reference=external_effect_reference,
            psp_confirmation_id=None,
            sor_transaction_ref=sor_transaction_ref,
            reconciliation_availability=availability,
            discoverable_outcome=discoverable_outcome,
            funds_captured=funds_captured,
            settlement_status=_SETTLEMENT_UNKNOWN,
            settlement_batch_ref=None,
            source_reliability_tier=_TIER_UNAVAILABLE,
            evidence_observed_at=observed_at,
        )

    if availability != _RECON_AVAILABLE:
        raise InvalidDatasetError(f"unsupported reconciliation.availability: {availability!r}")

    anchor = sor_transaction_ref or external_effect_reference
    settlement_status = _SETTLEMENT_SETTLED if funds_captured else _SETTLEMENT_NOT_SETTLED
    settlement_batch_ref = f"stl-batch-{anchor}" if funds_captured else None

    return PaymentReconciliationEvidence(
        correlation_id=correlation_id,
        external_effect_reference=external_effect_reference,
        psp_confirmation_id=f"psp-conf-{anchor}",
        sor_transaction_ref=sor_transaction_ref,
        reconciliation_availability=availability,
        discoverable_outcome=discoverable_outcome,
        funds_captured=funds_captured,
        settlement_status=settlement_status,
        settlement_batch_ref=settlement_batch_ref,
        source_reliability_tier=_TIER_AUTHORITATIVE,
        evidence_observed_at=observed_at,
    )
