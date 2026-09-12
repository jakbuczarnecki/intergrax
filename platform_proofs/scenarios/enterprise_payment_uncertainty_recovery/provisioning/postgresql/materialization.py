"""Data-driven mapping from logical dataset to PostgreSQL row payloads."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.dataset_loader import (
    LoadedScenarioPackage,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.sor_truth import (
    SorTruthFields,
    resolve_sor_truth_fields,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.reference.dataset_manifest import (
    InvalidDatasetError,
)

_INTEGRATION_OUTCOME_MAP: dict[str, str] = {
    "unknown": "INDETERMINATE",
}

_ORDER_LIFECYCLE_STATUS_MAP: dict[str, str] = {
    "fulfillment_hold_until_payment_truth_established": "AWAITING_PAYMENT_CONFIRMATION",
}

_RECONCILIATION_AVAILABILITY_TO_CASE: dict[str, str] = {
    "available": "OPEN",
    "unavailable": "OPEN",
}


def logical_id_to_uuid(logical_id: str) -> UUID:
    return UUID(bytes=hashlib.sha256(f"erl-qual-004:{logical_id}".encode()).digest()[:16])


def deterministic_timestamp(logical_id: str, *, offset_seconds: int = 0) -> datetime:
    digest = hashlib.sha256(logical_id.encode("utf-8")).hexdigest()
    seconds = int(digest[:6], 16) % 7200
    base = datetime(2026, 9, 12, 8, 0, 0, tzinfo=UTC)
    return base + timedelta(seconds=seconds + offset_seconds)


def _slug_reference(logical_id: str, prefix: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "-", logical_id).strip("-").upper()
    return f"{prefix}-{token[-24:]}" if len(token) > 24 else f"{prefix}-{token}"


def resolve_external_reality_fields(variant_document: dict[str, Any]) -> dict[str, Any]:
    fields: SorTruthFields = resolve_sor_truth_fields(variant_document)
    return {
        "terminal_outcome": fields.terminal_outcome,
        "funds_captured": fields.funds_captured,
        "truth_availability_state": fields.truth_availability_state,
    }


@dataclass(frozen=True, slots=True)
class MaterializedScenarioState:
    organization_id: UUID
    order_id: UUID
    payment_intent_id: UUID
    application_knowledge_id: UUID
    external_payment_effect_id: UUID
    external_reality_id: UUID
    reconciliation_case_id: UUID
    variant_id: str
    logical_fingerprint: str

    def execution_handles(self) -> tuple[str, ...]:
        return (
            f"organization_id:{self.organization_id}",
            f"order_id:{self.order_id}",
            f"payment_intent_id:{self.payment_intent_id}",
            f"application_knowledge_id:{self.application_knowledge_id}",
            f"external_payment_effect_id:{self.external_payment_effect_id}",
            f"external_reality_id:{self.external_reality_id}",
            f"reconciliation_case_id:{self.reconciliation_case_id}",
            f"variant_id:{self.variant_id}",
            f"logical_fingerprint:{self.logical_fingerprint}",
        )


def build_materialized_state(package: LoadedScenarioPackage) -> MaterializedScenarioState:
    order = package.shared["order"]
    effect = package.shared["external_effect"]
    knowledge = package.shared["application_knowledge_at_entry"]
    inventory = package.shared["inventory_context"]
    variant_id = package.resolution.variant.variant_id

    order_logical = str(order["order_id"])
    effect_logical = str(effect["effect_id"])

    organization_id = logical_id_to_uuid(f"org:{order['customer_reference']}")
    order_id = logical_id_to_uuid(order_logical)
    payment_intent_id = logical_id_to_uuid(f"intent:{effect_logical}")
    application_knowledge_id = logical_id_to_uuid(str(knowledge["snapshot_id"]))
    external_payment_effect_id = logical_id_to_uuid(effect_logical)
    external_reality_id = logical_id_to_uuid(f"reality:{effect_logical}:{variant_id}")
    reconciliation_case_id = logical_id_to_uuid(f"case:{order_logical}:{variant_id}")

    _ = inventory  # reservation knowledge carried on application_knowledge row
    _ = resolve_external_reality_fields(package.variant_document)

    return MaterializedScenarioState(
        organization_id=organization_id,
        order_id=order_id,
        payment_intent_id=payment_intent_id,
        application_knowledge_id=application_knowledge_id,
        external_payment_effect_id=external_payment_effect_id,
        external_reality_id=external_reality_id,
        reconciliation_case_id=reconciliation_case_id,
        variant_id=variant_id,
        logical_fingerprint=package.resolution.variant.logical_fingerprint,
    )


def row_payloads(package: LoadedScenarioPackage) -> dict[str, dict[str, Any]]:
    """Return insert payloads keyed by table name (schema-qualified)."""
    order = package.shared["order"]
    effect = package.shared["external_effect"]
    knowledge = package.shared["application_knowledge_at_entry"]
    inventory = package.shared["inventory_context"]
    variant_document = package.variant_document
    variant_id = package.resolution.variant.variant_id

    order_logical = str(order["order_id"])
    effect_logical = str(effect["effect_id"])
    state = build_materialized_state(package)
    reality_fields = resolve_external_reality_fields(variant_document)
    reconciliation = variant_document["reconciliation"]
    if not isinstance(reconciliation, dict):
        raise InvalidDatasetError("reconciliation must be an object")
    recon_availability = reconciliation.get("availability")
    if not isinstance(recon_availability, str):
        raise InvalidDatasetError("reconciliation.availability must be a string")
    case_resolution_state = _RECONCILIATION_AVAILABILITY_TO_CASE.get(recon_availability)
    if case_resolution_state is None:
        raise InvalidDatasetError(f"unsupported reconciliation.availability: {recon_availability!r}")

    integration_state = _INTEGRATION_OUTCOME_MAP.get(
        str(effect.get("immediate_integration_outcome", "")).lower(),
        "INDETERMINATE",
    )
    order_status = _ORDER_LIFECYCLE_STATUS_MAP.get(
        str(order.get("lifecycle_intent", "")),
        "AWAITING_PAYMENT_CONFIRMATION",
    )
    correlation_id = _slug_reference(effect_logical, "CORR")
    order_created = deterministic_timestamp(order_logical)
    intent_requested = deterministic_timestamp(effect_logical, offset_seconds=30)
    effect_observed = deterministic_timestamp(f"{effect_logical}:wire", offset_seconds=90)
    reality_processed = deterministic_timestamp(f"{effect_logical}:sor:{variant_id}", offset_seconds=45)
    knowledge_observed = deterministic_timestamp(str(knowledge["snapshot_id"]), offset_seconds=95)

    inventory_knowledge = str(
        inventory.get("reservation_state_at_scenario_entry", "")
    ).upper()
    order_payment_substate = str(knowledge.get("order_payment_state", "")).upper()

    return {
        "commerce.organizations": {
            "organization_id": state.organization_id,
            "organization_key": str(order["customer_reference"]),
            "legal_name": str(order["customer_reference"]),
            "account_reference": str(order["customer_reference"]),
            "created_at": order_created,
            "updated_at": order_created,
        },
        "commerce.orders": {
            "order_id": state.order_id,
            "organization_id": state.organization_id,
            "order_number": _slug_reference(order_logical, "PO"),
            "amount": order["amount"],
            "currency": str(order["currency"]),
            "business_status": order_status,
            "created_at": order_created,
            "updated_at": order_created,
            "fulfillment_eligible_at": None,
        },
        "commerce.payment_intents": {
            "payment_intent_id": state.payment_intent_id,
            "order_id": state.order_id,
            "intent_reference": _slug_reference(effect_logical, "PAY"),
            "amount": order["amount"],
            "currency": str(order["currency"]),
            "correlation_id": correlation_id,
            "attempt_ordinal": 1,
            "application_status": order_payment_substate,
            "requested_at": intent_requested,
            "created_at": intent_requested,
            "updated_at": intent_requested,
        },
        "external_sor.external_payment_effects": {
            "external_payment_effect_id": state.external_payment_effect_id,
            "payment_intent_id": state.payment_intent_id,
            "external_effect_reference": _slug_reference(effect_logical, "EXT"),
            "correlation_id": correlation_id,
            "requested_state": str(effect.get("business_operation", "capture")).upper(),
            "observed_integration_state": integration_state,
            "requested_at": intent_requested,
            "observed_at": effect_observed,
            "created_at": effect_observed,
        },
        "external_sor.external_reality": {
            "external_reality_id": state.external_reality_id,
            "external_payment_effect_id": state.external_payment_effect_id,
            "correlation_id": correlation_id,
            "sor_transaction_ref": _slug_reference(f"{effect_logical}:{variant_id}", "SOR"),
            "terminal_outcome": reality_fields["terminal_outcome"],
            "funds_captured": reality_fields["funds_captured"],
            "truth_availability_state": reality_fields["truth_availability_state"],
            "processed_at": reality_processed,
            "created_at": reality_processed,
        },
        "commerce.application_knowledge": {
            "application_knowledge_id": state.application_knowledge_id,
            "payment_intent_id": state.payment_intent_id,
            "external_payment_effect_id": state.external_payment_effect_id,
            "known_status": "UNKNOWN",
            "order_payment_substate": order_payment_substate,
            "confirmation_received": bool(knowledge.get("confirmation_received")),
            "uncertainty_explicit": True,
            "inventory_reservation_knowledge": inventory_knowledge,
            "observed_at": knowledge_observed,
            "created_at": knowledge_observed,
            "updated_at": knowledge_observed,
        },
        "reconciliation.reconciliation_cases": {
            "reconciliation_case_id": state.reconciliation_case_id,
            "case_reference": _slug_reference(f"{order_logical}:{variant_id}", "RECON"),
            "order_id": state.order_id,
            "payment_intent_id": state.payment_intent_id,
            "correlation_id": correlation_id,
            "variant_context": variant_id,
            "resolution_state": case_resolution_state,
            "opened_at": knowledge_observed,
            "resolved_at": None,
            "created_at": knowledge_observed,
            "updated_at": knowledge_observed,
        },
    }
