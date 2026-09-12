"""System-of-record truth resolution — shared with provisioning materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.reference.dataset_manifest import (
    InvalidDatasetError,
)


@dataclass(frozen=True, slots=True)
class SorTruthFields:
    terminal_outcome: str
    funds_captured: bool
    truth_availability_state: str


_SOR_TRUTH_BY_CAPTURE_OUTCOME: dict[str, SorTruthFields] = {
    "completed": SorTruthFields(
        terminal_outcome="PAYMENT_COMPLETED",
        funds_captured=True,
        truth_availability_state="AVAILABLE",
    ),
    "failed": SorTruthFields(
        terminal_outcome="PAYMENT_FAILED",
        funds_captured=False,
        truth_availability_state="AVAILABLE",
    ),
}

_SOR_TRUTH_BY_ESTABLISHMENT: dict[str, SorTruthFields] = {
    "unavailable_within_policy": SorTruthFields(
        terminal_outcome="TRUTH_INDETERMINATE",
        funds_captured=False,
        truth_availability_state="UNAVAILABLE",
    ),
}


def resolve_sor_truth_fields(variant_document: dict[str, Any]) -> SorTruthFields:
    """Map variant external_reality slice to persisted SoR columns (data-driven)."""
    external_reality = variant_document.get("external_reality")
    if not isinstance(external_reality, dict):
        raise InvalidDatasetError("external_reality must be an object")
    sor_truth = external_reality.get("system_of_record_truth")
    if not isinstance(sor_truth, dict):
        raise InvalidDatasetError("external_reality.system_of_record_truth must be an object")

    capture_outcome = sor_truth.get("payment_capture_outcome")
    if isinstance(capture_outcome, str):
        mapped = _SOR_TRUTH_BY_CAPTURE_OUTCOME.get(capture_outcome)
        if mapped is None:
            raise InvalidDatasetError(f"unsupported payment_capture_outcome: {capture_outcome!r}")
        return mapped

    establishment = sor_truth.get("authoritative_truth_establishment")
    if isinstance(establishment, str):
        mapped = _SOR_TRUTH_BY_ESTABLISHMENT.get(establishment)
        if mapped is None:
            raise InvalidDatasetError(
                f"unsupported authoritative_truth_establishment: {establishment!r}"
            )
        return mapped

    raise InvalidDatasetError(
        "system_of_record_truth must declare payment_capture_outcome "
        "or authoritative_truth_establishment"
    )
