# © Artur Czarnecki. All rights reserved.

"""EC3 observability vendor qualification evidence (projection-only vendors)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ObsDiagProofKind(StrEnum):
    UNIT_CONTRACT = "unit_contract"
    INTEGRATION = "integration"
    EXTERNAL_LIVE = "external_live"


class ObservabilityVendorQualificationStatus(StrEnum):
    ADAPTER_ONLY = "ADAPTER ONLY"
    CONTRACT_CONFORMANT = "CONTRACT CONFORMANT"
    LIVE_QUALIFIED = "LIVE QUALIFIED"


@dataclass(frozen=True, slots=True)
class ObsDiagProofReference:
    kind: ObsDiagProofKind
    module: str


@dataclass(frozen=True, slots=True)
class ObservabilityVendorQualificationEvidence:
    normal_delivery: ObsDiagProofReference | None
    failure_isolation: ObsDiagProofReference | None
    recovery: ObsDiagProofReference | None
    canonical_truth_isolation: ObsDiagProofReference | None
    privacy: ObsDiagProofReference | None

    def missing_live_categories(self) -> tuple[str, ...]:
        missing: list[str] = []
        for field in (
            "normal_delivery",
            "failure_isolation",
            "recovery",
            "canonical_truth_isolation",
            "privacy",
        ):
            if getattr(self, field) is None:
                missing.append(field)
        return tuple(missing)


@dataclass(frozen=True, slots=True)
class ObservabilityVendorQualificationRow:
    provider_id: str
    manifest_path: str
    integration_status: str
    contract_implemented: bool
    evidence: ObservabilityVendorQualificationEvidence
    qualification: ObservabilityVendorQualificationStatus


@dataclass(frozen=True, slots=True)
class ObservabilityQualifiedPathRow:
    path_id: str
    evidence: ObservabilityVendorQualificationEvidence
    qualification: ObservabilityVendorQualificationStatus
    privacy_required: bool = True
