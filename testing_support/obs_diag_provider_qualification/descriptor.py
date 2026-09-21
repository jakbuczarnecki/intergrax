# © Artur Czarnecki. All rights reserved.

"""Provider qualification descriptor for OBS/DIAG external provider matrix (X5 / X5A)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ObsDiagProviderSupportStatus(StrEnum):
    """Explicit X5 taxonomy — adapter existence ≠ qualification."""

    SUPPORTED_QUALIFIED = "SUPPORTED + QUALIFIED"
    SUPPORTED_NOT_QUALIFIED = "SUPPORTED + NOT QUALIFIED"
    EXPERIMENTAL = "EXPERIMENTAL"
    ADAPTER_ONLY = "ADAPTER ONLY"
    NOT_SUPPORTED = "NOT SUPPORTED"


class ObsDiagProviderDomain(StrEnum):
    PERSISTENCE = "persistence"
    TRANSPORT = "transport"
    TELEMETRY = "telemetry"


class ObsDiagProviderClass(StrEnum):
    PLATFORM_INTERNAL = "PLATFORM_INTERNAL"
    PLATFORM_EXPORT_SEMANTICS = "PLATFORM_EXPORT_SEMANTICS"
    EXTERNAL_VENDOR = "EXTERNAL_VENDOR"


@dataclass(frozen=True, slots=True)
class ObsDiagProviderQualificationDescriptor:
    """Minimal qualification record for matrix reconciliation and harness metadata."""

    provider_id: str
    domain: ObsDiagProviderDomain
    provider_class: ObsDiagProviderClass
    contract: str
    integration_status: str
    adapter_exists: bool
    live_proof_module: str | None
    failure_recovery_proof_module: str | None
    declared_status: ObsDiagProviderSupportStatus
    discovery_source: str
    delivery_or_durability_note: str = ""
