# © Artur Czarnecki. All rights reserved.

"""Bootstrap shipped security defense bundles and optional entry points (Phase SEC-BUNDLE-3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.runtime.security.defense_plugin_loader import (
    SecurityDefenseAdmissionPolicy,
    load_security_defense_plugin_report,
)
from intergrax.runtime.security.defense_registry import list_shipped_defense_bundle_ids


def register_security_payload_schemas() -> None:
    """Register typed payloads for platform.security spine kinds (SEC-ENT-2)."""
    from intergrax.runtime.events.event_kind_registry import register_event_kind
    from intergrax.runtime.events.payload_registry import register_payload_schema
    from intergrax.runtime.security.payloads import (
        SecurityDefenseBlockedPayloadV1,
        SecurityEncryptionDeniedPayloadV1,
    )
    from intergrax.runtime.security.security_events import (
        KIND_DEFENSE_BLOCKED,
        KIND_ENCRYPTION_DENIED,
    )

    register_payload_schema(SecurityDefenseBlockedPayloadV1)
    register_payload_schema(SecurityEncryptionDeniedPayloadV1)
    register_event_kind(KIND_DEFENSE_BLOCKED, SecurityDefenseBlockedPayloadV1.schema_id)
    register_event_kind(KIND_ENCRYPTION_DENIED, SecurityEncryptionDeniedPayloadV1.schema_id)


@dataclass(frozen=True, slots=True)
class SecurityBootstrapResult:
    shipped_bundle_ids: tuple[str, ...]
    entry_point_plugins: int
    load_report: DomainPluginLoadReport
    critical_bootstrap_acceptable: bool


def bootstrap_security_providers(
    *,
    discover_entry_points: bool = False,
    admission: SecurityDefenseAdmissionPolicy | None = None,
) -> SecurityBootstrapResult:
    """
    Register shipped defense bundles (always available) and optional EP plugins.

    Shipped bundles are registered at import time via ``defense_registry``;
    this function loads third-party entry points when requested.

    Isolation examines sibling EPs; this result does not exit the process.
    Production hosts must treat ``critical_bootstrap_acceptable`` as fail-closed.
    """
    report = load_security_defense_plugin_report(
        discover_entry_points=discover_entry_points,
        admission=admission,
    )
    register_security_payload_schemas()
    return SecurityBootstrapResult(
        shipped_bundle_ids=list_shipped_defense_bundle_ids(),
        entry_point_plugins=report.registered_count,
        load_report=report,
        critical_bootstrap_acceptable=report.critical_bootstrap_acceptable,
    )
