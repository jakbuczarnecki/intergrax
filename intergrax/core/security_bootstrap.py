# © Artur Czarnecki. All rights reserved.

"""Bootstrap shipped security defense bundles and optional entry points (Phase SEC-BUNDLE-3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.security.defense_plugin_loader import load_security_defense_plugins
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


def bootstrap_security_providers(*, discover_entry_points: bool = False) -> SecurityBootstrapResult:
    """
    Register shipped defense bundles (always available) and optional EP plugins.

    Shipped bundles are registered at import time via ``defense_registry``;
    this function loads third-party entry points when requested.
    """
    loaded = load_security_defense_plugins(discover_entry_points=discover_entry_points)
    register_security_payload_schemas()
    return SecurityBootstrapResult(
        shipped_bundle_ids=list_shipped_defense_bundle_ids(),
        entry_point_plugins=loaded,
    )
