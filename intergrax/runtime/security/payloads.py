# © Artur Czarnecki. All rights reserved.

"""Typed payloads for platform.security spine signals (Phase SEC-ENT-2)."""

from __future__ import annotations

from intergrax.runtime.events.payloads import RuntimeEventPayload


class SecurityDefenseBlockedPayloadV1(RuntimeEventPayload):
    """Emitted when an S2 defense plugin blocks at a UAEP hook."""

    schema_id = "platform.security.defense_blocked.v1"

    plugin_id: str = ""
    hook_point: str = ""
    reason: str = ""


class SecurityEncryptionDeniedPayloadV1(RuntimeEventPayload):
    """Emitted when RESTRICTED payload is denied by encryption enforcement."""

    schema_id = "platform.security.encryption_denied.v1"

    hook_point: str = ""
    reason: str = ""
    classification: str | None = None
