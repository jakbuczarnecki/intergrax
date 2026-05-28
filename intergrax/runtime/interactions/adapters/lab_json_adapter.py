# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Laboratory JSON intake — vendor-neutral dict → Task (Phase H.2)."""

from __future__ import annotations

from typing import Any, Dict

from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.interactions.metadata_keys import (
    INTERACTION_CHANNEL_KEY,
    INTERACTION_SOURCE_KEY,
)
from intergrax.runtime.interactions.models import InboundInteraction


class LabJsonInteractionAdapter(InteractionAdapter):
    """
    Accepts a normalized lab payload::

        {
            "tenant_id": "t1",
            "user_id": "u1",
            "message": "hello",
            "capability": "echo.basic",
            "metadata": {"source": "notebook"}
        }
    """

    @property
    def channel(self) -> str:
        return "lab"

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("command"):
            return False
        return bool(payload.get("message") or payload.get("capability"))

    def to_inbound(self, payload: Dict[str, Any], *, tenant_id: str, user_id: str) -> InboundInteraction:
        resolved_tenant = str(payload.get("tenant_id") or tenant_id)
        resolved_user = str(payload.get("user_id") or user_id)
        metadata = dict(payload.get("metadata") or {})
        metadata[INTERACTION_CHANNEL_KEY] = self.channel
        metadata[INTERACTION_SOURCE_KEY] = str(payload.get("source") or "lab_json")
        return InboundInteraction(
            channel=self.channel,
            tenant_id=resolved_tenant,
            user_id=resolved_user,
            message=str(payload.get("message") or ""),
            capability=payload.get("capability"),
            session_id=payload.get("session_id"),
            interaction_id=payload.get("interaction_id"),
            metadata=metadata,
            raw_payload=dict(payload),
        )
