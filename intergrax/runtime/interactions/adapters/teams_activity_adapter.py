# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Teams activity intake — Bot Framework JSON → Task (Phase H.5)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.interactions.metadata_keys import (
    INTERACTION_CHANNEL_KEY,
    INTERACTION_RAW_ID_KEY,
    INTERACTION_SOURCE_KEY,
    INTERACTION_TEAM_ID_KEY,
)
from intergrax.runtime.interactions.models import InboundInteraction
from intergrax.runtime.interactions.parsers.teams_activity import (
    extract_teams_tenant_id,
    extract_teams_user_id,
    parse_teams_activity_text,
)


class TeamsActivityInteractionAdapter(InteractionAdapter):
    """
    Parses Microsoft Teams Bot Framework / outgoing-webhook activities.

    Expected shape (subset): ``type``, ``channelId``, ``text``, ``from``,
    ``conversation``, ``entities``, ``serviceUrl``, ``id``.
    """

    @property
    def channel(self) -> str:
        return "teams"

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("command"):
            return False
        channel_id = str(payload.get("channelId") or "").lower()
        if channel_id == "msteams":
            return True
        activity_type = str(payload.get("type") or "").lower()
        service_url = str(payload.get("serviceUrl") or "").lower()
        return activity_type in {"message", "invoke"} and "teams" in service_url

    def to_inbound(self, payload: Dict[str, Any], *, tenant_id: str, user_id: str) -> InboundInteraction:
        entities = payload.get("entities")
        entity_list = entities if isinstance(entities, list) else None
        text = str(payload.get("text") or "")
        capability, message = parse_teams_activity_text(text, entities=entity_list)

        resolved_tenant = extract_teams_tenant_id(payload, fallback=tenant_id)
        resolved_user = extract_teams_user_id(payload, fallback=user_id)
        metadata = dict(payload.get("metadata") or {})
        metadata[INTERACTION_CHANNEL_KEY] = self.channel
        metadata[INTERACTION_SOURCE_KEY] = str(payload.get("source") or "teams_activity")

        channel_data = payload.get("channelData")
        if isinstance(channel_data, dict):
            teams_team_id = channel_data.get("teamsTeamId")
            if teams_team_id:
                metadata[INTERACTION_TEAM_ID_KEY] = str(teams_team_id)

        activity_id = payload.get("id")
        interaction_id: Optional[str] = str(activity_id) if activity_id else None
        if interaction_id:
            metadata[INTERACTION_RAW_ID_KEY] = interaction_id

        service_url = payload.get("serviceUrl")
        if service_url:
            metadata["interaction_service_url"] = str(service_url)

        return InboundInteraction(
            channel=self.channel,
            tenant_id=resolved_tenant,
            user_id=resolved_user,
            message=message,
            capability=capability,
            session_id=payload.get("session_id"),
            interaction_id=interaction_id,
            metadata=metadata,
            raw_payload=dict(payload),
        )
