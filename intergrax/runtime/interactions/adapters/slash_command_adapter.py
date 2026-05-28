# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slash-command intake — Slack/Teams-shaped payloads without vendor SDK (Phase H.2)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.interactions.metadata_keys import (
    INTERACTION_CHANNEL_KEY,
    INTERACTION_COMMAND_KEY,
    INTERACTION_RAW_ID_KEY,
    INTERACTION_RESPONSE_URL_KEY,
    INTERACTION_SOURCE_KEY,
    INTERACTION_TEAM_ID_KEY,
)
from intergrax.runtime.interactions.models import InboundInteraction
from intergrax.runtime.interactions.parsers.slash_command import parse_slash_command_text


class SlashCommandInteractionAdapter(InteractionAdapter):
    """
    Parses slash-command style payloads (Slack slash command, Teams, CLI).

    Expected keys (subset): ``command``, ``text``, ``user_id``, ``team_id``,
    ``trigger_id``, ``response_url``.
    """

    @property
    def channel(self) -> str:
        return "slash_command"

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("command"):
            return True
        text = str(payload.get("text") or "")
        return text.startswith("/")

    def to_inbound(self, payload: Dict[str, Any], *, tenant_id: str, user_id: str) -> InboundInteraction:
        command = str(payload.get("command") or "").strip()
        text = str(payload.get("text") or "").strip()
        if command and text:
            parse_input = text
        elif command:
            parse_input = command
        else:
            parse_input = text
        capability, message = parse_slash_command_text(parse_input)

        resolved_tenant = str(payload.get("tenant_id") or payload.get("team_id") or tenant_id)
        resolved_user = str(payload.get("user_id") or user_id)
        metadata = dict(payload.get("metadata") or {})
        metadata[INTERACTION_CHANNEL_KEY] = self.channel
        metadata[INTERACTION_SOURCE_KEY] = str(payload.get("source") or "slash_command")
        if command:
            metadata[INTERACTION_COMMAND_KEY] = command
        team_id = payload.get("team_id")
        if team_id:
            metadata[INTERACTION_TEAM_ID_KEY] = str(team_id)
        response_url = payload.get("response_url")
        if response_url:
            metadata[INTERACTION_RESPONSE_URL_KEY] = str(response_url)
        trigger_id = payload.get("trigger_id")
        interaction_id: Optional[str] = str(trigger_id) if trigger_id else None

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
