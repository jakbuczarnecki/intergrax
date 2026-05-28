# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reusable Microsoft Teams activity parsing (Bot Framework / outgoing webhook)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from intergrax.runtime.interactions.parsers.slash_command import parse_slash_command_text


def strip_teams_mentions(text: str, entities: Optional[List[Dict[str, Any]]]) -> str:
    """Remove ``<at>…</at>`` mention tokens using activity ``entities`` when present."""
    cleaned = (text or "").strip()
    if not cleaned or not entities:
        return cleaned
    for entity in entities:
        if not isinstance(entity, dict) or entity.get("type") != "mention":
            continue
        mention_text = str(entity.get("text") or "")
        if mention_text:
            cleaned = cleaned.replace(mention_text, "")
    return cleaned.strip()


def parse_teams_activity_text(
    text: str,
    *,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[str], str]:
    """Strip bot mentions, then parse ``capability message`` from Teams activity text."""
    normalized = strip_teams_mentions(text, entities)
    return parse_slash_command_text(normalized)


def extract_teams_tenant_id(payload: Dict[str, Any], *, fallback: str) -> str:
    conversation = payload.get("conversation")
    if isinstance(conversation, dict):
        tenant_id = conversation.get("tenantId")
        if tenant_id:
            return str(tenant_id)
    channel_data = payload.get("channelData")
    if isinstance(channel_data, dict):
        tenant = channel_data.get("tenant")
        if isinstance(tenant, dict) and tenant.get("id"):
            return str(tenant["id"])
        teams_team_id = channel_data.get("teamsTeamId")
        if teams_team_id:
            return str(teams_team_id)
    explicit = payload.get("tenant_id")
    if explicit:
        return str(explicit)
    return fallback


def extract_teams_user_id(payload: Dict[str, Any], *, fallback: str) -> str:
    sender = payload.get("from")
    if isinstance(sender, dict):
        for key in ("aadObjectId", "id"):
            value = sender.get(key)
            if value:
                return str(value)
    explicit = payload.get("user_id")
    if explicit:
        return str(explicit)
    return fallback
