# © Artur Czarnecki. All rights reserved.

"""Fail-closed Slack identity authorization for the LKW companion."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.conversation_channel import (
    ConversationEventKind,
    InboundConversationEvent,
)
from local_workspace_application.slack_companion.models import (
    AuthorizedSlackAskContext,
    AuthorizedSlackMessageContext,
)


@dataclass(frozen=True, slots=True)
class SlackCompanionAuthConfig:
    approved_team_id: str
    approved_user_id: str
    tenant_id: str
    active_workspace_id: str


def authorize_inbound_message(
    event: InboundConversationEvent,
    *,
    config: SlackCompanionAuthConfig,
) -> AuthorizedSlackMessageContext | None:
    """Return an authorized message context or ``None`` when the event must be ignored."""
    if event.kind is not ConversationEventKind.MESSAGE:
        return None

    event_id = (event.event_id or "").strip()
    if not event_id:
        return None

    team_id = (event.address.installation_id or "").strip()
    approved_team = config.approved_team_id.strip()
    if not team_id or not approved_team or team_id != approved_team:
        return None

    actor_id = (event.actor.actor_id or "").strip()
    approved_user = config.approved_user_id.strip()
    if not actor_id or not approved_user or actor_id != approved_user:
        return None

    if event.actor.is_bot:
        return None

    text = (event.text or "").strip()
    if not text and not event.attachments:
        return None

    tenant_id = config.tenant_id.strip()
    workspace_id = config.active_workspace_id.strip()
    if not tenant_id or not workspace_id:
        return None

    return AuthorizedSlackMessageContext(
        team_id=team_id,
        user_id=actor_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        text=text,
        event_id=event_id,
    )


def authorize_inbound_ask(
    event: InboundConversationEvent,
    *,
    config: SlackCompanionAuthConfig,
) -> AuthorizedSlackAskContext | None:
    """Return an authorized Ask context or ``None`` when the event must be ignored.

    Fail-closed: any missing identity, bot actor, wrong team/user, or blank
    question yields ``None`` (no Ask, no product side effects).
    """
    authorized = authorize_inbound_message(event, config=config)
    if authorized is None:
        return None
    question = authorized.text.strip()
    if not question:
        return None
    return AuthorizedSlackAskContext(
        team_id=authorized.team_id,
        user_id=authorized.user_id,
        tenant_id=authorized.tenant_id,
        workspace_id=authorized.workspace_id,
        question=question,
        event_id=authorized.event_id,
    )
