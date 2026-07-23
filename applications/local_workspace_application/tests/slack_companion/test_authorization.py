# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.integrations.contracts.conversation_channel import (
    ConversationActor,
    ConversationAddress,
    ConversationEventKind,
    InboundConversationEvent,
)
from local_workspace_application.slack_companion.authorization import (
    SlackCompanionAuthConfig,
    authorize_inbound_ask,
)

pytestmark = pytest.mark.unit

_AUTH = SlackCompanionAuthConfig(
    approved_team_id="T_APPROVED",
    approved_user_id="U_APPROVED",
    tenant_id="tenant-1",
    active_workspace_id="ws-1",
)


def _message(
    *,
    team_id: str = "T_APPROVED",
    user_id: str = "U_APPROVED",
    event_id: str = "Ev123",
    text: str = "What is the policy?",
    is_bot: bool = False,
) -> InboundConversationEvent:
    return InboundConversationEvent(
        event_id=event_id,
        address=ConversationAddress(
            installation_id=team_id,
            conversation_id="D123",
            thread_id="1710000.000100",
        ),
        actor=ConversationActor(actor_id=user_id, is_bot=is_bot),
        kind=ConversationEventKind.MESSAGE,
        text=text,
        occurred_at=datetime.now(timezone.utc),
    )


def test_approved_team_and_user_accepted() -> None:
    result = authorize_inbound_ask(_message(), config=_AUTH)
    assert result is not None
    assert result.tenant_id == "tenant-1"
    assert result.workspace_id == "ws-1"
    assert result.question == "What is the policy?"
    assert result.event_id == "Ev123"


def test_wrong_installation_team_rejected() -> None:
    assert authorize_inbound_ask(_message(team_id="T_OTHER"), config=_AUTH) is None


def test_wrong_user_rejected() -> None:
    assert authorize_inbound_ask(_message(user_id="U_OTHER"), config=_AUTH) is None


def test_bot_actor_rejected() -> None:
    assert authorize_inbound_ask(_message(is_bot=True), config=_AUTH) is None


def test_blank_message_rejected_by_contract() -> None:
    with pytest.raises(ValueError):
        _message(text="   ")


def test_whitespace_question_not_accepted_via_stripped_empty() -> None:
    # Direct model construction with strip-empty is rejected by contract.
    # Authorization also requires non-blank text when somehow present.
    event = InboundConversationEvent.model_construct(
        event_id="Ev123",
        address=ConversationAddress(
            installation_id="T_APPROVED",
            conversation_id="D123",
            thread_id="1710000.000100",
        ),
        actor=ConversationActor(actor_id="U_APPROVED", is_bot=False),
        kind=ConversationEventKind.MESSAGE,
        text="   ",
        occurred_at=None,
        action=None,
        metadata={},
    )
    assert authorize_inbound_ask(event, config=_AUTH) is None


def test_missing_event_identity_fail_closed() -> None:
    event = InboundConversationEvent.model_construct(
        event_id="",
        address=ConversationAddress(
            installation_id="T_APPROVED",
            conversation_id="D123",
            thread_id="1710000.000100",
        ),
        actor=ConversationActor(actor_id="U_APPROVED", is_bot=False),
        kind=ConversationEventKind.MESSAGE,
        text="hello",
        occurred_at=None,
        action=None,
        metadata={},
    )
    assert authorize_inbound_ask(event, config=_AUTH) is None
