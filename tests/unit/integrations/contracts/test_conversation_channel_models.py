# © Artur Czarnecki. All rights reserved.

"""Shared conversation channel model contract tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.conversation_channel import (
    MAX_CONVERSATION_CHOICE_OPTIONS,
    ConversationActionSelection,
    ConversationActor,
    ConversationAddress,
    ConversationChoiceOption,
    ConversationDeliveryReceipt,
    ConversationEventKind,
    ConversationSingleChoice,
    InboundConversationEvent,
    OutboundConversationMessage,
)

pytestmark = pytest.mark.unit

_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "team_id",
        "workspace_id",
        "channel_id",
        "thread_ts",
        "envelope_id",
        "block_id",
        "block_actions",
        "adaptive_card",
        "guild_id",
        "chat_id",
        "update_id",
        "activity_id",
    }
)


def _address(**overrides: object) -> ConversationAddress:
    payload = {"installation_id": "inst-1", "conversation_id": "conv-1"}
    payload.update(overrides)
    return ConversationAddress.model_validate(payload)


def test_conversation_address_valid_and_optional_thread() -> None:
    address = _address(thread_id="thread-1")
    assert address.installation_id == "inst-1"
    assert address.conversation_id == "conv-1"
    assert address.thread_id == "thread-1"


@pytest.mark.parametrize("field", ["installation_id", "conversation_id"])
def test_conversation_address_rejects_blank_identifiers(field: str) -> None:
    with pytest.raises(ValidationError):
        _address(**{field: "   "})


def test_shared_models_are_vendor_neutral() -> None:
    models = (
        ConversationAddress,
        ConversationActor,
        ConversationActionSelection,
        ConversationChoiceOption,
        ConversationSingleChoice,
        InboundConversationEvent,
        OutboundConversationMessage,
        ConversationDeliveryReceipt,
    )
    for model in models:
        field_names = set(model.model_fields)
        assert field_names.isdisjoint(_FORBIDDEN_FIELD_NAMES)


def test_conversation_actor_valid_and_bot_flag() -> None:
    actor = ConversationActor(actor_id="u-1", display_name="Ada", is_bot=True)
    assert actor.actor_id == "u-1"
    assert actor.is_bot is True


def test_conversation_actor_rejects_blank_id() -> None:
    with pytest.raises(ValidationError):
        ConversationActor(actor_id=" ")


def test_inbound_message_requires_text_and_forbids_action() -> None:
    event = InboundConversationEvent(
        event_id="e-1",
        address=_address(),
        actor=ConversationActor(actor_id="u-1"),
        kind=ConversationEventKind.MESSAGE,
        text="hello",
    )
    assert event.text == "hello"
    assert event.metadata == {}

    with pytest.raises(ValidationError):
        InboundConversationEvent(
            event_id="e-2",
            address=_address(),
            actor=ConversationActor(actor_id="u-1"),
            kind=ConversationEventKind.MESSAGE,
            text=" ",
        )

    with pytest.raises(ValidationError):
        InboundConversationEvent(
            event_id="e-3",
            address=_address(),
            actor=ConversationActor(actor_id="u-1"),
            kind=ConversationEventKind.MESSAGE,
            text="hello",
            action=ConversationActionSelection(action_id="a1", selected_value="v1"),
        )


def test_inbound_action_requires_selection() -> None:
    event = InboundConversationEvent(
        event_id="e-4",
        address=_address(),
        actor=ConversationActor(actor_id="u-1"),
        kind=ConversationEventKind.ACTION,
        action=ConversationActionSelection(action_id="pick", selected_value="ws-1"),
        text="optional",
    )
    assert event.action is not None

    with pytest.raises(ValidationError):
        InboundConversationEvent(
            event_id="e-5",
            address=_address(),
            actor=ConversationActor(actor_id="u-1"),
            kind=ConversationEventKind.ACTION,
        )


def test_inbound_metadata_defaults_are_independent() -> None:
    first = InboundConversationEvent(
        event_id="e-6",
        address=_address(),
        actor=ConversationActor(actor_id="u-1"),
        kind=ConversationEventKind.MESSAGE,
        text="a",
    )
    second = InboundConversationEvent(
        event_id="e-7",
        address=_address(),
        actor=ConversationActor(actor_id="u-1"),
        kind=ConversationEventKind.MESSAGE,
        text="b",
    )
    assert first.metadata is not second.metadata
    first.metadata["x"] = 1
    assert "x" not in second.metadata


def test_single_choice_option_bounds() -> None:
    one = ConversationSingleChoice(
        action_id="pick",
        options=(ConversationChoiceOption(value="a", label="A"),),
    )
    assert len(one.options) == 1

    options = tuple(
        ConversationChoiceOption(value=f"v{i}", label=f"L{i}")
        for i in range(MAX_CONVERSATION_CHOICE_OPTIONS)
    )
    ConversationSingleChoice(action_id="pick", options=options)

    with pytest.raises(ValidationError):
        ConversationSingleChoice(action_id="pick", options=())

    with pytest.raises(ValidationError):
        ConversationSingleChoice(
            action_id="pick",
            options=(
                ConversationChoiceOption(value="a", label="A"),
                ConversationChoiceOption(value="a", label="B"),
            ),
        )

    with pytest.raises(ValidationError):
        ConversationSingleChoice(
            action_id=" ",
            options=(ConversationChoiceOption(value="a", label="A"),),
        )

    too_many = options + (ConversationChoiceOption(value="extra", label="Extra"),)
    with pytest.raises(ValidationError):
        ConversationSingleChoice(action_id="pick", options=too_many)


def test_outbound_message_text_and_single_component() -> None:
    choice = ConversationSingleChoice(
        action_id="pick",
        options=(ConversationChoiceOption(value="a", label="A"),),
    )
    plain = OutboundConversationMessage(address=_address(), text="hello")
    with_choice = OutboundConversationMessage(address=_address(), text="choose", components=(choice,))
    assert plain.components == ()
    assert len(with_choice.components) == 1

    with pytest.raises(ValidationError):
        OutboundConversationMessage(address=_address(), text=" ")

    with pytest.raises(ValidationError):
        OutboundConversationMessage(address=_address(), text="x", components=(choice, choice))


def test_delivery_receipt_preserves_address() -> None:
    address = _address(thread_id="t-1")
    receipt = ConversationDeliveryReceipt(
        message_id="m-1",
        address=address,
        delivered_at=datetime.now(timezone.utc),
    )
    assert receipt.message_id == "m-1"
    assert receipt.address == address
