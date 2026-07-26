# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Conversation-channel attachment contract unit tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.conversation_channel import (
    ConversationActionSelection,
    ConversationActor,
    ConversationAddress,
    ConversationAttachmentContent,
    ConversationAttachmentFetchError,
    ConversationAttachmentFetcher,
    ConversationAttachmentReference,
    ConversationEventKind,
    InboundConversationEvent,
)

pytestmark = pytest.mark.unit


def _address() -> ConversationAddress:
    return ConversationAddress(
        installation_id="T1",
        conversation_id="C1",
        thread_id="1.0",
    )


def test_text_only_message_accepted() -> None:
    event = InboundConversationEvent(
        event_id="e-1",
        address=_address(),
        actor=ConversationActor(actor_id="u-1"),
        kind=ConversationEventKind.MESSAGE,
        text="hello",
    )
    assert event.attachments == ()


def test_attachment_only_message_accepted() -> None:
    event = InboundConversationEvent(
        event_id="e-2",
        address=_address(),
        actor=ConversationActor(actor_id="u-1"),
        kind=ConversationEventKind.MESSAGE,
        text=None,
        attachments=(
            ConversationAttachmentReference(attachment_id="F1", file_name="a.pdf"),
        ),
    )
    assert event.text is None
    assert len(event.attachments) == 1


def test_message_with_text_and_attachments_accepted() -> None:
    event = InboundConversationEvent(
        event_id="e-3",
        address=_address(),
        actor=ConversationActor(actor_id="u-1"),
        kind=ConversationEventKind.MESSAGE,
        text="caption",
        attachments=(ConversationAttachmentReference(attachment_id="F1"),),
    )
    assert event.text == "caption"
    assert event.attachments[0].attachment_id == "F1"


def test_message_with_neither_rejected() -> None:
    with pytest.raises(ValidationError):
        InboundConversationEvent(
            event_id="e-4",
            address=_address(),
            actor=ConversationActor(actor_id="u-1"),
            kind=ConversationEventKind.MESSAGE,
            text="   ",
            attachments=(),
        )


def test_message_with_action_rejected() -> None:
    with pytest.raises(ValidationError):
        InboundConversationEvent(
            event_id="e-5",
            address=_address(),
            actor=ConversationActor(actor_id="u-1"),
            kind=ConversationEventKind.MESSAGE,
            text="hello",
            action=ConversationActionSelection(action_id="a", selected_value="v"),
        )


def test_action_with_attachments_rejected() -> None:
    with pytest.raises(ValidationError):
        InboundConversationEvent(
            event_id="e-6",
            address=_address(),
            actor=ConversationActor(actor_id="u-1"),
            kind=ConversationEventKind.ACTION,
            action=ConversationActionSelection(action_id="a", selected_value="v"),
            attachments=(ConversationAttachmentReference(attachment_id="F1"),),
        )


def test_attachment_reference_immutable() -> None:
    ref = ConversationAttachmentReference(attachment_id="F1", file_name="a.txt")
    with pytest.raises(ValidationError):
        ref.attachment_id = "F2"  # type: ignore[misc]


def test_attachment_content_immutable() -> None:
    content = ConversationAttachmentContent(
        attachment_id="F1",
        file_name="a.txt",
        content_type="text/plain",
        body=b"hi",
    )
    with pytest.raises(ValidationError):
        content.body = b"no"  # type: ignore[misc]


def test_blank_attachment_id_rejected() -> None:
    with pytest.raises(ValidationError):
        ConversationAttachmentReference(attachment_id="  ")


def test_negative_size_rejected() -> None:
    with pytest.raises(ValidationError):
        ConversationAttachmentReference(attachment_id="F1", size_bytes=-1)


def test_control_character_filename_rejected() -> None:
    with pytest.raises(ValidationError):
        ConversationAttachmentReference(attachment_id="F1", file_name="bad\nname")


def test_non_bytes_body_rejected() -> None:
    with pytest.raises(ValidationError):
        ConversationAttachmentContent(
            attachment_id="F1",
            file_name="a.txt",
            content_type="text/plain",
            body=bytearray(b"x"),  # type: ignore[arg-type]
        )


def test_optional_fetch_protocol_runtime_check() -> None:
    class _Fetcher:
        async def fetch_attachment(
            self,
            attachment: ConversationAttachmentReference,
            *,
            max_bytes: int,
        ) -> ConversationAttachmentContent:
            del max_bytes
            return ConversationAttachmentContent(
                attachment_id=attachment.attachment_id,
                file_name=attachment.file_name or "x.bin",
                content_type="application/octet-stream",
                body=b"",
            )

    assert isinstance(_Fetcher(), ConversationAttachmentFetcher)
    assert not isinstance(object(), ConversationAttachmentFetcher)


def test_fetch_error_exposes_stable_kind_only() -> None:
    err = ConversationAttachmentFetchError(kind="attachment_too_large")
    assert err.kind == "attachment_too_large"
    assert str(err) == "attachment_too_large"
    assert "token" not in str(err).casefold()
    with pytest.raises(ValueError):
        ConversationAttachmentFetchError(kind="  ")
