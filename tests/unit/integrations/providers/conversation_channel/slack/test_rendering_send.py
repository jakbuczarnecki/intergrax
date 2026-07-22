# © Artur Czarnecki. All rights reserved.

"""Slack rendering and outbound send tests."""

from __future__ import annotations

from typing import Any

import pytest

from intergrax.integrations.contracts.conversation_channel import (
    ConversationAddress,
    ConversationChoiceOption,
    ConversationSingleChoice,
    OutboundConversationMessage,
)
from intergrax.integrations.providers.conversation_channel.slack.backend import (
    SlackConversationChannelBackend,
    SlackConversationSendError,
)
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.rendering import (
    SlackConversationRenderError,
    render_chat_post_message_args,
)

pytestmark = pytest.mark.unit

_ADDRESS = ConversationAddress(
    installation_id="TTEAM1",
    conversation_id="DCHANNEL1",
    thread_id="1710000000.000100",
)


class _FakeWebClient:
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _FakeSlackApiError(Exception):
    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.response = {"ok": False, "error": error}


def _backend(web: _FakeWebClient) -> SlackConversationChannelBackend:
    return SlackConversationChannelBackend(
        config=SlackConversationChannelIntegrationConfig(
            enabled=True,
            app_token="xapp-test-aaaa",
            bot_token="xoxb-test-bbbb",
        ),
        web_client=web,
        socket_client=object(),
        slack_api_error_cls=_FakeSlackApiError,
    )


def test_plain_text_chat_post_message_args() -> None:
    args = render_chat_post_message_args(
        OutboundConversationMessage(address=_ADDRESS, text="hello"),
    )
    assert args["channel"] == "DCHANNEL1"
    assert args["text"] == "hello"
    assert args["thread_ts"] == "1710000000.000100"
    assert "blocks" not in args


def test_missing_thread_id_omits_thread_ts() -> None:
    address = ConversationAddress(installation_id="TTEAM1", conversation_id="DCHANNEL1")
    args = render_chat_post_message_args(OutboundConversationMessage(address=address, text="hello"))
    assert "thread_ts" not in args


def test_single_choice_maps_to_static_select() -> None:
    choice = ConversationSingleChoice(
        action_id="choose",
        prompt="Pick one",
        options=(
            ConversationChoiceOption(value="a", label="Option A"),
            ConversationChoiceOption(value="b", label="Option B"),
        ),
    )
    args = render_chat_post_message_args(
        OutboundConversationMessage(address=_ADDRESS, text="Choose workspace", components=(choice,)),
    )
    assert args["text"] == "Choose workspace"
    block = args["blocks"][0]
    assert block["accessory"]["type"] == "static_select"
    assert block["accessory"]["action_id"] == "choose"
    assert block["accessory"]["placeholder"]["text"] == "Pick one"
    assert block["accessory"]["options"][0]["text"]["text"] == "Option A"
    assert block["accessory"]["options"][0]["value"] == "a"


def test_slack_option_label_limit_raises() -> None:
    choice = ConversationSingleChoice(
        action_id="choose",
        options=(ConversationChoiceOption(value="a", label="x" * 76),),
    )
    with pytest.raises(SlackConversationRenderError):
        render_chat_post_message_args(
            OutboundConversationMessage(address=_ADDRESS, text="t", components=(choice,)),
        )


@pytest.mark.asyncio
async def test_send_maps_response_ts_to_receipt() -> None:
    web = _FakeWebClient({"ok": True, "ts": "1710000000.000999"})
    backend = _backend(web)
    receipt = await backend.send(OutboundConversationMessage(address=_ADDRESS, text="hello"))
    assert receipt.message_id == "1710000000.000999"
    assert receipt.address == _ADDRESS
    assert web.calls[0]["channel"] == "DCHANNEL1"
    assert web.calls[0]["thread_ts"] == "1710000000.000100"


@pytest.mark.asyncio
async def test_malformed_slack_response_raises() -> None:
    web = _FakeWebClient({"ok": True})
    backend = _backend(web)
    with pytest.raises(SlackConversationSendError):
        await backend.send(OutboundConversationMessage(address=_ADDRESS, text="hello"))


@pytest.mark.asyncio
async def test_slack_api_error_raises_typed_error() -> None:
    web = _FakeWebClient(_FakeSlackApiError("channel_not_found"))
    backend = _backend(web)
    with pytest.raises(SlackConversationSendError, match="channel_not_found"):
        await backend.send(OutboundConversationMessage(address=_ADDRESS, text="hello"))
