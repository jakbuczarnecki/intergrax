# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared outbound conversation message → Slack Web API / Block Kit."""

from __future__ import annotations

from typing import Any

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import (
    ConversationSingleChoice,
    OutboundConversationMessage,
)

# Slack Block Kit hard limits smaller than shared conversation bounds.
SLACK_OPTION_TEXT_MAX = 75
SLACK_OPTION_VALUE_MAX = 75
SLACK_PLACEHOLDER_MAX = 150
_DEFAULT_PLACEHOLDER = "Choose an option"


class SlackConversationRenderError(IntegrationConfigurationError):
    """Outbound message cannot be rendered to Slack without silent truncation."""


def _validate_slack_choice(choice: ConversationSingleChoice) -> None:
    if len(choice.action_id) > 255:
        raise SlackConversationRenderError("Slack action_id exceeds 255 characters")
    prompt = choice.prompt.strip() if choice.prompt else _DEFAULT_PLACEHOLDER
    if len(prompt) > SLACK_PLACEHOLDER_MAX:
        raise SlackConversationRenderError(
            f"Slack placeholder exceeds {SLACK_PLACEHOLDER_MAX} characters",
        )
    for option in choice.options:
        if len(option.label) > SLACK_OPTION_TEXT_MAX:
            raise SlackConversationRenderError(
                f"Slack option label exceeds {SLACK_OPTION_TEXT_MAX} characters",
            )
        if len(option.value) > SLACK_OPTION_VALUE_MAX:
            raise SlackConversationRenderError(
                f"Slack option value exceeds {SLACK_OPTION_VALUE_MAX} characters",
            )


def render_single_choice_blocks(choice: ConversationSingleChoice, *, text: str) -> list[dict[str, Any]]:
    """Render one ConversationSingleChoice as Slack static_select Block Kit."""
    _validate_slack_choice(choice)
    placeholder = (choice.prompt or _DEFAULT_PLACEHOLDER).strip() or _DEFAULT_PLACEHOLDER
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
            "accessory": {
                "type": "static_select",
                "action_id": choice.action_id,
                "placeholder": {"type": "plain_text", "text": placeholder},
                "options": [
                    {
                        "text": {"type": "plain_text", "text": option.label},
                        "value": option.value,
                    }
                    for option in choice.options
                ],
            },
        }
    ]


def render_chat_post_message_args(message: OutboundConversationMessage) -> dict[str, Any]:
    """Map outbound conversation message to ``chat.postMessage`` keyword arguments."""
    args: dict[str, Any] = {
        "channel": message.address.conversation_id,
        "text": message.text,
    }
    if message.address.thread_id is not None:
        args["thread_ts"] = message.address.thread_id
    if message.components:
        choice = message.components[0]
        args["blocks"] = render_single_choice_blocks(choice, text=message.text)
    return args


__all__ = [
    "SLACK_OPTION_TEXT_MAX",
    "SLACK_OPTION_VALUE_MAX",
    "SLACK_PLACEHOLDER_MAX",
    "SlackConversationRenderError",
    "render_chat_post_message_args",
    "render_single_choice_blocks",
]
