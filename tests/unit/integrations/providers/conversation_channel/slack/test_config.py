# © Artur Czarnecki. All rights reserved.

"""Slack conversation-channel configuration tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)

pytestmark = pytest.mark.unit

_FAKE_APP = "xapp-test-aaaa"
_FAKE_BOT = "xoxb-test-bbbb"


def test_disabled_config_allows_absent_tokens() -> None:
    config = SlackConversationChannelIntegrationConfig(enabled=False)
    assert config.app_token is None
    assert config.bot_token is None


def test_enabled_runtime_rejects_absent_app_token() -> None:
    config = SlackConversationChannelIntegrationConfig(
        enabled=True,
        bot_token=_FAKE_BOT,
    )
    with pytest.raises(IntegrationConfigurationError):
        config.validate_for_runtime()


def test_enabled_runtime_rejects_absent_bot_token() -> None:
    config = SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token=_FAKE_APP,
    )
    with pytest.raises(IntegrationConfigurationError):
        config.validate_for_runtime()


def test_wrong_app_token_prefix_rejected() -> None:
    with pytest.raises(IntegrationConfigurationError, match="xapp-"):
        SlackConversationChannelIntegrationConfig(app_token="xoxb-wrong-prefix")


def test_wrong_bot_token_prefix_rejected() -> None:
    with pytest.raises(IntegrationConfigurationError, match="xoxb-"):
        SlackConversationChannelIntegrationConfig(bot_token="xapp-wrong-prefix")


def test_wrong_knowledge_user_token_prefix_rejected() -> None:
    with pytest.raises(IntegrationConfigurationError, match="xoxp-"):
        SlackConversationChannelIntegrationConfig(knowledge_user_token="xoxb-wrong-prefix")


def test_blank_knowledge_user_token_normalizes_to_none() -> None:
    config = SlackConversationChannelIntegrationConfig(knowledge_user_token="   ")
    assert config.knowledge_user_token is None


def test_knowledge_user_token_absent_from_repr() -> None:
    token = "xoxp-test-knowledge-user-token"
    config = SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token=_FAKE_APP,
        bot_token=_FAKE_BOT,
        knowledge_user_token=token,
    )
    rendered = repr(config)
    assert token not in rendered


def test_positive_timeout_accepted() -> None:
    config = SlackConversationChannelIntegrationConfig(api_timeout_seconds=15.0)
    assert config.api_timeout_seconds == 15.0


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValidationError):
        SlackConversationChannelIntegrationConfig(api_timeout_seconds=0)
    with pytest.raises(ValidationError):
        SlackConversationChannelIntegrationConfig(api_timeout_seconds=999)


def test_secret_values_absent_from_repr() -> None:
    config = SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token=_FAKE_APP,
        bot_token=_FAKE_BOT,
    )
    rendered = repr(config)
    assert _FAKE_APP not in rendered
    assert _FAKE_BOT not in rendered


def test_secret_values_absent_from_validation_errors() -> None:
    secret = "not-a-valid-prefix-token-value"
    with pytest.raises(IntegrationConfigurationError) as exc_info:
        SlackConversationChannelIntegrationConfig(app_token=secret)
    message = str(exc_info.value)
    assert secret not in message
    assert "xapp-" in message


def test_public_view_strips_tokens() -> None:
    config = SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token=_FAKE_APP,
        bot_token=_FAKE_BOT,
        knowledge_user_token="xoxp-test-knowledge-user",
        api_timeout_seconds=12.0,
    )
    public = config.public_view()
    assert "app_token" not in public
    assert "bot_token" not in public
    assert "knowledge_user_token" not in public
    assert public["api_timeout_seconds"] == 12.0
