# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack conversation-channel vendor configuration."""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field, SecretStr, field_validator

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

_DEFAULT_API_TIMEOUT_SECONDS = 30.0
_MIN_API_TIMEOUT_SECONDS = 1.0
_MAX_API_TIMEOUT_SECONDS = 120.0

ENV_APP_TOKEN = "INTERGRAX_SLACK_APP_TOKEN"
ENV_BOT_TOKEN = "INTERGRAX_SLACK_BOT_TOKEN"
ENV_KNOWLEDGE_USER_TOKEN = "INTERGRAX_SLACK_KNOWLEDGE_USER_TOKEN"
ENV_API_TIMEOUT = "INTERGRAX_SLACK_API_TIMEOUT_SECONDS"
ENV_ENABLED = "INTERGRAX_SLACK_CONVERSATION_ENABLED"


def _secret_or_none(value: str | SecretStr | None) -> SecretStr | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        raw = value.get_secret_value()
    else:
        raw = value
    if raw is None:
        return None
    normalized = str(raw).strip()
    if not normalized:
        return None
    return SecretStr(normalized)


def _assert_token_prefixes(
    *,
    app_token: SecretStr | None,
    bot_token: SecretStr | None,
    knowledge_user_token: SecretStr | None = None,
) -> None:
    """Validate token prefixes without echoing secret values into exceptions."""
    if app_token is not None and not app_token.get_secret_value().startswith("xapp-"):
        raise IntegrationConfigurationError("app_token must begin with 'xapp-'")
    if bot_token is not None and not bot_token.get_secret_value().startswith("xoxb-"):
        raise IntegrationConfigurationError("bot_token must begin with 'xoxb-'")
    if knowledge_user_token is not None and not knowledge_user_token.get_secret_value().startswith(
        "xoxp-"
    ):
        raise IntegrationConfigurationError("knowledge_user_token must begin with 'xoxp-'")


class SlackConversationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed Slack vendor config for conversation-channel Socket Mode + Web API."""

    app_token: SecretStr | None = Field(default=None, repr=False)
    bot_token: SecretStr | None = Field(default=None, repr=False)
    knowledge_user_token: SecretStr | None = Field(default=None, repr=False)
    api_timeout_seconds: float = Field(default=_DEFAULT_API_TIMEOUT_SECONDS)

    @field_validator("app_token", "bot_token", "knowledge_user_token", mode="before")
    @classmethod
    def _normalize_optional_secret(cls, value: Any) -> SecretStr | None:
        return _secret_or_none(value)

    @field_validator("api_timeout_seconds")
    @classmethod
    def _validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("api_timeout_seconds must be positive")
        if value < _MIN_API_TIMEOUT_SECONDS or value > _MAX_API_TIMEOUT_SECONDS:
            raise ValueError(
                f"api_timeout_seconds must be between {_MIN_API_TIMEOUT_SECONDS} "
                f"and {_MAX_API_TIMEOUT_SECONDS}"
            )
        return float(value)

    def model_post_init(self, __context: Any) -> None:
        _assert_token_prefixes(
            app_token=self.app_token,
            bot_token=self.bot_token,
            knowledge_user_token=self.knowledge_user_token,
        )

    def validate_for_runtime(self) -> None:
        """Validate tokens for real Socket Mode / Web API construction."""
        if not self.enabled:
            raise IntegrationConfigurationError(
                "Slack conversation runtime requires enabled=True",
            )
        self.require_runtime_tokens()

    @classmethod
    def from_env(
        cls,
        *,
        enabled: bool | None = None,
    ) -> SlackConversationChannelIntegrationConfig:
        """Build config from ``INTERGRAX_SLACK_*`` environment variables."""
        env_enabled = os.environ.get(ENV_ENABLED, "").strip().lower()
        resolved_enabled = (
            enabled
            if enabled is not None
            else env_enabled in {"1", "true", "yes", "on"}
        )
        timeout_raw = os.environ.get(ENV_API_TIMEOUT, "").strip()
        timeout = float(timeout_raw) if timeout_raw else _DEFAULT_API_TIMEOUT_SECONDS
        try:
            return cls(
                enabled=resolved_enabled,
                app_token=os.environ.get(ENV_APP_TOKEN),
                bot_token=os.environ.get(ENV_BOT_TOKEN),
                knowledge_user_token=os.environ.get(ENV_KNOWLEDGE_USER_TOKEN),
                api_timeout_seconds=timeout,
            )
        except IntegrationConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize to configuration error
            raise IntegrationConfigurationError(
                "Invalid Slack conversation channel configuration "
                "(tokens redacted; check prefixes, presence, and timeout bounds)"
            ) from exc

    def require_runtime_tokens(self) -> tuple[str, str]:
        """Return validated app/bot token strings for runtime construction."""
        if not self.enabled:
            raise IntegrationConfigurationError(
                "Slack conversation runtime requires enabled=True",
            )
        if self.app_token is None or self.bot_token is None:
            raise IntegrationConfigurationError(
                "Slack conversation runtime requires app_token and bot_token",
            )
        app = self.app_token.get_secret_value().strip()
        bot = self.bot_token.get_secret_value().strip()
        if not app or not bot:
            raise IntegrationConfigurationError(
                "Slack conversation runtime requires non-blank app_token and bot_token",
            )
        _assert_token_prefixes(
            app_token=self.app_token,
            bot_token=self.bot_token,
            knowledge_user_token=self.knowledge_user_token,
        )
        return app, bot

    def knowledge_user_token_value(self) -> str | None:
        """Return validated knowledge user token string or None when not configured."""
        if self.knowledge_user_token is None:
            return None
        token = self.knowledge_user_token.get_secret_value().strip()
        return token or None


__all__ = [
    "ENV_API_TIMEOUT",
    "ENV_APP_TOKEN",
    "ENV_BOT_TOKEN",
    "ENV_ENABLED",
    "ENV_KNOWLEDGE_USER_TOKEN",
    "SlackConversationChannelIntegrationConfig",
]
