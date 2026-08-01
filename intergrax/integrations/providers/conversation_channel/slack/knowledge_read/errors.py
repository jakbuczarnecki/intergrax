# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack conversation knowledge-read error boundaries."""

from __future__ import annotations

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)

_SLUG = "slack"


class SlackConversationReadError(IntegrationDependencyError):
    """Slack Web API knowledge-read dependency failure."""

    def __init__(
        self,
        *,
        slack_error: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        code = slack_error.strip() or "unknown_error"
        super().__init__(
            f"Slack conversation knowledge read failed: {code}",
            integration_name=_SLUG,
        )
        self.slack_error = code
        self.retry_after_seconds = retry_after_seconds


class SlackConversationReadConfigurationError(IntegrationConfigurationError):
    """Invalid Slack conversation knowledge-read request or configuration."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class SlackConversationMessageNotFound(IntegrationDependencyError):
    """Exact Slack message lookup returned no accessible message."""

    def __init__(self) -> None:
        super().__init__(
            "Slack conversation message not found",
            integration_name=_SLUG,
        )


class SlackConversationMessageChanged(IntegrationDependencyError):
    """Slack message revision changed between descriptor and exact read."""

    def __init__(self) -> None:
        super().__init__(
            "Slack conversation message changed during exact read",
            integration_name=_SLUG,
        )


class SlackConversationContentTooLarge(IntegrationConfigurationError):
    """Normalized Slack message content exceeds configured limit."""

    def __init__(self) -> None:
        super().__init__(
            "Slack conversation message exceeds the configured content limit",
        )


__all__ = [
    "SlackConversationContentTooLarge",
    "SlackConversationMessageChanged",
    "SlackConversationMessageNotFound",
    "SlackConversationReadConfigurationError",
    "SlackConversationReadError",
]
