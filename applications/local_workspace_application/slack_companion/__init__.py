# © Artur Czarnecki. All rights reserved.

"""LKW-owned Slack Ask companion (optional product workflow)."""

from __future__ import annotations

from local_workspace_application.slack_companion.companion import (
    SlackCompanion,
    wire_slack_companion,
)

__all__ = [
    "SlackCompanion",
    "wire_slack_companion",
]
