# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lightweight notification backend configuration contract (§18)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

ENV_NOTIFICATION_BACKEND = "INTERGRAX_NOTIFICATION_BACKEND"
ENV_WEBHOOK_URL = "INTERGRAX_WEBHOOK_URL"
ENV_SLACK_WEBHOOK_URL = "INTERGRAX_SLACK_WEBHOOK_URL"
ENV_TEAMS_WEBHOOK_URL = "INTERGRAX_TEAMS_WEBHOOK_URL"


class NotificationBackend(str, Enum):
    LOG = "log"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"


@dataclass(frozen=True)
class NotificationSettings:
    backend: NotificationBackend = NotificationBackend.LOG
    webhook_url: str = ""
    slack_webhook_url: str = ""
    teams_webhook_url: str = ""


def resolve_notification_settings(
    *,
    backend: str | None = None,
    webhook_url: str | None = None,
    slack_webhook_url: str | None = None,
    teams_webhook_url: str | None = None,
) -> NotificationSettings:
    raw_backend = (
        backend
        or os.environ.get(ENV_NOTIFICATION_BACKEND, NotificationBackend.LOG.value)
    ).strip().lower()
    try:
        resolved_backend = NotificationBackend(raw_backend)
    except ValueError:
        resolved_backend = NotificationBackend.LOG
    return NotificationSettings(
        backend=resolved_backend,
        webhook_url=(webhook_url or os.environ.get(ENV_WEBHOOK_URL, "")).strip(),
        slack_webhook_url=(
            slack_webhook_url or os.environ.get(ENV_SLACK_WEBHOOK_URL, "")
        ).strip(),
        teams_webhook_url=(
            teams_webhook_url or os.environ.get(ENV_TEAMS_WEBHOOK_URL, "")
        ).strip(),
    )
