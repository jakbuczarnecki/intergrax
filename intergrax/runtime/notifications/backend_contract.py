# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lightweight notification backend configuration contract (§18)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

from intergrax.integrations.contracts.base import IntegrationConfigurationError, LabSelectorOptIn

ENV_NOTIFICATION_BACKEND = "INTERGRAX_NOTIFICATION_BACKEND"
ENV_WEBHOOK_URL = "INTERGRAX_WEBHOOK_URL"
ENV_SLACK_WEBHOOK_URL = "INTERGRAX_SLACK_WEBHOOK_URL"
ENV_TEAMS_WEBHOOK_URL = "INTERGRAX_TEAMS_WEBHOOK_URL"


class NotificationBackend(StrEnum):
    LOG = "log"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"


@dataclass(frozen=True)
class NotificationSettings:
    backend: NotificationBackend
    webhook_url: str = ""
    slack_webhook_url: str = ""
    teams_webhook_url: str = ""


def resolve_notification_settings(
    *,
    backend: str | None = None,
    webhook_url: str | None = None,
    slack_webhook_url: str | None = None,
    teams_webhook_url: str | None = None,
    lab_opt_in: LabSelectorOptIn | None = None,
) -> NotificationSettings:
    raw_backend = backend if backend is not None else os.environ.get(ENV_NOTIFICATION_BACKEND)
    if raw_backend is None or not str(raw_backend).strip():
        if lab_opt_in is None:
            raise IntegrationConfigurationError(
                f"{ENV_NOTIFICATION_BACKEND} is unset and no notification backend was selected"
            )
        raw_backend = lab_opt_in.selector
    normalized = str(raw_backend).strip().lower()
    try:
        resolved_backend = NotificationBackend(normalized)
    except ValueError as exc:
        raise IntegrationConfigurationError(f"unknown notification backend {normalized!r}") from exc
    return NotificationSettings(
        backend=resolved_backend,
        webhook_url=(webhook_url or os.environ.get(ENV_WEBHOOK_URL, "")).strip(),
        slack_webhook_url=(slack_webhook_url or os.environ.get(ENV_SLACK_WEBHOOK_URL, "")).strip(),
        teams_webhook_url=(teams_webhook_url or os.environ.get(ENV_TEAMS_WEBHOOK_URL, "")).strip(),
    )
