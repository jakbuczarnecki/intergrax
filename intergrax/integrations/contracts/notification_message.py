# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical notification message model for integration catalog contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NotificationMessage(BaseModel):
    channel: str
    subject: str
    body: str
    task_id: str
    tenant_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["NotificationMessage"]
