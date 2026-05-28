# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical notification message model (§18)."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class NotificationMessage(BaseModel):
    channel: str
    subject: str
    body: str
    task_id: str
    tenant_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
