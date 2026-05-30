# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class NotifySendInput(BaseModel):
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    channel: str = Field(default="default", description="Logical notification channel label.")
    task_id: str = Field(default="", description="Related task id for tracing.")
    tenant_id: str = Field(default="default")
    metadata: dict[str, Any] = Field(default_factory=dict)


class NotifySendOutput(BaseModel):
    sent: bool
    channel: str
    detail: str = ""
