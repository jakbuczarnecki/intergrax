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


class NotifyBatchMessageInput(BaseModel):
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    channel: str = Field(default="default")
    metadata: dict[str, Any] = Field(default_factory=dict)


class NotifySendBatchInput(BaseModel):
    tenant_id: str = Field(default="default")
    task_id: str = Field(default="")
    messages: list[NotifyBatchMessageInput] = Field(..., min_length=1, max_length=50)


class NotifySendBatchOutput(BaseModel):
    sent_count: int = 0
    failed_count: int = 0
    details: list[str] = Field(default_factory=list)


class NotifyScheduleInput(BaseModel):
    tenant_id: str = Field(default="default")
    channel: str = Field(default="default")
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    deliver_at_utc: str = Field(..., min_length=1, description="ISO8601 UTC delivery timestamp.")


class NotifyScheduleOutput(BaseModel):
    scheduled: bool = False
    schedule_id: str = ""
    deliver_at_utc: str = ""
    detail: str = ""


class NotifyListScheduledInput(BaseModel):
    tenant_id: str = Field(default="default")
    limit: int = Field(default=50, ge=1, le=200)
    status: str = Field(default="pending", description="Filter by schedule status (empty = all).")


class NotifyScheduledItemOutput(BaseModel):
    schedule_id: str
    tenant_id: str
    channel: str
    subject: str = ""
    deliver_at_utc: str
    status: str


class NotifyListScheduledOutput(BaseModel):
    used: bool = False
    schedules: list[NotifyScheduledItemOutput] = Field(default_factory=list)
    total: int = 0
    detail: str = ""


class NotifyCancelScheduledInput(BaseModel):
    tenant_id: str = Field(default="default")
    schedule_id: str = Field(..., min_length=1)


class NotifyCancelScheduledOutput(BaseModel):
    cancelled: bool = False
    schedule_id: str = ""
    detail: str = ""
