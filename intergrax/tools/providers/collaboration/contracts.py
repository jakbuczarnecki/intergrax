# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class CollaborationSendMailInput(BaseModel):
    user_id: str = Field(..., min_length=1, description="Mailbox owner / sender identity.")
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    to: list[str] = Field(..., min_length=1)


class CollaborationSendMailOutput(BaseModel):
    sent: bool = True
    recipient_count: int = 0


class CollaborationListMessagesInput(BaseModel):
    user_id: str = Field(..., min_length=1)
    folder: str = Field(default="inbox")
    limit: int = Field(default=25, ge=1, le=100)


class CollaborationMailMessageOutput(BaseModel):
    id: str
    subject: str
    body_preview: str = ""
    from_address: str | None = None
    received_at: str | None = None


class CollaborationListMessagesOutput(BaseModel):
    messages: list[CollaborationMailMessageOutput] = Field(default_factory=list)
    total: int = 0


class CollaborationGetMessageInput(BaseModel):
    user_id: str = Field(..., min_length=1)
    message_id: str = Field(..., min_length=1)


class CollaborationGetMessageOutput(BaseModel):
    message: CollaborationMailMessageOutput


class CollaborationListCalendarInput(BaseModel):
    user_id: str = Field(..., min_length=1)
    start: str = Field(..., min_length=1, description="ISO8601 window start.")
    end: str = Field(..., min_length=1, description="ISO8601 window end.")
    limit: int = Field(default=50, ge=1, le=200)


class CollaborationCalendarEventOutput(BaseModel):
    id: str
    subject: str
    start: str
    end: str
    location: str = ""
    organizer: str | None = None


class CollaborationListCalendarOutput(BaseModel):
    events: list[CollaborationCalendarEventOutput] = Field(default_factory=list)
    total: int = 0


class CollaborationGetUserInput(BaseModel):
    user_id: str = Field(..., min_length=1)


class CollaborationUserOutput(BaseModel):
    id: str
    display_name: str
    email: str | None = None


class CollaborationGetUserOutput(BaseModel):
    user: CollaborationUserOutput
