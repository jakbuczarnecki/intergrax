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
