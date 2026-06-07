# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class CrmGetAccountInput(BaseModel):
    account_id: str = Field(..., min_length=1)


class CrmAccountOutput(BaseModel):
    account_id: str
    name: str = ""
    industry: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class CrmGetAccountOutput(BaseModel):
    account: CrmAccountOutput


class CrmListContactsInput(BaseModel):
    account_id: str = Field(..., min_length=1)
    limit: int = Field(default=50, ge=1, le=500)


class CrmContactOutput(BaseModel):
    contact_id: str
    email: str = ""
    name: str = ""
    account_id: str = ""


class CrmListContactsOutput(BaseModel):
    account_id: str
    contacts: list[CrmContactOutput] = Field(default_factory=list)
    total: int = 0


class CrmListTicketsInput(BaseModel):
    account_id: str = Field(..., min_length=1)
    limit: int = Field(default=50, ge=1, le=500)


class CrmTicketOutput(BaseModel):
    ticket_id: str
    subject: str = ""
    status: str = ""
    account_id: str = ""


class CrmListTicketsOutput(BaseModel):
    account_id: str
    tickets: list[CrmTicketOutput] = Field(default_factory=list)
    total: int = 0
