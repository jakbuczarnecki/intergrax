# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CRM integration contract (Phase M.6 P6)."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class CrmAccount(BaseModel):
    """CRM account/company row."""

    account_id: str
    name: str = ""
    industry: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class CrmContact(BaseModel):
    """CRM contact row."""

    contact_id: str
    email: str = ""
    name: str = ""
    account_id: Optional[str] = None
    metadata: dict[str, str] = Field(default_factory=dict)


class CrmTicket(BaseModel):
    """Support ticket row for agent context."""

    ticket_id: str
    subject: str = ""
    status: str = ""
    account_id: Optional[str] = None
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class CrmBackend(Protocol):
    """Read-only CRM context facade for support harness agents."""

    def get_account(self, account_id: str) -> CrmAccount:
        """Fetch a CRM account by id."""

    def list_contacts(self, *, account_id: str, limit: int = 50) -> Sequence[CrmContact]:
        """List contacts for an account."""

    def list_tickets(self, *, account_id: str, limit: int = 50) -> Sequence[CrmTicket]:
        """List support tickets for an account."""
