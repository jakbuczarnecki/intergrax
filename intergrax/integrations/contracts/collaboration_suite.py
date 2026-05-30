# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Collaboration suite integration contract (§7.1.2, Phase M.6)."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class MailMessage(BaseModel):
    """Normalized mail message for agent tools and Tier-3 composition."""

    id: str
    subject: str
    body_preview: str = ""
    from_address: Optional[str] = None
    received_at: Optional[str] = None


class MailListResult(BaseModel):
    messages: Sequence[MailMessage] = Field(default_factory=list)
    total: int = 0


class CalendarEvent(BaseModel):
    id: str
    subject: str
    start: str
    end: str
    location: str = ""
    organizer: Optional[str] = None


class CalendarEventsResult(BaseModel):
    events: Sequence[CalendarEvent] = Field(default_factory=list)
    total: int = 0


class UserRecord(BaseModel):
    id: str
    display_name: str
    email: Optional[str] = None


@runtime_checkable
class CollaborationSuite(Protocol):
    """
    Backend-agnostic mail, calendar, and directory facade.

    Implementations: ms365_graph, google_workspace, …
    """

    def get_message(self, user_id: str, message_id: str) -> MailMessage:
        """Fetch a single mail message."""

    def list_messages(
        self,
        user_id: str,
        *,
        folder: str = "inbox",
        limit: int = 25,
    ) -> MailListResult:
        """List messages in a mail folder."""

    def send_mail(
        self,
        user_id: str,
        *,
        subject: str,
        body: str,
        to: Sequence[str],
    ) -> None:
        """Send mail on behalf of ``user_id``."""

    def list_calendar_events(
        self,
        user_id: str,
        *,
        start: str,
        end: str,
        limit: int = 50,
    ) -> CalendarEventsResult:
        """List calendar events in an ISO8601 window."""

    def get_user(self, user_id: str) -> UserRecord:
        """Resolve a directory user (id, UPN, or mail alias)."""
