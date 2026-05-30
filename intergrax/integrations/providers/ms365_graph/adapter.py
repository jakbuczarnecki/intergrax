# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MS365 Graph collaboration suite adapter — ``CollaborationSuite`` facade (no HTTP here)."""

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.collaboration_suite import (
    CalendarEventsResult,
    MailListResult,
    MailMessage,
    UserRecord,
)
from intergrax.integrations.providers.ms365_graph.client import GraphRestClient


class Ms365GraphCollaborationSuite:
    """
    Catalog facade over ``GraphRestClient``.

    Instantiate via ``create_ms365_graph_collaboration_suite()`` — not from agent code.
    """

    def __init__(self, client: GraphRestClient) -> None:
        self._client = client

    @property
    def rest_client(self) -> GraphRestClient:
        return self._client

    def get_message(self, user_id: str, message_id: str) -> MailMessage:
        return self._client.get_message(user_id, message_id)

    def list_messages(
        self,
        user_id: str,
        *,
        folder: str = "inbox",
        limit: int = 25,
    ) -> MailListResult:
        return self._client.list_messages(user_id, folder=folder, limit=limit)

    def send_mail(
        self,
        user_id: str,
        *,
        subject: str,
        body: str,
        to: Sequence[str],
    ) -> None:
        self._client.send_mail(user_id, subject=subject, body=body, to=to)

    def list_calendar_events(
        self,
        user_id: str,
        *,
        start: str,
        end: str,
        limit: int = 50,
    ) -> CalendarEventsResult:
        return self._client.list_calendar_events(
            user_id,
            start=start,
            end=end,
            limit=limit,
        )

    def get_user(self, user_id: str) -> UserRecord:
        return self._client.get_user(user_id)
