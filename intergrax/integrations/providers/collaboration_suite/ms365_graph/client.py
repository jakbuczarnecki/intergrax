# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph REST client — HTTP client injected from ``opens.py`` only."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from urllib.parse import quote

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.collaboration_suite import (
    CalendarEvent,
    CalendarEventsResult,
    MailListResult,
    MailMessage,
    UserRecord,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import Ms365GraphIntegrationConfig

_MESSAGE_SELECT = "id,subject,bodyPreview,from,receivedDateTime"
_EVENT_SELECT = "id,subject,start,end,location,organizer"
_USER_SELECT = "id,displayName,mail,userPrincipalName"


def _email_from_address(raw: object) -> str | None:
    if not isinstance(raw, dict):
        return None
    email_obj = raw.get("emailAddress")
    if not isinstance(email_obj, dict):
        return None
    address = email_obj.get("address")
    return str(address) if address else None


def _message_from_payload(payload: Mapping[str, Any]) -> MailMessage:
    from_obj = payload.get("from")
    return MailMessage(
        id=str(payload.get("id") or ""),
        subject=str(payload.get("subject") or ""),
        body_preview=str(payload.get("bodyPreview") or ""),
        from_address=_email_from_address(from_obj),
        received_at=str(payload.get("receivedDateTime") or "") or None,
    )


def _event_from_payload(payload: Mapping[str, Any]) -> CalendarEvent:
    start_obj = payload.get("start")
    end_obj = payload.get("end")
    start = start_obj.get("dateTime") if isinstance(start_obj, dict) else ""
    end = end_obj.get("dateTime") if isinstance(end_obj, dict) else ""
    location_obj = payload.get("location")
    location = location_obj.get("displayName") if isinstance(location_obj, dict) else ""
    organizer_obj = payload.get("organizer")
    organizer = _email_from_address(organizer_obj) if isinstance(organizer_obj, dict) else None
    if organizer is None and isinstance(organizer_obj, dict):
        email_obj = organizer_obj.get("emailAddress")
        if isinstance(email_obj, dict) and email_obj.get("name"):
            organizer = str(email_obj.get("name"))
    return CalendarEvent(
        id=str(payload.get("id") or ""),
        subject=str(payload.get("subject") or ""),
        start=str(start or ""),
        end=str(end or ""),
        location=str(location or ""),
        organizer=organizer,
    )


def _user_from_payload(payload: Mapping[str, Any]) -> UserRecord:
    email = payload.get("mail") or payload.get("userPrincipalName")
    return UserRecord(
        id=str(payload.get("id") or ""),
        display_name=str(payload.get("displayName") or ""),
        email=str(email) if email else None,
    )


class GraphRestClient:
    """Minimal Microsoft Graph v1.0 client — sync HTTP via injected client."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        http_client: Any,
    ) -> None:
        if not config.tenant_id:
            raise IntegrationConfigurationError(
                "MS365 tenant_id is required (INTERGRAX_MS365_TENANT_ID)"
            )
        if not config.client_id or not config.client_secret:
            raise IntegrationConfigurationError(
                "MS365 client_id and client_secret are required "
                "(INTERGRAX_MS365_CLIENT_ID, INTERGRAX_MS365_CLIENT_SECRET)"
            )
        self._config = config
        self._http_client = http_client

    @property
    def config(self) -> Ms365GraphIntegrationConfig:
        return self._config

    def get_message(self, user_id: str, message_id: str) -> MailMessage:
        path = f"/users/{quote(user_id, safe='')}/messages/{quote(message_id, safe='')}"
        response = self._http_client.get(path, params={"$select": _MESSAGE_SELECT})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise IntegrationConfigurationError("Unexpected Graph get_message response")
        return _message_from_payload(payload)

    def list_messages(
        self,
        user_id: str,
        *,
        folder: str = "inbox",
        limit: int = 25,
    ) -> MailListResult:
        folder_segment = quote(folder, safe="")
        path = f"/users/{quote(user_id, safe='')}/mailFolders/{folder_segment}/messages"
        response = self._http_client.get(
            path,
            params={
                "$top": max(1, int(limit)),
                "$select": _MESSAGE_SELECT,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise IntegrationConfigurationError("Unexpected Graph list_messages response")
        raw_messages = payload.get("value")
        messages = [
            _message_from_payload(item)
            for item in raw_messages
            if isinstance(item, dict)
        ]
        total = len(messages)
        return MailListResult(messages=messages, total=total)

    def send_mail(
        self,
        user_id: str,
        *,
        subject: str,
        body: str,
        to: Sequence[str],
    ) -> None:
        if not to:
            raise IntegrationConfigurationError("send_mail requires at least one recipient")
        payload = {
            "message": {
                "subject": subject,
                "body": {"contentType": "Text", "content": body},
                "toRecipients": [
                    {"emailAddress": {"address": address}} for address in to
                ],
            },
            "saveToSentItems": True,
        }
        path = f"/users/{quote(user_id, safe='')}/sendMail"
        response = self._http_client.post(path, json=payload)
        response.raise_for_status()

    def list_calendar_events(
        self,
        user_id: str,
        *,
        start: str,
        end: str,
        limit: int = 50,
    ) -> CalendarEventsResult:
        path = f"/users/{quote(user_id, safe='')}/calendar/calendarView"
        response = self._http_client.get(
            path,
            params={
                "startDateTime": start,
                "endDateTime": end,
                "$top": max(1, int(limit)),
                "$select": _EVENT_SELECT,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise IntegrationConfigurationError("Unexpected Graph list_calendar_events response")
        raw_events = payload.get("value")
        events = [
            _event_from_payload(item)
            for item in raw_events
            if isinstance(item, dict)
        ]
        return CalendarEventsResult(events=events, total=len(events))

    def get_user(self, user_id: str) -> UserRecord:
        path = f"/users/{quote(user_id, safe='')}"
        response = self._http_client.get(path, params={"$select": _USER_SELECT})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise IntegrationConfigurationError("Unexpected Graph get_user response")
        return _user_from_payload(payload)
