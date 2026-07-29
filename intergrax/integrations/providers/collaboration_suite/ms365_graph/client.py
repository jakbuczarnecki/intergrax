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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    DEFAULT_CALENDAR_ATTACHMENT_MAX_BYTES,
    DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS,
    DEFAULT_DRIVE_CONTENT_MAX_BYTES,
    DEFAULT_MAIL_ATTACHMENT_MAX_BYTES,
    DEFAULT_MAIL_CONTENT_MAX_CHARS,
    MsGraphCalendar,
    MsGraphCalendarAttachment,
    MsGraphCalendarAttachmentPage,
    MsGraphCalendarAttachmentsReader,
    MsGraphCalendarContentReader,
    MsGraphCalendarEventChange,
    MsGraphCalendarEventContent,
    MsGraphCalendarEventDeltaPage,
    MsGraphCalendarEventSnapshotPage,
    MsGraphCalendarEventsReader,
    MsGraphCalendarFileAttachmentContent,
    MsGraphCalendarPage,
    MsGraphCalendarsReader,
    MsGraphCalendarViewWindow,
    MsGraphDriveDeltaPage,
    MsGraphDriveFileContent,
    MsGraphDriveItem,
    MsGraphDriveKnowledgeReader,
    MsGraphDriveContentReader,
    MsGraphDrivePermissionPage,
    MsGraphDrivePermissionsReader,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeTransport,
    MsGraphMailAttachment,
    MsGraphMailAttachmentPage,
    MsGraphMailAttachmentsReader,
    MsGraphMailContentReader,
    MsGraphMailFileAttachmentContent,
    MsGraphMailFolderPage,
    MsGraphMailFoldersReader,
    MsGraphMailMessageChange,
    MsGraphMailMessageContent,
    MsGraphMailMessageDeltaPage,
    MsGraphMailMessagesReader,
    DEFAULT_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES,
    DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    MsGraphTeamsChat,
    MsGraphTeamsChatHostedContent,
    MsGraphTeamsChatHostedContentBytes,
    MsGraphTeamsChatHostedContentPage,
    MsGraphTeamsChatHostedContentReader,
    MsGraphTeamsChatMemberPage,
    MsGraphTeamsChatMembersReader,
    MsGraphTeamsChatMessage,
    MsGraphTeamsChatMessageSnapshotPage,
    MsGraphTeamsChatMessageWindow,
    MsGraphTeamsChatMessagesReader,
    MsGraphTeamsChatPage,
    MsGraphTeamsChatsReader,
)

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
        download_http_client: Any = None,
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
        self._download_http_client = download_http_client
        self._knowledge_transport = MsGraphKnowledgeTransport(
            config=config,
            http_client=http_client,
        )
        self._drive_knowledge_reader = MsGraphDriveKnowledgeReader(
            config=config,
            transport=self._knowledge_transport,
        )
        self._drive_permissions_reader = MsGraphDrivePermissionsReader(
            config=config,
            transport=self._knowledge_transport,
        )
        self._mail_folders_reader = MsGraphMailFoldersReader(
            config=config,
            transport=self._knowledge_transport,
        )
        self._mail_messages_reader = MsGraphMailMessagesReader(
            config=config,
            transport=self._knowledge_transport,
        )
        self._mail_content_reader = MsGraphMailContentReader(
            config=config,
            transport=self._knowledge_transport,
        )
        self._mail_attachments_reader = MsGraphMailAttachmentsReader(
            config=config,
            transport=self._knowledge_transport,
            graph_http_client=http_client,
        )
        self._calendars_reader = MsGraphCalendarsReader(
            config=config,
            transport=self._knowledge_transport,
        )
        self._calendar_events_reader = MsGraphCalendarEventsReader(
            config=config,
            transport=self._knowledge_transport,
        )
        self._calendar_content_reader = MsGraphCalendarContentReader(
            config=config,
            transport=self._knowledge_transport,
        )
        self._calendar_attachments_reader = MsGraphCalendarAttachmentsReader(
            config=config,
            transport=self._knowledge_transport,
            graph_http_client=http_client,
        )
        self._teams_chats_reader = MsGraphTeamsChatsReader(
            config=config,
            transport=self._knowledge_transport,
        )
        self._teams_chat_members_reader = MsGraphTeamsChatMembersReader(
            config=config,
            transport=self._knowledge_transport,
        )
        self._teams_chat_messages_reader = MsGraphTeamsChatMessagesReader(
            config=config,
            transport=self._knowledge_transport,
        )
        self._teams_chat_hosted_content_reader = MsGraphTeamsChatHostedContentReader(
            config=config,
            transport=self._knowledge_transport,
            graph_http_client=http_client,
        )
        self._drive_content_reader: MsGraphDriveContentReader | None = None
        if download_http_client is not None:
            self._drive_content_reader = MsGraphDriveContentReader(
                config=config,
                graph_transport=self._knowledge_transport,
                graph_http_client=http_client,
                download_http_client=download_http_client,
            )

    @property
    def config(self) -> Ms365GraphIntegrationConfig:
        return self._config

    def read_drive_delta_page(
        self,
        *,
        drive_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphDriveDeltaPage:
        return self._drive_knowledge_reader.read_delta_page(
            drive_id=drive_id,
            continuation=continuation,
            limit=limit,
        )

    def read_drive_permissions_page(
        self,
        *,
        item: MsGraphDriveItem,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphDrivePermissionPage:
        return self._drive_permissions_reader.read_permissions_page(
            item=item,
            continuation=continuation,
        )

    def read_mail_folders_page(
        self,
        *,
        mailbox_user_id: str,
        parent_folder_id: str | None = None,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailFolderPage:
        return self._mail_folders_reader.read_folders_page(
            mailbox_user_id=mailbox_user_id,
            parent_folder_id=parent_folder_id,
            continuation=continuation,
            limit=limit,
        )

    def read_mail_messages_delta_page(
        self,
        *,
        mailbox_user_id: str,
        folder_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailMessageDeltaPage:
        return self._mail_messages_reader.read_delta_page(
            mailbox_user_id=mailbox_user_id,
            folder_id=folder_id,
            continuation=continuation,
            limit=limit,
        )

    def read_mail_message_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        max_chars: int = DEFAULT_MAIL_CONTENT_MAX_CHARS,
    ) -> MsGraphMailMessageContent:
        return self._mail_content_reader.read_message_content(
            message=message,
            max_chars=max_chars,
        )

    def read_mail_attachments_page(
        self,
        *,
        message: MsGraphMailMessageChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailAttachmentPage:
        return self._mail_attachments_reader.read_attachments_page(
            message=message,
            continuation=continuation,
            limit=limit,
        )

    def read_mail_file_attachment_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        attachment: MsGraphMailAttachment,
        max_bytes: int = DEFAULT_MAIL_ATTACHMENT_MAX_BYTES,
    ) -> MsGraphMailFileAttachmentContent:
        return self._mail_attachments_reader.read_file_attachment_content(
            message=message,
            attachment=attachment,
            max_bytes=max_bytes,
        )

    def read_calendars_page(
        self,
        *,
        mailbox_user_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarPage:
        return self._calendars_reader.read_calendars_page(
            mailbox_user_id=mailbox_user_id,
            continuation=continuation,
            limit=limit,
        )

    def read_calendar_events_delta_page(
        self,
        *,
        calendar: MsGraphCalendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarEventDeltaPage:
        return self._calendar_events_reader.read_delta_page(
            calendar=calendar,
            window=window,
            continuation=continuation,
            limit=limit,
        )

    def read_calendar_events_snapshot_page(
        self,
        *,
        calendar: MsGraphCalendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarEventSnapshotPage:
        return self._calendar_events_reader.read_snapshot_page(
            calendar=calendar,
            window=window,
            continuation=continuation,
            limit=limit,
        )

    def read_calendar_event_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        max_chars: int = DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS,
    ) -> MsGraphCalendarEventContent:
        return self._calendar_content_reader.read_calendar_event_content(
            event=event,
            max_chars=max_chars,
        )

    def read_calendar_attachments_page(
        self,
        *,
        event: MsGraphCalendarEventChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarAttachmentPage:
        return self._calendar_attachments_reader.read_attachments_page(
            event=event,
            continuation=continuation,
            limit=limit,
        )

    def read_calendar_file_attachment_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        attachment: MsGraphCalendarAttachment,
        max_bytes: int = DEFAULT_CALENDAR_ATTACHMENT_MAX_BYTES,
    ) -> MsGraphCalendarFileAttachmentContent:
        return self._calendar_attachments_reader.read_file_attachment_content(
            event=event,
            attachment=attachment,
            max_bytes=max_bytes,
        )

    def read_teams_chats_page(
        self,
        *,
        mailbox_user_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
    ) -> MsGraphTeamsChatPage:
        return self._teams_chats_reader.read_chats_page(
            mailbox_user_id=mailbox_user_id,
            continuation=continuation,
            limit=limit,
        )

    def read_teams_chat_members_page(
        self,
        *,
        chat: MsGraphTeamsChat,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphTeamsChatMemberPage:
        return self._teams_chat_members_reader.read_members_page(
            chat=chat,
            continuation=continuation,
        )

    def read_teams_chat_messages_snapshot_page(
        self,
        *,
        chat: MsGraphTeamsChat,
        window: MsGraphTeamsChatMessageWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
        max_chars_per_message: int = DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChatMessageSnapshotPage:
        return self._teams_chat_messages_reader.read_messages_snapshot_page(
            chat=chat,
            window=window,
            continuation=continuation,
            limit=limit,
            max_chars_per_message=max_chars_per_message,
        )

    def read_teams_chat_hosted_contents_page(
        self,
        *,
        message: MsGraphTeamsChatMessage,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphTeamsChatHostedContentPage:
        return self._teams_chat_hosted_content_reader.read_hosted_contents_page(
            message=message,
            continuation=continuation,
        )

    def read_teams_chat_hosted_content_bytes(
        self,
        *,
        message: MsGraphTeamsChatMessage,
        hosted_content: MsGraphTeamsChatHostedContent,
        max_bytes: int = DEFAULT_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES,
    ) -> MsGraphTeamsChatHostedContentBytes:
        return self._teams_chat_hosted_content_reader.read_hosted_content_bytes(
            message=message,
            hosted_content=hosted_content,
            max_bytes=max_bytes,
        )

    def read_drive_file_content(
        self,
        *,
        item: MsGraphDriveItem,
        max_bytes: int = DEFAULT_DRIVE_CONTENT_MAX_BYTES,
    ) -> MsGraphDriveFileContent:
        if self._drive_content_reader is None:
            raise IntegrationConfigurationError(
                "Microsoft Graph Drive download client is not configured"
            )
        return self._drive_content_reader.read_file_content(item=item, max_bytes=max_bytes)

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

    def reply_message(self, user_id: str, message_id: str, *, body: str) -> None:
        path = f"/users/{quote(user_id, safe='')}/messages/{quote(message_id, safe='')}/reply"
        response = self._http_client.post(path, json={"comment": body})
        response.raise_for_status()

    def create_event(
        self,
        user_id: str,
        *,
        subject: str,
        start: str,
        end: str,
        location: str = "",
        attendees: Sequence[str] = (),
    ) -> CalendarEvent:
        payload: dict[str, Any] = {
            "subject": subject,
            "start": {"dateTime": start, "timeZone": "UTC"},
            "end": {"dateTime": end, "timeZone": "UTC"},
        }
        if location:
            payload["location"] = {"displayName": location}
        if attendees:
            payload["attendees"] = [
                {"emailAddress": {"address": address}, "type": "required"} for address in attendees
            ]
        path = f"/users/{quote(user_id, safe='')}/calendar/events"
        response = self._http_client.post(path, json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise IntegrationConfigurationError("Unexpected Graph create_event response")
        return _event_from_payload(body)
