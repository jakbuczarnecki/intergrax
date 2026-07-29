# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MS365 Graph collaboration suite adapter — ``CollaborationSuite`` facade (no HTTP here)."""

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.collaboration_suite import (
    CalendarEvent,
    CalendarEventsResult,
    MailListResult,
    MailMessage,
    UserRecord,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.client import GraphRestClient
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    DEFAULT_CALENDAR_ATTACHMENT_MAX_BYTES,
    DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS,
    DEFAULT_DRIVE_CONTENT_MAX_BYTES,
    DEFAULT_MAIL_ATTACHMENT_MAX_BYTES,
    DEFAULT_MAIL_CONTENT_MAX_CHARS,
    MsGraphCalendar,
    MsGraphCalendarAttachment,
    MsGraphCalendarAttachmentPage,
    MsGraphCalendarEventChange,
    MsGraphCalendarEventContent,
    MsGraphCalendarEventDeltaPage,
    MsGraphCalendarEventSnapshotPage,
    MsGraphCalendarFileAttachmentContent,
    MsGraphCalendarPage,
    MsGraphCalendarViewWindow,
    MsGraphDriveDeltaPage,
    MsGraphDriveFileContent,
    MsGraphDriveItem,
    MsGraphDrivePermissionPage,
    MsGraphKnowledgeContinuation,
    MsGraphMailAttachment,
    MsGraphMailAttachmentPage,
    MsGraphMailFileAttachmentContent,
    MsGraphMailFolderPage,
    MsGraphMailMessageChange,
    MsGraphMailMessageContent,
    MsGraphMailMessageDeltaPage,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    MsGraphTeamsChat,
    MsGraphTeamsChatPage,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_members import (
    MsGraphTeamsChatMemberPage,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
    DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    MsGraphTeamsChatMessage,
    MsGraphTeamsChatMessageSnapshotPage,
    MsGraphTeamsChatMessageWindow,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_hosted_content import (
    DEFAULT_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES,
    MsGraphTeamsChatHostedContent,
    MsGraphTeamsChatHostedContentBytes,
    MsGraphTeamsChatHostedContentPage,
)


class _Ms365GraphCollaborationSuite:
    """
    Catalog facade over ``GraphRestClient``.

    Instantiate via ``create_ms365_graph_collaboration_suite()`` — not from agent code.
    """

    def __init__(self, client: GraphRestClient) -> None:
        self._client = client

    @property
    def rest_client(self) -> GraphRestClient:
        return self._client

    def read_drive_delta_page(
        self,
        *,
        drive_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphDriveDeltaPage:
        return self._client.read_drive_delta_page(
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
        return self._client.read_drive_permissions_page(item=item, continuation=continuation)

    def read_mail_folders_page(
        self,
        *,
        mailbox_user_id: str,
        parent_folder_id: str | None = None,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailFolderPage:
        return self._client.read_mail_folders_page(
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
        return self._client.read_mail_messages_delta_page(
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
        return self._client.read_mail_message_content(message=message, max_chars=max_chars)

    def read_mail_attachments_page(
        self,
        *,
        message: MsGraphMailMessageChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailAttachmentPage:
        return self._client.read_mail_attachments_page(
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
        return self._client.read_mail_file_attachment_content(
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
        return self._client.read_calendars_page(
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
        return self._client.read_calendar_events_delta_page(
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
        return self._client.read_calendar_events_snapshot_page(
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
        return self._client.read_calendar_event_content(event=event, max_chars=max_chars)

    def read_calendar_attachments_page(
        self,
        *,
        event: MsGraphCalendarEventChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarAttachmentPage:
        return self._client.read_calendar_attachments_page(
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
        return self._client.read_calendar_file_attachment_content(
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
        return self._client.read_teams_chats_page(
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
        return self._client.read_teams_chat_members_page(
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
        return self._client.read_teams_chat_messages_snapshot_page(
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
        return self._client.read_teams_chat_hosted_contents_page(
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
        return self._client.read_teams_chat_hosted_content_bytes(
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
        return self._client.read_drive_file_content(item=item, max_bytes=max_bytes)

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

    def reply_message(self, user_id: str, message_id: str, *, body: str) -> None:
        self._client.reply_message(user_id, message_id, body=body)

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
        return self._client.create_event(
            user_id,
            subject=subject,
            start=start,
            end=end,
            location=location,
            attendees=attendees,
        )
