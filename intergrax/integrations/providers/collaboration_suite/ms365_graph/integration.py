# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ms365 Graph collaboration suite integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.adapter import (
    _Ms365GraphCollaborationSuite,
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
    MsGraphCalendarAttachmentsReadClient,
    MsGraphCalendarContentReadClient,
    MsGraphCalendarEventChange,
    MsGraphCalendarEventContent,
    MsGraphCalendarEventDeltaPage,
    MsGraphCalendarEventSnapshotPage,
    MsGraphCalendarEventSnapshotsReadClient,
    MsGraphCalendarEventsReadClient,
    MsGraphCalendarFileAttachmentContent,
    MsGraphCalendarPage,
    MsGraphCalendarsReadClient,
    MsGraphCalendarViewWindow,
    MsGraphDriveContentReadClient,
    MsGraphDriveDeltaPage,
    MsGraphDriveFileContent,
    MsGraphDriveItem,
    MsGraphDriveKnowledgeReadClient,
    MsGraphDrivePermissionPage,
    MsGraphDrivePermissionsReadClient,
    MsGraphKnowledgeContinuation,
    MsGraphMailAttachment,
    MsGraphMailAttachmentPage,
    MsGraphMailAttachmentsReadClient,
    MsGraphMailContentReadClient,
    MsGraphMailFileAttachmentContent,
    MsGraphMailFolderPage,
    MsGraphMailFoldersReadClient,
    MsGraphMailMessageChange,
    MsGraphMailMessageContent,
    MsGraphMailMessageDeltaPage,
    MsGraphMailMessagesReadClient,
    validate_msgraph_drive_permission_page,
    validate_msgraph_calendar_attachment_page,
    validate_msgraph_calendar_event_content,
    validate_msgraph_calendar_event_delta_page,
    validate_msgraph_calendar_event_snapshot_page,
    validate_msgraph_calendar_file_attachment_content,
    validate_msgraph_calendar_page,
    validate_msgraph_mail_attachment_page,
    validate_msgraph_mail_file_attachment_content,
    validate_msgraph_mail_folder_page,
    validate_msgraph_mail_message_content,
    validate_msgraph_mail_message_delta_page,
    DEFAULT_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES,
    DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    ABSOLUTE_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES,
    ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    MsGraphTeamsChat,
    MsGraphTeamsChatHostedContent,
    MsGraphTeamsChatHostedContentBytes,
    MsGraphTeamsChatHostedContentPage,
    MsGraphTeamsChatHostedContentReadClient,
    MsGraphTeamsChatMemberPage,
    MsGraphTeamsChatMembersReadClient,
    MsGraphTeamsChatMessage,
    MsGraphTeamsChatMessageSnapshotPage,
    MsGraphTeamsChatMessageWindow,
    MsGraphTeamsChatMessagesReadClient,
    MsGraphTeamsChatMessageReference,
    MsGraphTeamsChatReference,
    MsGraphTeamsChatReferencePagingReadClient,
    MsGraphTeamsChatContentReadClient,
    MsGraphTeamsChatPage,
    MsGraphTeamsChatsReadClient,
    validate_msgraph_teams_chat,
    validate_msgraph_teams_chat_hosted_content,
    validate_msgraph_teams_chat_hosted_content_bytes,
    validate_msgraph_teams_chat_hosted_content_page,
    validate_msgraph_teams_chat_member_page,
    validate_msgraph_teams_chat_message,
    validate_msgraph_teams_chat_message_content,
    validate_msgraph_teams_chat_message_reference,
    validate_msgraph_teams_chat_message_snapshot_page,
    validate_msgraph_teams_chat_reference,
    validate_msgraph_teams_chat_page,
    validate_msgraph_teams_chats_continuation,
    validate_msgraph_teams_chat_members_continuation,
    validate_msgraph_teams_chat_messages_continuation,
    validate_msgraph_teams_chat_hosted_contents_continuation,
    DEFAULT_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES,
    DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    ABSOLUTE_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES,
    ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    MsGraphTeamsChannel,
    MsGraphTeamsChannelContentReadClient,
    MsGraphTeamsChannelHostedContent,
    MsGraphTeamsChannelHostedContentBytes,
    MsGraphTeamsChannelHostedContentPage,
    MsGraphTeamsChannelHostedContentReadClient,
    MsGraphTeamsChannelMemberPage,
    MsGraphTeamsChannelMembersReadClient,
    MsGraphTeamsChannelMessage,
    MsGraphTeamsChannelMessageReference,
    MsGraphTeamsChannelReference,
    MsGraphTeamsChannelRootMessageReference,
    MsGraphTeamsChannelReplyPage,
    MsGraphTeamsChannelRootMessagePage,
    MsGraphTeamsChannelMessagesReadClient,
    MsGraphTeamsChannelReferencePagingReadClient,
    MsGraphTeamsChannelPage,
    MsGraphTeamsChannelsReadClient,
    validate_msgraph_teams_channel,
    validate_msgraph_teams_channel_hosted_content,
    validate_msgraph_teams_channel_hosted_content_bytes,
    validate_msgraph_teams_channel_hosted_content_page,
    validate_msgraph_teams_channel_member_page,
    validate_msgraph_teams_channel_message,
    validate_msgraph_teams_channel_message_content,
    validate_msgraph_teams_channel_message_reference,
    validate_msgraph_teams_channel_reference,
    validate_msgraph_teams_channel_root_message_reference,
    validate_msgraph_teams_channel_reply_page,
    validate_msgraph_teams_channel_root_message_page,
    validate_msgraph_teams_channel_page,
    validate_msgraph_teams_channels_continuation,
    validate_msgraph_teams_channel_members_continuation,
    validate_msgraph_teams_channel_root_messages_continuation,
    validate_msgraph_teams_channel_replies_continuation,
    validate_msgraph_teams_channel_hosted_contents_continuation,
)
from intergrax.runtime.integrations.categories.collaboration import CollaborationSuiteIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID = "ms365_graph"
_INVALID_HOSTED_CONTENT_REQUEST = "invalid Microsoft Graph Teams hosted content request"
_INVALID_TEAMS_CHANNEL_CONTENT_REQUEST = "invalid Microsoft Graph Teams channel message content request"
_INVALID_TEAMS_CHAT_CONTENT_REQUEST = "invalid Microsoft Graph Teams chat message content request"


class Ms365GraphCollaborationSuiteIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Ms365 Graph collaboration suite integration."""

    pass


Ms365GraphCollaborationSuiteClient = CollaborationSuite

class Ms365GraphCollaborationSuiteIntegration(CollaborationSuiteIntegrationContract):
    """
    Single public Ms365 Graph collaboration suite entrypoint.

    Legacy catalog factory (create_ms365_graph_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: Ms365GraphCollaborationSuiteIntegrationConfig = Ms365GraphCollaborationSuiteIntegrationConfig()
    _client: Ms365GraphCollaborationSuiteClient | None = PrivateAttr(default=None)
    

    def read_drive_delta_page(
        self,
        *,
        drive_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphDriveDeltaPage:
        return self._require_drive_client().read_drive_delta_page(
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
        result = self._require_drive_permissions_client().read_drive_permissions_page(
            item=item,
            continuation=continuation,
        )
        return validate_msgraph_drive_permission_page(result)

    def read_mail_folders_page(
        self,
        *,
        mailbox_user_id: str,
        parent_folder_id: str | None = None,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailFolderPage:
        result = self._require_mail_folders_client().read_mail_folders_page(
            mailbox_user_id=mailbox_user_id,
            parent_folder_id=parent_folder_id,
            continuation=continuation,
            limit=limit,
        )
        return validate_msgraph_mail_folder_page(result)

    def read_mail_messages_delta_page(
        self,
        *,
        mailbox_user_id: str,
        folder_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailMessageDeltaPage:
        result = self._require_mail_messages_client().read_mail_messages_delta_page(
            mailbox_user_id=mailbox_user_id,
            folder_id=folder_id,
            continuation=continuation,
            limit=limit,
        )
        graph_base_url = self._graph_base_url_for_mail_messages_validation()
        return validate_msgraph_mail_message_delta_page(
            result,
            mailbox_user_id=mailbox_user_id,
            folder_id=folder_id,
            graph_base_url=graph_base_url,
        )

    def read_mail_message_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        max_chars: int = DEFAULT_MAIL_CONTENT_MAX_CHARS,
    ) -> MsGraphMailMessageContent:
        result = self._require_mail_content_client().read_mail_message_content(
            message=message,
            max_chars=max_chars,
        )
        return validate_msgraph_mail_message_content(
            result,
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
        result = self._require_mail_attachments_client().read_mail_attachments_page(
            message=message,
            continuation=continuation,
            limit=limit,
        )
        graph_base_url = self._graph_base_url_for_mail_attachments_validation()
        return validate_msgraph_mail_attachment_page(
            result,
            message=message,
            graph_base_url=graph_base_url,
        )

    def read_mail_file_attachment_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        attachment: MsGraphMailAttachment,
        max_bytes: int = DEFAULT_MAIL_ATTACHMENT_MAX_BYTES,
    ) -> MsGraphMailFileAttachmentContent:
        result = self._require_mail_attachments_client().read_mail_file_attachment_content(
            message=message,
            attachment=attachment,
            max_bytes=max_bytes,
        )
        return validate_msgraph_mail_file_attachment_content(
            result,
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
        result = self._require_calendars_client().read_calendars_page(
            mailbox_user_id=mailbox_user_id,
            continuation=continuation,
            limit=limit,
        )
        graph_base_url = self._graph_base_url_for_calendar_validation()
        return validate_msgraph_calendar_page(
            result,
            mailbox_user_id=mailbox_user_id,
            graph_base_url=graph_base_url,
        )

    def read_calendar_events_delta_page(
        self,
        *,
        calendar: MsGraphCalendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarEventDeltaPage:
        result = self._require_calendar_events_client().read_calendar_events_delta_page(
            calendar=calendar,
            window=window,
            continuation=continuation,
            limit=limit,
        )
        graph_base_url = self._graph_base_url_for_calendar_validation()
        return validate_msgraph_calendar_event_delta_page(
            result,
            calendar=calendar,
            window=window,
            graph_base_url=graph_base_url,
        )

    def read_calendar_events_snapshot_page(
        self,
        *,
        calendar: MsGraphCalendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarEventSnapshotPage:
        result = self._require_calendar_event_snapshots_client().read_calendar_events_snapshot_page(
            calendar=calendar,
            window=window,
            continuation=continuation,
            limit=limit,
        )
        graph_base_url = self._graph_base_url_for_calendar_validation()
        return validate_msgraph_calendar_event_snapshot_page(
            result,
            calendar=calendar,
            window=window,
            graph_base_url=graph_base_url,
        )

    def read_calendar_event_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        max_chars: int = DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS,
    ) -> MsGraphCalendarEventContent:
        result = self._require_calendar_content_client().read_calendar_event_content(
            event=event,
            max_chars=max_chars,
        )
        return validate_msgraph_calendar_event_content(
            result,
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
        result = self._require_calendar_attachments_client().read_calendar_attachments_page(
            event=event,
            continuation=continuation,
            limit=limit,
        )
        graph_base_url = self._graph_base_url_for_calendar_validation()
        return validate_msgraph_calendar_attachment_page(
            result,
            event=event,
            graph_base_url=graph_base_url,
        )

    def read_calendar_file_attachment_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        attachment: MsGraphCalendarAttachment,
        max_bytes: int = DEFAULT_CALENDAR_ATTACHMENT_MAX_BYTES,
    ) -> MsGraphCalendarFileAttachmentContent:
        result = self._require_calendar_attachments_client().read_calendar_file_attachment_content(
            event=event,
            attachment=attachment,
            max_bytes=max_bytes,
        )
        return validate_msgraph_calendar_file_attachment_content(
            result,
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
        try:
            from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
                validate_msgraph_mailbox_user_id,
            )

            validate_msgraph_mailbox_user_id(mailbox_user_id)
        except ValueError:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams chats request"
            ) from None
        if type(limit) is not int or limit < 1 or limit > 50:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams chats request"
            ) from None
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_chat_validation()
            validated_continuation = validate_msgraph_teams_chats_continuation(
                continuation,
                mailbox_user_id=mailbox_user_id,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_chats_client().read_teams_chats_page(
            mailbox_user_id=mailbox_user_id,
            continuation=validated_continuation,
            limit=limit,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_chat_validation()
        return validate_msgraph_teams_chat_page(
            result,
            mailbox_user_id=mailbox_user_id,
            graph_base_url=graph_base_url,
        )

    def read_teams_chat_members_page(
        self,
        *,
        chat: MsGraphTeamsChat,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphTeamsChatMemberPage:
        validated_chat = validate_msgraph_teams_chat(chat)
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_chat_validation()
            validated_continuation = validate_msgraph_teams_chat_members_continuation(
                continuation,
                mailbox_user_id=validated_chat.mailbox_user_id,
                chat_id=validated_chat.remote_id,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_chat_members_client().read_teams_chat_members_page(
            chat=validated_chat,
            continuation=validated_continuation,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_chat_validation()
        return validate_msgraph_teams_chat_member_page(
            result,
            chat=validated_chat,
            graph_base_url=graph_base_url,
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
        validated_chat = validate_msgraph_teams_chat(chat)
        try:
            validated_window = MsGraphTeamsChatMessageWindow.model_validate(
                window.model_dump(mode="python")
            )
        except (ValueError, TypeError, AttributeError):
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams chat messages request"
            ) from None
        if type(limit) is not int or limit < 1 or limit > 50:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams chat messages request"
            ) from None
        try:
            from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
                _validate_message_max_chars,
            )

            _validate_message_max_chars(max_chars_per_message)
        except ValueError:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams chat messages request"
            ) from None
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_chat_validation()
            validated_continuation = validate_msgraph_teams_chat_messages_continuation(
                continuation,
                mailbox_user_id=validated_chat.mailbox_user_id,
                chat_id=validated_chat.remote_id,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_chat_messages_client().read_teams_chat_messages_snapshot_page(
            chat=validated_chat,
            window=validated_window,
            continuation=validated_continuation,
            limit=limit,
            max_chars_per_message=max_chars_per_message,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_chat_validation()
        return validate_msgraph_teams_chat_message_snapshot_page(
            result,
            chat=validated_chat,
            window=validated_window,
            graph_base_url=graph_base_url,
            max_chars_per_message=max_chars_per_message,
        )

    def read_teams_chat_messages_snapshot_page_by_reference(
        self,
        *,
        chat: MsGraphTeamsChatReference,
        window: MsGraphTeamsChatMessageWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
        max_chars_per_message: int = DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChatMessageSnapshotPage:
        try:
            validated_chat = validate_msgraph_teams_chat_reference(chat)
        except ValueError:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams chat messages request"
            ) from None
        try:
            validated_window = MsGraphTeamsChatMessageWindow.model_validate(
                window.model_dump(mode="python")
            )
        except (ValueError, TypeError, AttributeError):
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams chat messages request"
            ) from None
        if type(limit) is not int or limit < 1 or limit > 50:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams chat messages request"
            ) from None
        try:
            from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
                _validate_message_max_chars,
            )

            _validate_message_max_chars(max_chars_per_message)
        except ValueError:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams chat messages request"
            ) from None
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_chat_validation()
            validated_continuation = validate_msgraph_teams_chat_messages_continuation(
                continuation,
                mailbox_user_id=validated_chat.mailbox_user_id,
                chat_id=validated_chat.chat_remote_id,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_chat_reference_paging_client().read_teams_chat_messages_snapshot_page_by_reference(
            chat=validated_chat,
            window=validated_window,
            continuation=validated_continuation,
            limit=limit,
            max_chars_per_message=max_chars_per_message,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_chat_validation()
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
            validate_msgraph_teams_chat_message_snapshot_page_by_reference,
        )

        return validate_msgraph_teams_chat_message_snapshot_page_by_reference(
            result,
            chat=validated_chat,
            window=validated_window,
            graph_base_url=graph_base_url,
            max_chars_per_message=max_chars_per_message,
        )

    def read_teams_chat_message_content(
        self,
        *,
        message: MsGraphTeamsChatMessageReference,
        max_chars: int = DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChatMessage:
        validated_reference = validate_msgraph_teams_chat_message_reference(message)
        if (
            type(max_chars) is not int
            or max_chars < 1
            or max_chars > ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS
        ):
            raise IntegrationConfigurationError(_INVALID_TEAMS_CHAT_CONTENT_REQUEST) from None
        result = self._require_teams_chat_content_client().read_teams_chat_message_content(
            message=validated_reference,
            max_chars=max_chars,
        )
        return validate_msgraph_teams_chat_message_content(
            result,
            reference=validated_reference,
            max_chars=max_chars,
        )

    def read_teams_chat_hosted_contents_page(
        self,
        *,
        message: MsGraphTeamsChatMessage,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphTeamsChatHostedContentPage:
        validated_message = validate_msgraph_teams_chat_message(message)
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
            MsGraphTeamsChatMessageState,
        )

        if validated_message.state is not MsGraphTeamsChatMessageState.ACTIVE:
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST) from None
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_chat_validation()
            validated_continuation = validate_msgraph_teams_chat_hosted_contents_continuation(
                continuation,
                mailbox_user_id=validated_message.mailbox_user_id,
                chat_id=validated_message.chat_remote_id,
                message_id=validated_message.remote_id,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_chat_hosted_content_client().read_teams_chat_hosted_contents_page(
            message=validated_message,
            continuation=validated_continuation,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_chat_validation()
        return validate_msgraph_teams_chat_hosted_content_page(
            result,
            message=validated_message,
            graph_base_url=graph_base_url,
        )

    def read_teams_chat_hosted_content_bytes(
        self,
        *,
        message: MsGraphTeamsChatMessage,
        hosted_content: MsGraphTeamsChatHostedContent,
        max_bytes: int = DEFAULT_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES,
    ) -> MsGraphTeamsChatHostedContentBytes:
        validated_message = validate_msgraph_teams_chat_message(message)
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
            MsGraphTeamsChatMessageState,
        )

        if validated_message.state is not MsGraphTeamsChatMessageState.ACTIVE:
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST) from None
        validated_hosted = validate_msgraph_teams_chat_hosted_content(hosted_content)
        if (
            validated_hosted.mailbox_user_id != validated_message.mailbox_user_id
            or validated_hosted.chat_remote_id != validated_message.chat_remote_id
            or validated_hosted.message_remote_id != validated_message.remote_id
            or validated_hosted.message_revision != validated_message.revision
        ):
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST) from None
        if (
            type(max_bytes) is not int
            or max_bytes < 1
            or max_bytes > ABSOLUTE_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES
        ):
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST) from None
        result = self._require_teams_chat_hosted_content_client().read_teams_chat_hosted_content_bytes(
            message=validated_message,
            hosted_content=validated_hosted,
            max_bytes=max_bytes,
        )
        return validate_msgraph_teams_chat_hosted_content_bytes(
            result,
            message=validated_message,
            hosted_content=validated_hosted,
            max_bytes=max_bytes,
        )

    def read_teams_channels_page(
        self,
        *,
        team_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphTeamsChannelPage:
        try:
            from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
                validate_msgraph_teams_team_id,
            )

            validate_msgraph_teams_team_id(team_id)
        except ValueError:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channels request"
            ) from None
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
            validated_continuation = validate_msgraph_teams_channels_continuation(
                continuation,
                team_id=team_id,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_channels_client().read_teams_channels_page(
            team_id=team_id,
            continuation=validated_continuation,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
        return validate_msgraph_teams_channel_page(
            result,
            team_id=team_id,
            graph_base_url=graph_base_url,
        )

    def read_teams_channel_members_page(
        self,
        *,
        channel: MsGraphTeamsChannel,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphTeamsChannelMemberPage:
        validated_channel = validate_msgraph_teams_channel(channel)
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
            validated_continuation = validate_msgraph_teams_channel_members_continuation(
                continuation,
                team_id=validated_channel.team_remote_id,
                channel_id=validated_channel.remote_id,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_channel_members_client().read_teams_channel_members_page(
            channel=validated_channel,
            continuation=validated_continuation,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
        return validate_msgraph_teams_channel_member_page(
            result,
            channel=validated_channel,
            graph_base_url=graph_base_url,
        )

    def read_teams_channel_root_messages_page(
        self,
        *,
        channel: MsGraphTeamsChannel,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
        max_chars_per_message: int = DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChannelRootMessagePage:
        validated_channel = validate_msgraph_teams_channel(channel)
        if type(limit) is not int or limit < 1 or limit > 50:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channel messages request"
            ) from None
        try:
            from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
                _validate_message_max_chars,
            )

            _validate_message_max_chars(max_chars_per_message)
        except ValueError:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channel messages request"
            ) from None
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
            validated_continuation = validate_msgraph_teams_channel_root_messages_continuation(
                continuation,
                team_id=validated_channel.team_remote_id,
                channel_id=validated_channel.remote_id,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_channel_messages_client().read_teams_channel_root_messages_page(
            channel=validated_channel,
            continuation=validated_continuation,
            limit=limit,
            max_chars_per_message=max_chars_per_message,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
        return validate_msgraph_teams_channel_root_message_page(
            result,
            team_id=validated_channel.team_remote_id,
            channel_id=validated_channel.remote_id,
            graph_base_url=graph_base_url,
            max_chars_per_message=max_chars_per_message,
        )

    def read_teams_channel_replies_page(
        self,
        *,
        root_message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
        max_chars_per_message: int = DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChannelReplyPage:
        validated_root = validate_msgraph_teams_channel_message(root_message)
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
            MsGraphTeamsChannelMessageKind,
        )

        if validated_root.message_kind is not MsGraphTeamsChannelMessageKind.ROOT:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channel messages request"
            ) from None
        if type(limit) is not int or limit < 1 or limit > 50:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channel messages request"
            ) from None
        try:
            from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
                _validate_message_max_chars,
            )

            _validate_message_max_chars(max_chars_per_message)
        except ValueError:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channel messages request"
            ) from None
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
            validated_continuation = validate_msgraph_teams_channel_replies_continuation(
                continuation,
                team_id=validated_root.team_remote_id,
                channel_id=validated_root.channel_remote_id,
                root_message_remote_id=validated_root.remote_id,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_channel_messages_client().read_teams_channel_replies_page(
            root_message=validated_root,
            continuation=validated_continuation,
            limit=limit,
            max_chars_per_message=max_chars_per_message,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
        return validate_msgraph_teams_channel_reply_page(
            result,
            team_id=validated_root.team_remote_id,
            channel_id=validated_root.channel_remote_id,
            root_message_remote_id=validated_root.remote_id,
            root_message_revision=validated_root.revision,
            graph_base_url=graph_base_url,
            max_chars_per_message=max_chars_per_message,
        )

    def read_teams_channel_root_messages_page_by_reference(
        self,
        *,
        channel: MsGraphTeamsChannelReference,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
        max_chars_per_message: int = DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChannelRootMessagePage:
        try:
            validated_channel = validate_msgraph_teams_channel_reference(channel)
        except ValueError:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channel messages request"
            ) from None
        if type(limit) is not int or limit < 1 or limit > 50:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channel messages request"
            ) from None
        try:
            from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
                _validate_message_max_chars,
            )

            _validate_message_max_chars(max_chars_per_message)
        except ValueError:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channel messages request"
            ) from None
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
            validated_continuation = validate_msgraph_teams_channel_root_messages_continuation(
                continuation,
                team_id=validated_channel.team_remote_id,
                channel_id=validated_channel.channel_remote_id,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_channel_reference_paging_client().read_teams_channel_root_messages_page_by_reference(
            channel=validated_channel,
            continuation=validated_continuation,
            limit=limit,
            max_chars_per_message=max_chars_per_message,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
        return validate_msgraph_teams_channel_root_message_page(
            result,
            team_id=validated_channel.team_remote_id,
            channel_id=validated_channel.channel_remote_id,
            graph_base_url=graph_base_url,
            max_chars_per_message=max_chars_per_message,
        )

    def read_teams_channel_replies_page_by_reference(
        self,
        *,
        root_message: MsGraphTeamsChannelRootMessageReference,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
        max_chars_per_message: int = DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChannelReplyPage:
        try:
            validated_root = validate_msgraph_teams_channel_root_message_reference(root_message)
        except ValueError:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channel messages request"
            ) from None
        if type(limit) is not int or limit < 1 or limit > 50:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channel messages request"
            ) from None
        try:
            from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
                _validate_message_max_chars,
            )

            _validate_message_max_chars(max_chars_per_message)
        except ValueError:
            raise IntegrationConfigurationError(
                "invalid Microsoft Graph Teams channel messages request"
            ) from None
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
            validated_continuation = validate_msgraph_teams_channel_replies_continuation(
                continuation,
                team_id=validated_root.team_remote_id,
                channel_id=validated_root.channel_remote_id,
                root_message_remote_id=validated_root.remote_id,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_channel_reference_paging_client().read_teams_channel_replies_page_by_reference(
            root_message=validated_root,
            continuation=validated_continuation,
            limit=limit,
            max_chars_per_message=max_chars_per_message,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
        return validate_msgraph_teams_channel_reply_page(
            result,
            team_id=validated_root.team_remote_id,
            channel_id=validated_root.channel_remote_id,
            root_message_remote_id=validated_root.remote_id,
            root_message_revision=validated_root.revision,
            graph_base_url=graph_base_url,
            max_chars_per_message=max_chars_per_message,
        )

    def read_teams_channel_message_content(
        self,
        *,
        message: MsGraphTeamsChannelMessageReference,
        max_chars: int = DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChannelMessage:
        validated_reference = validate_msgraph_teams_channel_message_reference(message)
        if (
            type(max_chars) is not int
            or max_chars < 1
            or max_chars > ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS
        ):
            raise IntegrationConfigurationError(_INVALID_TEAMS_CHANNEL_CONTENT_REQUEST) from None
        result = self._require_teams_channel_content_client().read_teams_channel_message_content(
            message=validated_reference,
            max_chars=max_chars,
        )
        return validate_msgraph_teams_channel_message_content(
            result,
            reference=validated_reference,
            max_chars=max_chars,
        )

    def read_teams_channel_hosted_contents_page(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphTeamsChannelHostedContentPage:
        validated_message = validate_msgraph_teams_channel_message(message)
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
            MsGraphTeamsChannelMessageState,
        )

        if validated_message.state is not MsGraphTeamsChannelMessageState.ACTIVE:
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST) from None
        graph_base_url: str | None = None
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
            validated_continuation = validate_msgraph_teams_channel_hosted_contents_continuation(
                continuation,
                team_id=validated_message.team_remote_id,
                channel_id=validated_message.channel_remote_id,
                thread_root_id=validated_message.thread_root_remote_id,
                message_id=validated_message.remote_id,
                message_kind=validated_message.message_kind,
                graph_base_url=graph_base_url,
            )
        result = self._require_teams_channel_hosted_content_client().read_teams_channel_hosted_contents_page(
            message=validated_message,
            continuation=validated_continuation,
        )
        if graph_base_url is None:
            graph_base_url = self._graph_base_url_for_teams_channel_validation()
        return validate_msgraph_teams_channel_hosted_content_page(
            result,
            message=validated_message,
            graph_base_url=graph_base_url,
        )

    def read_teams_channel_hosted_content_bytes(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        hosted_content: MsGraphTeamsChannelHostedContent,
        max_bytes: int = DEFAULT_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES,
    ) -> MsGraphTeamsChannelHostedContentBytes:
        validated_message = validate_msgraph_teams_channel_message(message)
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
            MsGraphTeamsChannelMessageState,
        )

        if validated_message.state is not MsGraphTeamsChannelMessageState.ACTIVE:
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST) from None
        validated_hosted = validate_msgraph_teams_channel_hosted_content(hosted_content)
        if (
            validated_hosted.team_remote_id != validated_message.team_remote_id
            or validated_hosted.channel_remote_id != validated_message.channel_remote_id
            or validated_hosted.message_remote_id != validated_message.remote_id
            or validated_hosted.thread_root_remote_id != validated_message.thread_root_remote_id
            or validated_hosted.message_kind != validated_message.message_kind
            or validated_hosted.message_revision != validated_message.revision
        ):
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST) from None
        if (
            type(max_bytes) is not int
            or max_bytes < 1
            or max_bytes > ABSOLUTE_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES
        ):
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST) from None
        result = self._require_teams_channel_hosted_content_client().read_teams_channel_hosted_content_bytes(
            message=validated_message,
            hosted_content=validated_hosted,
            max_bytes=max_bytes,
        )
        return validate_msgraph_teams_channel_hosted_content_bytes(
            result,
            message=validated_message,
            hosted_content=validated_hosted,
            max_bytes=max_bytes,
        )

    def read_drive_file_content(
        self,
        *,
        item: MsGraphDriveItem,
        max_bytes: int = DEFAULT_DRIVE_CONTENT_MAX_BYTES,
    ) -> MsGraphDriveFileContent:
        return self._require_drive_content_client().read_drive_file_content(
            item=item,
            max_bytes=max_bytes,
        )

    def create_event(
        self,
        user_id: str,
        *,
        subject: str,
        start: str,
        end: str,
        location: str = "",
        attendees: Sequence[str] = (),
    ):
        return self._require_client().create_event(
            user_id,
            subject=subject,
            start=start,
            end=end,
            location=location,
            attendees=attendees,
        )

    def get_message(self, user_id: str, message_id: str):
        return self._require_client().get_message(user_id, message_id)

    def get_user(self, user_id: str):
        return self._require_client().get_user(user_id)

    def list_calendar_events(
        self,
        user_id: str,
        *,
        start: str,
        end: str,
        limit: int = 50,
    ):
        return self._require_client().list_calendar_events(
            user_id,
            start=start,
            end=end,
            limit=limit,
        )

    def list_messages(
        self,
        user_id: str,
        *,
        folder: str = "inbox",
        limit: int = 25,
    ):
        return self._require_client().list_messages(user_id, folder=folder, limit=limit)

    def reply_message(self, user_id: str, message_id: str, *, body: str) -> None:
        return self._require_client().reply_message(user_id, message_id, body=body)

    def send_mail(
        self,
        user_id: str,
        *,
        subject: str,
        body: str,
        to: Sequence[str],
    ) -> None:
        return self._require_client().send_mail(user_id, subject=subject, body=body, to=to)

    def _require_client(self) -> CollaborationSuite:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client

    def _require_drive_client(self) -> MsGraphDriveKnowledgeReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphDriveKnowledgeReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Drive knowledge capability",
            )
        return client

    def _require_drive_content_client(self) -> MsGraphDriveContentReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphDriveContentReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph Drive download client is not configured",
            )
        return client

    def _require_drive_permissions_client(self) -> MsGraphDrivePermissionsReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphDrivePermissionsReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Drive permissions capability",
            )
        return client

    def _require_mail_folders_client(self) -> MsGraphMailFoldersReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphMailFoldersReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Mail folders knowledge capability",
            )
        return client

    def _require_mail_messages_client(self) -> MsGraphMailMessagesReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphMailMessagesReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Mail messages delta capability",
            )
        return client

    def _require_mail_content_client(self) -> MsGraphMailContentReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphMailContentReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Mail content capability",
            )
        return client

    def _require_mail_attachments_client(self) -> MsGraphMailAttachmentsReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphMailAttachmentsReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Mail attachments capability",
            )
        return client

    def _require_calendars_client(self) -> MsGraphCalendarsReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphCalendarsReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Calendar inventory capability",
            )
        return client

    def _require_calendar_events_client(self) -> MsGraphCalendarEventsReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphCalendarEventsReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Calendar events delta capability",
            )
        return client

    def _require_calendar_event_snapshots_client(self) -> MsGraphCalendarEventSnapshotsReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphCalendarEventSnapshotsReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Calendar events snapshot capability",
            )
        return client

    def _require_calendar_content_client(self) -> MsGraphCalendarContentReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphCalendarContentReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Calendar event content capability",
            )
        return client

    def _require_calendar_attachments_client(self) -> MsGraphCalendarAttachmentsReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphCalendarAttachmentsReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Calendar attachments capability",
            )
        return client

    def _require_teams_chats_client(self) -> MsGraphTeamsChatsReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChatsReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams chats capability",
            )
        return client

    def _require_teams_chat_members_client(self) -> MsGraphTeamsChatMembersReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChatMembersReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams chat members capability",
            )
        return client

    def _require_teams_chat_messages_client(self) -> MsGraphTeamsChatMessagesReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChatMessagesReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams chat messages capability",
            )
        return client

    def _require_teams_chat_reference_paging_client(
        self,
    ) -> MsGraphTeamsChatReferencePagingReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChatReferencePagingReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams chat reference paging capability",
            )
        return client

    def _require_teams_chat_content_client(self) -> MsGraphTeamsChatContentReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChatContentReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams chat message content capability",
            )
        return client

    def _require_teams_chat_hosted_content_client(self) -> MsGraphTeamsChatHostedContentReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChatHostedContentReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams hosted content capability",
            )
        return client

    def _require_teams_channels_client(self) -> MsGraphTeamsChannelsReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChannelsReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams channels capability",
            )
        return client

    def _require_teams_channel_members_client(self) -> MsGraphTeamsChannelMembersReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChannelMembersReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams channel members capability",
            )
        return client

    def _require_teams_channel_messages_client(self) -> MsGraphTeamsChannelMessagesReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChannelMessagesReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams channel messages capability",
            )
        return client

    def _require_teams_channel_reference_paging_client(
        self,
    ) -> MsGraphTeamsChannelReferencePagingReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChannelReferencePagingReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams channel reference paging capability",
            )
        return client

    def _require_teams_channel_content_client(self) -> MsGraphTeamsChannelContentReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChannelContentReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams channel message content capability",
            )
        return client

    def _require_teams_channel_hosted_content_client(
        self,
    ) -> MsGraphTeamsChannelHostedContentReadClient:
        client = self._require_client()
        if not isinstance(client, MsGraphTeamsChannelHostedContentReadClient):
            raise IntegrationConfigurationError(
                "Microsoft Graph integration does not expose Teams channel hosted content capability",
            )
        return client

    def _graph_base_url_for_mail_messages_validation(self) -> str:
        client = self._require_client()
        if isinstance(client, GraphRestClient):
            return client.config.graph_base_url
        if isinstance(client, _Ms365GraphCollaborationSuite):
            return client.rest_client.config.graph_base_url
        raise IntegrationConfigurationError(
            "Microsoft Graph Mail messages delta validation is not configured",
        )

    def _graph_base_url_for_mail_attachments_validation(self) -> str:
        client = self._require_client()
        if isinstance(client, GraphRestClient):
            return client.config.graph_base_url
        if isinstance(client, _Ms365GraphCollaborationSuite):
            return client.rest_client.config.graph_base_url
        raise IntegrationConfigurationError(
            "Microsoft Graph Mail attachment validation is not configured",
        )

    def _graph_base_url_for_calendar_validation(self) -> str:
        client = self._require_client()
        if isinstance(client, GraphRestClient):
            return client.config.graph_base_url
        if isinstance(client, _Ms365GraphCollaborationSuite):
            return client.rest_client.config.graph_base_url
        raise IntegrationConfigurationError(
            "Microsoft Graph Calendar validation is not configured",
        )

    def _graph_base_url_for_teams_chat_validation(self) -> str:
        client = self._require_client()
        if isinstance(client, GraphRestClient):
            return client.config.graph_base_url
        if isinstance(client, _Ms365GraphCollaborationSuite):
            return client.rest_client.config.graph_base_url
        raise IntegrationConfigurationError(
            "Microsoft Graph Teams Chat validation is not configured",
        )

    def _graph_base_url_for_teams_channel_validation(self) -> str:
        client = self._require_client()
        if isinstance(client, GraphRestClient):
            return client.config.graph_base_url
        if isinstance(client, _Ms365GraphCollaborationSuite):
            return client.rest_client.config.graph_base_url
        raise IntegrationConfigurationError(
            "Microsoft Graph Teams Channel validation is not configured",
        )

    @classmethod
    def from_client(
        cls,
        client: Ms365GraphCollaborationSuiteClient,
        *,
        enabled: bool = False,
    ) -> Ms365GraphCollaborationSuiteIntegration:
        integration = cls.for_provider(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            display_name="Ms365 Graph",
            config=Ms365GraphCollaborationSuiteIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> Ms365GraphCollaborationSuiteClient | None:
        return self._client

CollaborationSuite.register(Ms365GraphCollaborationSuiteIntegration)
