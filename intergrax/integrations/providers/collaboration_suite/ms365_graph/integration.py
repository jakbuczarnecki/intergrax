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
    DEFAULT_DRIVE_CONTENT_MAX_BYTES,
    MsGraphDriveContentReadClient,
    MsGraphDriveDeltaPage,
    MsGraphDriveFileContent,
    MsGraphDriveItem,
    MsGraphDriveKnowledgeReadClient,
    MsGraphDrivePermissionPage,
    MsGraphDrivePermissionsReadClient,
    MsGraphKnowledgeContinuation,
    MsGraphMailFolderPage,
    MsGraphMailFoldersReadClient,
    MsGraphMailMessageDeltaPage,
    MsGraphMailMessagesReadClient,
    validate_msgraph_drive_permission_page,
    validate_msgraph_mail_folder_page,
    validate_msgraph_mail_message_delta_page,
)
from intergrax.runtime.integrations.categories.collaboration import CollaborationSuiteIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID = "ms365_graph"


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

    def _graph_base_url_for_mail_messages_validation(self) -> str:
        client = self._require_client()
        if isinstance(client, GraphRestClient):
            return client.config.graph_base_url
        if isinstance(client, _Ms365GraphCollaborationSuite):
            return client.rest_client.config.graph_base_url
        raise IntegrationConfigurationError(
            "Microsoft Graph Mail messages delta validation is not configured",
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
