# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ms365 Graph collaboration suite integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    DEFAULT_DRIVE_CONTENT_MAX_BYTES,
    MsGraphDriveContentReadClient,
    MsGraphDriveDeltaPage,
    MsGraphDriveFileContent,
    MsGraphDriveItem,
    MsGraphDriveKnowledgeReadClient,
    MsGraphKnowledgeContinuation,
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
