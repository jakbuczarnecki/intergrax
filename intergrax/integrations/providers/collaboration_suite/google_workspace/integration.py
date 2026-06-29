# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace collaboration suite integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.runtime.integrations.categories.collaboration import CollaborationSuiteIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID = "google_workspace"


class GoogleWorkspaceCollaborationSuiteIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Google Workspace collaboration suite integration."""

    pass


GoogleWorkspaceCollaborationSuiteClient = CollaborationSuite

class GoogleWorkspaceCollaborationSuiteIntegration(CollaborationSuiteIntegrationContract):
    """
    Single public Google Workspace collaboration suite entrypoint.

    Legacy catalog factory (create_google_workspace_collaboration_suite) owns catalog behavior; legacy factories use from_client().
    """

    config: GoogleWorkspaceCollaborationSuiteIntegrationConfig = GoogleWorkspaceCollaborationSuiteIntegrationConfig()
    _client: GoogleWorkspaceCollaborationSuiteClient | None = PrivateAttr(default=None)
    

    def create_event(self, user_id, subject, start, end, location: str = '', attendees: Sequence[str] = ()):
        return self._require_client().create_event(user_id, subject, start, end, location=location, attendees=attendees)

    def get_message(self, user_id, message_id):
        return self._require_client().get_message(user_id, message_id)

    def get_user(self, user_id):
        return self._require_client().get_user(user_id)

    def list_calendar_events(self, user_id, start, end, limit: int = 50):
        return self._require_client().list_calendar_events(user_id, start, end, limit=limit)

    def list_messages(self, user_id, folder: str = 'inbox', limit: int = 25):
        return self._require_client().list_messages(user_id, folder=folder, limit=limit)

    def reply_message(self, user_id, message_id, body):
        return self._require_client().reply_message(user_id, message_id, body)

    def send_mail(self, user_id, subject, body, to):
        return self._require_client().send_mail(user_id, subject, body, to)

    def _require_client(self) -> CollaborationSuite:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: GoogleWorkspaceCollaborationSuiteClient,
        *,
        enabled: bool = False,
    ) -> GoogleWorkspaceCollaborationSuiteIntegration:
        integration = cls.for_provider(
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            display_name="Google Workspace",
            config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GoogleWorkspaceCollaborationSuiteClient | None:
        return self._client

CollaborationSuite.register(GoogleWorkspaceCollaborationSuiteIntegration)
