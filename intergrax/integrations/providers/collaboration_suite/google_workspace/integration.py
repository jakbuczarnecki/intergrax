# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace collaboration suite integration foundation shell."""

from __future__ import annotations

import threading

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.google_workspace.config import (
    GoogleWorkspaceCollaborationSuiteCompositionMode,
    GoogleWorkspaceCollaborationSuiteIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS,
    GoogleWorkspaceClientFactory,
    GoogleWorkspaceClientFamily,
    GoogleWorkspaceCredentialResolver,
    GoogleWorkspaceSourceKind,
    GoogleWorkspaceTransport,
    copy_google_workspace_credential_material,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
    GoogleDriveChangePage,
    GoogleDriveItem,
    GoogleDriveItemPage,
    GoogleDriveKnowledgeReader,
    GoogleDriveScope,
    GoogleDriveSharedDrivePage,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspacePageToken,
)
from intergrax.runtime.integrations.categories._base import (
    _CONNECT_READ_WRITE_HEALTH,
    category_for_provider,
)
from intergrax.runtime.integrations.categories.collaboration import CollaborationSuiteIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationHealth,
    PlatformIntegrationKind,
    PlatformIntegrationStatus,
)

GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID = "google_workspace"

_DISABLED_INTEGRATION_MESSAGE = "Google Workspace integration is disabled"
_CREDENTIAL_RESOLUTION_FAILURE_MESSAGE = "Google Workspace credential resolution failed"
_CLIENT_FACTORY_FAILURE_MESSAGE = "Google Workspace client family could not be created"
_INVALID_FACTORY_FAMILY_MESSAGE = (
    "Google Workspace client factory returned an invalid client family"
)
_INVALID_INJECTED_FAMILY_MESSAGE = (
    "Google Workspace injected client does not expose a valid client family"
)

GoogleWorkspaceCollaborationSuiteClient = CollaborationSuite


def _validate_client_family(
    family: object,
    *,
    injected: bool,
) -> GoogleWorkspaceClientFamily:
    if not isinstance(family, GoogleWorkspaceClientFamily):
        if injected:
            raise IntegrationConfigurationError(_INVALID_INJECTED_FAMILY_MESSAGE)
        raise IntegrationConfigurationError(_INVALID_FACTORY_FAMILY_MESSAGE)
    try:
        transport = family.transport
    except Exception:
        if injected:
            raise IntegrationConfigurationError(_INVALID_INJECTED_FAMILY_MESSAGE) from None
        raise IntegrationConfigurationError(_INVALID_FACTORY_FAMILY_MESSAGE) from None
    if not isinstance(transport, GoogleWorkspaceTransport):
        if injected:
            raise IntegrationConfigurationError(_INVALID_INJECTED_FAMILY_MESSAGE)
        raise IntegrationConfigurationError(_INVALID_FACTORY_FAMILY_MESSAGE)
    return family


class GoogleWorkspaceCollaborationSuiteIntegration(CollaborationSuiteIntegrationContract):
    """
    Single public Google Workspace collaboration suite entrypoint.

  Foundation shell only — credential resolution and client creation are injected
    and validated at runtime when enabled.
    """

    config: GoogleWorkspaceCollaborationSuiteIntegrationConfig = (
        GoogleWorkspaceCollaborationSuiteIntegrationConfig()
    )
    _credential_resolver: GoogleWorkspaceCredentialResolver | None = PrivateAttr(default=None)
    _client_factory: GoogleWorkspaceClientFactory | None = PrivateAttr(default=None)
    _client: GoogleWorkspaceCollaborationSuiteClient | None = PrivateAttr(default=None)
    _client_family: GoogleWorkspaceClientFamily | None = PrivateAttr(default=None)
    _client_family_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def supported_source_kinds(self) -> tuple[GoogleWorkspaceSourceKind, ...]:
        return GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS

    @property
    def credential_resolver(self) -> GoogleWorkspaceCredentialResolver | None:
        return self._credential_resolver

    @property
    def client_factory(self) -> GoogleWorkspaceClientFactory | None:
        return self._client_factory

    def validate_runtime(self) -> None:
        """Fail closed when enabled integration lacks required injected dependencies."""
        if not self.config.enabled:
            return
        if (
            self.config.composition_mode
            == GoogleWorkspaceCollaborationSuiteCompositionMode.INJECTED_CLIENT
        ):
            if self._client is None:
                raise IntegrationConfigurationError(
                    f"{type(self).__name__} requires an injected client when enabled=True "
                    "in injected_client composition mode",
                )
            return
        if self._credential_resolver is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires an injected credential resolver when enabled=True",
            )
        if self._client_factory is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires an injected client factory when enabled=True",
            )

    def require_client_family(self) -> GoogleWorkspaceClientFamily:
        """Materialize and cache the shared client family on first successful request."""
        if not self.config.enabled:
            raise IntegrationConfigurationError(_DISABLED_INTEGRATION_MESSAGE)

        if (
            self.config.composition_mode
            == GoogleWorkspaceCollaborationSuiteCompositionMode.INJECTED_CLIENT
        ):
            client = self._client
            if client is None:
                raise IntegrationConfigurationError(
                    f"{type(self).__name__} requires an injected client when enabled=True "
                    "in injected_client composition mode",
                )
            if isinstance(client, GoogleWorkspaceClientFamily):
                return _validate_client_family(client, injected=True)
            raise IntegrationConfigurationError(_INVALID_INJECTED_FAMILY_MESSAGE)

        cached = self._client_family
        if cached is not None:
            return cached

        with self._client_family_lock:
            cached = self._client_family
            if cached is not None:
                return cached

            self.validate_runtime()
            resolver = self._credential_resolver
            factory = self._client_factory
            if resolver is None or factory is None:
                raise IntegrationConfigurationError(
                    f"{type(self).__name__} requires injected credential resolver and client "
                    "factory when enabled=True",
                )

            try:
                resolved_material = resolver.resolve_credential(self.config.credential_ref)
            except Exception:
                raise IntegrationConfigurationError(
                    _CREDENTIAL_RESOLUTION_FAILURE_MESSAGE,
                ) from None

            credential_material = copy_google_workspace_credential_material(resolved_material)

            try:
                family = factory.create_client_family(credential_material=credential_material)
            except Exception:
                raise IntegrationDependencyError(_CLIENT_FACTORY_FAILURE_MESSAGE) from None

            validated_family = _validate_client_family(family, injected=False)
            self._client_family = validated_family
            return validated_family

    def _drive_reader(self) -> GoogleDriveKnowledgeReader:
        family = self.require_client_family()
        return GoogleDriveKnowledgeReader(transport=family.transport)

    def list_drive_shared_drives_page(
        self,
        *,
        page_token: GoogleWorkspacePageToken | None = None,
        limit: int = 100,
    ) -> GoogleDriveSharedDrivePage:
        return self._drive_reader().list_shared_drives_page(
            page_token=page_token,
            limit=limit,
        )

    def read_drive_items_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken | None = None,
        limit: int = 200,
    ) -> GoogleDriveItemPage:
        return self._drive_reader().read_items_page(
            scope=scope,
            page_token=page_token,
            limit=limit,
        )

    def read_drive_item(
        self,
        *,
        scope: GoogleDriveScope,
        file_id: str,
    ) -> GoogleDriveItem:
        return self._drive_reader().read_item(scope=scope, file_id=file_id)

    def read_drive_start_page_token(
        self,
        *,
        scope: GoogleDriveScope,
    ) -> GoogleWorkspacePageToken:
        return self._drive_reader().read_start_page_token(scope=scope)

    def read_drive_changes_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken,
        limit: int = 200,
    ) -> GoogleDriveChangePage:
        return self._drive_reader().read_changes_page(
            scope=scope,
            page_token=page_token,
            limit=limit,
        )

    def check_health(self) -> PlatformIntegrationHealth:
        if not self.config.enabled:
            return PlatformIntegrationHealth(
                status=PlatformIntegrationStatus.DISABLED,
                message="integration is disabled",
            )
        try:
            self.validate_runtime()
        except IntegrationConfigurationError as exc:
            return PlatformIntegrationHealth(
                status=PlatformIntegrationStatus.UNAVAILABLE,
                message=str(exc),
            )
        return super().check_health()

    @classmethod
    def compose(
        cls,
        *,
        config: GoogleWorkspaceCollaborationSuiteIntegrationConfig,
        credential_resolver: GoogleWorkspaceCredentialResolver,
        client_factory: GoogleWorkspaceClientFactory,
        display_name: str = "Google Workspace",
    ) -> GoogleWorkspaceCollaborationSuiteIntegration:
        integration = cls.for_provider(
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            display_name=display_name,
            config=config,
        )
        integration._credential_resolver = credential_resolver
        integration._client_factory = client_factory
        return integration

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
            config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
                enabled=enabled,
                composition_mode=GoogleWorkspaceCollaborationSuiteCompositionMode.INJECTED_CLIENT,
            ),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GoogleWorkspaceCollaborationSuiteClient | None:
        return self._client

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: GoogleWorkspaceCollaborationSuiteIntegrationConfig | None = None,
    ) -> GoogleWorkspaceCollaborationSuiteIntegration:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.COLLABORATION_SUITE.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config or GoogleWorkspaceCollaborationSuiteIntegrationConfig(),
        )


CollaborationSuite.register(GoogleWorkspaceCollaborationSuiteIntegration)
