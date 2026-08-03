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

GoogleWorkspaceCollaborationSuiteClient = CollaborationSuite


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
                return client
            raise IntegrationConfigurationError(
                f"{type(self).__name__} injected client does not expose a Google Workspace "
                "client family",
            )

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
                credential_material = resolver.resolve_credential(self.config.credential_ref)
            except IntegrationConfigurationError:
                raise
            except Exception:
                raise IntegrationConfigurationError(
                    f"{type(self).__name__} could not resolve configured credential reference",
                ) from None

            try:
                family = factory.create_client_family(credential_material=credential_material)
            except IntegrationDependencyError:
                raise
            except IntegrationConfigurationError:
                raise
            except Exception:
                raise IntegrationDependencyError(
                    f"{type(self).__name__} could not create Google Workspace client family",
                ) from None

            if not isinstance(family, GoogleWorkspaceClientFamily):
                raise IntegrationConfigurationError(
                    f"{type(self).__name__} client factory returned an invalid client family",
                )

            self._client_family = family
            return family

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
