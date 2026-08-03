# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default Google Workspace client family and factory."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceClientFamily,
    GoogleWorkspaceRequestExecutorFactory,
    GoogleWorkspaceTransport,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceHttpTransport,
    GoogleWorkspaceRetryPolicy,
)


@dataclass(frozen=True, slots=True)
class DefaultGoogleWorkspaceClientFamily:
    """Immutable client family exposing one shared transport."""

    _transport: GoogleWorkspaceTransport

    @property
    def transport(self) -> GoogleWorkspaceTransport:
        return self._transport


class DefaultGoogleWorkspaceClientFactory:
    """Build one shared client family from credential material via executor factory."""

    def __init__(
        self,
        executor_factory: GoogleWorkspaceRequestExecutorFactory,
        retry_policy: GoogleWorkspaceRetryPolicy,
        *,
        sleeper: Callable[[float], None] | None = None,
        jitter_source: Callable[[], float] | None = None,
    ) -> None:
        self._executor_factory = executor_factory
        self._retry_policy = retry_policy
        self._sleeper = sleeper
        self._jitter_source = jitter_source

    def create_client_family(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> GoogleWorkspaceClientFamily:
        if not isinstance(credential_material, Mapping) or not credential_material:
            raise ValueError("credential_material must be a non-empty mapping")
        material_copy = dict(credential_material)
        try:
            executor = self._executor_factory.create_request_executor(
                credential_material=material_copy,
            )
        except (IntegrationDependencyError, ValueError):
            raise
        except Exception:
            raise IntegrationDependencyError(
                "Google Workspace client family could not be created",
            ) from None
        transport = GoogleWorkspaceHttpTransport(
            executor=executor,
            retry_policy=self._retry_policy,
            sleeper=self._sleeper,
            jitter_source=self._jitter_source,
        )
        return DefaultGoogleWorkspaceClientFamily(_transport=transport)
