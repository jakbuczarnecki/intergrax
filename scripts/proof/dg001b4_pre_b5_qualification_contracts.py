# © Artur Czarnecki. All rights reserved.

"""Qualification-owned contracts for DG-001B4 pre-B5 worker bootstrap failure injection."""

from __future__ import annotations

from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.contracts.document_store import DocumentStore
from local_workspace_application.host.background_worker_main import (
    LocalWorkspaceWorkerBootstrapDiagnostics,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

_QUALIFICATION_SECRET_SENTINEL = "DG001B-B4-PRE-B5-SECRET-SENTINEL"
_TYPED_PRE_B5_EXCEPTION_MESSAGE = (
    f"worker bootstrap diagnostics composition failure {_QUALIFICATION_SECRET_SENTINEL}"
)


def qualification_secret_sentinel() -> str:
    """Return the qualification-only sentinel used in thrown pre-B5 bootstrap exceptions."""
    return _QUALIFICATION_SECRET_SENTINEL


class ControlledFailingWorkerBootstrapDiagnosticsSegment:
    """Inject a deterministic pre-B5 failure through the worker diagnostics segment."""

    def __call__(
        self,
        *,
        registry_projection: MaterializedRegistryProjection,
        settings: LocalWorkspaceBackendSettings,
        environment_profile: ApplicationEnvironmentProfile,
        document_store: DocumentStore,
    ) -> LocalWorkspaceWorkerBootstrapDiagnostics:
        del registry_projection, settings, environment_profile, document_store
        raise RuntimeError(_TYPED_PRE_B5_EXCEPTION_MESSAGE)


def controlled_failing_worker_bootstrap_diagnostics_segment() -> (
    ControlledFailingWorkerBootstrapDiagnosticsSegment
):
    return ControlledFailingWorkerBootstrapDiagnosticsSegment()
