# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public factory-facing context for Tier-3 agent materialization (EBH-2D-A)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, overload

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest

TSettings = TypeVar("TSettings")


@dataclass(frozen=True)
class ApplicationBuildContext(Generic[TSettings]):
    """
    Declarative inputs available when materializing agents for an application host.

    Runtime registries, event buses, and wiring objects live in
    :class:`intergrax.applications._shared.application_composition_context.ApplicationCompositionContext`
    (composition layer — not public ABI).

    ``settings`` is application-specific (e.g. ``LabApplicationSettings``,
    ``LegalBackendSettings``), injected by the host after env/filesystem load.
    The type parameter ties each host factory to its owned settings DTO without
    central registration in platform core.
    """

    manifest: ApplicationManifest
    settings: TSettings | None = None
    strict_harness: bool = False
    trace_db_path: Path | None = None
    environment: ApplicationEnvironmentProfile | None = None

    @overload
    @classmethod
    def for_manifest(
        cls,
        manifest: ApplicationManifest,
        *,
        settings: None = None,
        strict_harness: bool = False,
        trace_db_path: Path | None = None,
        environment: ApplicationEnvironmentProfile | None = None,
    ) -> ApplicationBuildContext[None]: ...

    @overload
    @classmethod
    def for_manifest(
        cls,
        manifest: ApplicationManifest,
        *,
        settings: TSettings,
        strict_harness: bool = False,
        trace_db_path: Path | None = None,
        environment: ApplicationEnvironmentProfile | None = None,
    ) -> ApplicationBuildContext[TSettings]: ...

    @classmethod
    def for_manifest(
        cls,
        manifest: ApplicationManifest,
        *,
        settings: TSettings | None = None,
        strict_harness: bool = False,
        trace_db_path: Path | None = None,
        environment: ApplicationEnvironmentProfile | None = None,
    ) -> ApplicationBuildContext[TSettings] | ApplicationBuildContext[None]:
        return cls(
            manifest=manifest,
            settings=settings,
            strict_harness=strict_harness,
            trace_db_path=trace_db_path,
            environment=environment,
        )


from intergrax.applications.contracts.manifest import _rebuild_application_manifest_model

_rebuild_application_manifest_model()
