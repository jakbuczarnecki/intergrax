# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public factory-facing context for Tier-3 agent materialization (EBH-2D-A)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest


@dataclass(frozen=True)
class ApplicationBuildContext:
    """
    Declarative inputs available when materializing agents for an application host.

    Runtime registries, event buses, and wiring objects live in
    :class:`intergrax.applications._shared.application_composition_context.ApplicationCompositionContext`
    (composition layer — not public ABI).

    ``settings`` is application-specific (e.g. ``LabApplicationSettings``,
    ``LegalBackendSettings``). Factories read env-backed settings here — not
    from global process env directly.
    """

    manifest: ApplicationManifest
    settings: Any = None
    strict_harness: bool = False
    trace_db_path: Path | None = None
    environment: ApplicationEnvironmentProfile | None = None

    @classmethod
    def for_manifest(
        cls,
        manifest: ApplicationManifest,
        *,
        settings: Any = None,
        strict_harness: bool = False,
        trace_db_path: Path | None = None,
        environment: ApplicationEnvironmentProfile | None = None,
    ) -> ApplicationBuildContext:
        return cls(
            manifest=manifest,
            settings=settings,
            strict_harness=strict_harness,
            trace_db_path=trace_db_path,
            environment=environment,
        )


from intergrax.applications.contracts.manifest import _rebuild_application_manifest_model

_rebuild_application_manifest_model()
