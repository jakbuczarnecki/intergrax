# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime context passed to Tier-3 agent factories (Phase N.2.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.integrations.registry.profile import IntegrationProfile


@dataclass(frozen=True)
class ApplicationBuildContext:
    """
    Inputs available when materializing agents for an application host.

    ``settings`` is application-specific (e.g. ``LabApplicationSettings``,
    ``LegalBackendSettings``). Factories read env-backed settings here — not
    from global process env directly.
    """

    manifest: Any
    settings: Any = None
    integration_profile: IntegrationProfile | None = None

    @classmethod
    def for_manifest(
        cls,
        manifest: Any,
        *,
        settings: Any = None,
    ) -> ApplicationBuildContext:
        profile = getattr(manifest, "integration_profile", None)
        return cls(
            manifest=manifest,
            settings=settings,
            integration_profile=profile,
        )
