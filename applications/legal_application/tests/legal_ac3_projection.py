# © Artur Czarnecki. All rights reserved.

"""Legal application test registry projection helper (AC-3)."""

from __future__ import annotations

from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.wiring import build_legal_environment_profile, build_legal_manifest
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection


def build_legal_test_registry_projection(
    settings: LegalBackendSettings | None = None,
    *,
    revision_id: str = "legal-test-runtime-revision",
) -> MaterializedRegistryProjection:
    resolved_settings = settings or LegalBackendSettings.from_env()
    manifest = build_legal_manifest(resolved_settings)
    env = manifest.environment or build_legal_environment_profile(resolved_settings)
    bindings = tuple(manifest.enabled_agents())
    return build_test_registry_projection(
        manifest,
        env,
        enabled_bindings=bindings,
        revision_id=revision_id,
        settings=resolved_settings,
    )
