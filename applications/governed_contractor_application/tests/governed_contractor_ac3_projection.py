# © Artur Czarnecki. All rights reserved.

"""Governed contractor test registry projection helper (AC-3)."""

from __future__ import annotations

from governed_contractor_application.host.agent_builders import GOVERNED_CONTRACTOR_AGENT_BUILDERS
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.manifest import build_governed_contractor_manifest
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection


def build_governed_contractor_test_registry_projection(
    *,
    revision_id: str = "governed-contractor-test-revision",
) -> MaterializedRegistryProjection:
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    return build_test_registry_projection(
        manifest,
        env,
        builders=GOVERNED_CONTRACTOR_AGENT_BUILDERS,
        revision_id=revision_id,
    )
