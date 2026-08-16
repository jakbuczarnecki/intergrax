# © Artur Czarnecki. All rights reserved.

"""Research application test registry projection helper (AC-3)."""

from __future__ import annotations

from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from research_application.host.agent_builders import RESEARCH_AGENT_BUILDERS
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_environment_profile
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection


def build_research_test_registry_projection(
    settings: ResearchBackendSettings | None = None,
    *,
    revision_id: str = "research-test-runtime-revision",
    enabled_contract_ids: tuple[str, ...] | None = None,
) -> MaterializedRegistryProjection:
    """Revision-bound projection for research factory integration tests."""
    resolved_settings = settings or ResearchBackendSettings.from_env()
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(resolved_settings)
    return build_test_registry_projection(
        manifest,
        env,
        builders=RESEARCH_AGENT_BUILDERS,
        revision_id=revision_id,
        settings=resolved_settings,
        enabled_contract_stems=frozenset(enabled_contract_ids) if enabled_contract_ids else None,
    )
