# © Artur Czarnecki. All rights reserved.

"""Reference production lifecycle input for the Research STRICT host."""

from __future__ import annotations

from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications._shared.registry_projection_input_bundle import (
    build_reference_activation_request,
    build_reference_registry_projection_input_bundle,
)
from research_application.host.agent_builders import RESEARCH_AGENT_BUILDERS
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_environment_profile
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST


def build_research_reference_lifecycle_input(
    settings: ResearchBackendSettings | None = None,
    *,
    runtime_revision_id: str = "research-reference-runtime-revision",
    enabled_contract_ids: tuple[str, ...] | None = ("research",),
) -> tuple[RegistryProjectionInputBundle, ActivateRuntimeRevisionRequest]:
    """Explicit deploy input for reference Research production (not host startup)."""
    resolved_settings = settings or ResearchBackendSettings.from_env()
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(resolved_settings)
    projection_input = build_reference_registry_projection_input_bundle(
        manifest,
        env,
        builders=RESEARCH_AGENT_BUILDERS,
        runtime_revision_id=runtime_revision_id,
        enabled_contract_stems=frozenset(enabled_contract_ids) if enabled_contract_ids else None,
        settings=resolved_settings,
    )
    activation_request = build_reference_activation_request(projection_input)
    return projection_input, activation_request


__all__ = ["build_research_reference_lifecycle_input"]
