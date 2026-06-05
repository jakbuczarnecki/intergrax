# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Construct Tier-2 Legal Agent and registry from :class:`~legal_application.host.settings.LegalBackendSettings`."""

from __future__ import annotations

from legal.legal_agent import LegalAgent
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from legal_application.host.settings import LegalBackendSettings


def build_legal_manifest(settings: LegalBackendSettings) -> ApplicationManifest:
    """Legal manifest with runtime contract id from settings."""
    from legal_application.manifest import LEGAL_APPLICATION_MANIFEST

    base = LEGAL_APPLICATION_MANIFEST.agents[0]
    binding = base.model_copy(update={"contract_id": settings.legal_default_agent_id})
    return LEGAL_APPLICATION_MANIFEST.model_copy(update={"agents": [binding]})


def build_legal_environment_profile(settings: LegalBackendSettings) -> ApplicationEnvironmentProfile:
    """Product environment for legal host (H-APP.5.2)."""
    from intergrax.runtime.modality.modality_profile import ModalityProfile, ModalityPlane
    from legal_application.manifest import LEGAL_APPLICATION_MANIFEST

    tool_ids = list(settings.enabled_tool_ids)
    modality_profile = None
    if settings.enable_modality_tools:
        modality_profile = ModalityProfile(
            profile_id="legal.modality",
            allowed_planes={ModalityPlane.DEDICATED_INFERENCE},
        )
        for tool_id in (
            "vision.detect",
            "vision.segment",
            "vision.ocr_regions",
            "speech.synthesize",
            "speech.transcribe",
            "ml.predict",
            "ml.explain",
            "ml.batch_predict",
        ):
            if tool_id not in tool_ids:
                tool_ids.append(tool_id)
    return ApplicationEnvironmentProfile.product_defaults(
        profile_id="legal.product",
        skill_bundles=["legal"],
        tool_ids=tool_ids,
        domain_fragments={"legal.contract_review.policy": "legal.contract_review.policy"},
    ).model_copy(
        update={
            "integration_profile": LEGAL_APPLICATION_MANIFEST.integration_profile,
            "modality_profile": modality_profile,
        },
    ).with_harness_memory()


def build_legal_registry(settings: LegalBackendSettings) -> AgentRegistry:
    """Materialize Legal agent via unified Tier-3 environment wiring."""
    manifest = build_legal_manifest(settings)
    env = manifest.environment or build_legal_environment_profile(settings)
    if manifest.environment is None:
        manifest = manifest.model_copy(update={"environment": env})
    env_wiring = wire_application_environment(manifest, env, settings=settings)
    return build_application_registry(manifest, env_wiring.build_context)


def build_legal_agent(
    settings: LegalBackendSettings,
    *,
    ctx: ApplicationBuildContext | None = None,
) -> LegalAgent:
    """Scaffold baseline: zero-arg UAEP agent; Tier-3 supplies tools/skills via ``ctx``."""
    _ = settings, ctx
    return LegalAgent()
