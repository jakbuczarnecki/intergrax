# © Artur Czarnecki. All rights reserved.

"""Legal application test registry projection helper (AC-3)."""

from __future__ import annotations

from intergrax.applications._shared.production_platform_persistence import (
    resolve_reference_production_strict_host_environment,
)
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.skills.registry.profile import SkillProfile
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.wiring import build_legal_environment_profile, build_legal_manifest
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection

_LEGAL_HOST_TEST_SKILL_IDS: tuple[str, ...] = ("legal.contract_review",)


def build_legal_host_test_environment(
    settings: LegalBackendSettings,
    *,
    strict_platform_backing: bool = False,
) -> ApplicationEnvironmentProfile:
    env = build_legal_environment_profile(settings)
    env = env.model_copy(
        update={
            "skill_profile": SkillProfile(
                enabled_bundles=["legal"],
                enabled=list(_LEGAL_HOST_TEST_SKILL_IDS),
            ),
        },
    )
    if strict_platform_backing and env.execution_mode.value == "strict":
        env = resolve_reference_production_strict_host_environment(env)
    return env


def build_legal_host_test_manifest(
    settings: LegalBackendSettings | None = None,
    *,
    strict_platform_backing: bool = False,
) -> ApplicationManifest:
    resolved_settings = settings or LegalBackendSettings.from_env()
    return build_legal_manifest(resolved_settings).model_copy(
        update={
            "environment": build_legal_host_test_environment(
                resolved_settings,
                strict_platform_backing=strict_platform_backing,
            ),
        },
    )


def build_legal_test_registry_projection(
    settings: LegalBackendSettings | None = None,
    *,
    revision_id: str = "legal-test-runtime-revision",
) -> MaterializedRegistryProjection:
    resolved_settings = settings or LegalBackendSettings.from_env()
    manifest = build_legal_host_test_manifest(resolved_settings)
    env = manifest.environment
    bindings = tuple(manifest.enabled_agents())
    return build_test_registry_projection(
        manifest,
        env,
        enabled_bindings=bindings,
        revision_id=revision_id,
        settings=resolved_settings,
    )
