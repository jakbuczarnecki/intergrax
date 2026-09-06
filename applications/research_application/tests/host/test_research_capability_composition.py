# © Artur Czarnecki. All rights reserved.

"""Research host capability composition semantics (NPSC-3B-R3V-R1)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.capability_dependency.composition import (
    validate_capability_dependencies_for_environment,
)
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.skill_wiring import build_application_skill_wiring
from intergrax.applications.contracts.capability_dependency import (
    RequiredCapabilityDependencyUnavailableError,
)
from intergrax.skills.providers.research.manifests import (
    RESEARCH_SOURCE_VALIDATOR,
    RESEARCH_WEB_CACHE_ADMIN,
)
from intergrax.skills.registry.factory import build_registry_from_profile
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.tool_requirements import (
    available_tool_ids_for_profile,
    resolve_skill_tool_requirements,
)
from intergrax.tools.registry.profile import ToolProfile
from research_application.host.environment_profile import build_research_environment_profile
from research_application.host.settings import ResearchBackendSettings
from research_application.host.skill_wiring import (
    RESEARCH_DEFAULT_ENABLED_SKILL_IDS,
    build_research_skill_profile,
)
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_default_research_environment_composes_successfully() -> None:
    env = build_research_environment_profile(ResearchBackendSettings())
    wire_application_environment(
        RESEARCH_APPLICATION_MANIFEST,
        env,
        settings=ResearchBackendSettings(),
        conformance_check=False,
    )


def test_default_selected_skills_require_subset_of_authorized_tools() -> None:
    env = build_research_environment_profile(ResearchBackendSettings())
    skill_wiring = build_application_skill_wiring(env.skill_profile)
    resolution = resolve_skill_tool_requirements(
        skill_wiring.registry,
        available_tool_ids_for_profile(env.tool_profile),
    )
    assert resolution.is_satisfied
    assert set(resolution.required_tool_ids) <= set(
        available_tool_ids_for_profile(env.tool_profile),
    )
    assert tuple(env.skill_profile.enabled) == RESEARCH_DEFAULT_ENABLED_SKILL_IDS


def test_unselected_web_cache_admin_does_not_widen_tools() -> None:
    env = build_research_environment_profile(ResearchBackendSettings())
    available = set(available_tool_ids_for_profile(env.tool_profile))
    assert RESEARCH_WEB_CACHE_ADMIN.skill_id not in env.skill_profile.enabled
    assert "websearch.invalidate_cache" not in available


def test_explicit_source_validator_without_parse_preview_fails_closed() -> None:
    env = build_research_environment_profile(ResearchBackendSettings()).model_copy(
        update={
            "skill_profile": build_research_skill_profile(
                enabled_skill_ids=(
                    *RESEARCH_DEFAULT_ENABLED_SKILL_IDS,
                    RESEARCH_SOURCE_VALIDATOR.skill_id,
                ),
            ),
        },
    )
    skill_wiring = build_application_skill_wiring(env.skill_profile)
    with pytest.raises(RequiredCapabilityDependencyUnavailableError):
        validate_capability_dependencies_for_environment(
            env,
            skill_registry=skill_wiring.registry,
        )


def test_explicit_tool_authorization_allows_selected_skill() -> None:
    base = build_research_environment_profile(ResearchBackendSettings())
    enabled_tools = list(base.tool_profile.enabled)
    if "document.parse_preview" not in enabled_tools:
        enabled_tools.append("document.parse_preview")
    env = base.model_copy(
        update={
            "tool_profile": ToolProfile(enabled=enabled_tools),
            "skill_profile": build_research_skill_profile(
                enabled_skill_ids=(
                    *RESEARCH_DEFAULT_ENABLED_SKILL_IDS,
                    RESEARCH_SOURCE_VALIDATOR.skill_id,
                ),
            ),
        },
    )
    skill_wiring = build_application_skill_wiring(env.skill_profile)
    validate_capability_dependencies_for_environment(
        env,
        skill_registry=skill_wiring.registry,
    )
    registry = build_registry_from_profile(env.skill_profile)
    assert RESEARCH_SOURCE_VALIDATOR.skill_id in registry.skill_ids()
