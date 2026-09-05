# © Artur Czarnecki. All rights reserved.

"""P1.3 — capability dependency validation."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.capability_dependency import (
    CapabilityDependencyValidator,
    SkillToolCapabilityDependencyProvider,
    enrich_profile_resolution_with_capability_dependencies,
    validate_capability_dependencies,
    validate_capability_dependencies_for_environment,
)
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.profile_resolution import resolve_profile
from intergrax.applications.contracts.capability_dependency import (
    CapabilityDependency,
    CapabilityDependencyAvailabilityStatus,
    CapabilityDependencyKind,
    CapabilityDependencyProvider,
    CapabilityDependencyProviderConflictError,
    CapabilityDependencyRequirement,
    CapabilityDependencyValidationContext,
    RequiredCapabilityDependencyUnavailableError,
)
from intergrax.applications.contracts.capability_dependency.dependency import CapabilityRef
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _SyntheticDependencyProvider:
    def __init__(
        self,
        *,
        provider_id: str | None = None,
        source_domain: str,
        declarations: tuple[CapabilityDependency, ...],
        availability: dict[tuple[str, str, str], tuple[CapabilityDependencyAvailabilityStatus, str]],
    ) -> None:
        self._provider_id = provider_id or source_domain
        self._source_domain = source_domain
        self._declarations = declarations
        self._availability = availability

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def source_domain(self) -> str:
        return self._source_domain

    def dependencies_for(
        self,
        context: CapabilityDependencyValidationContext,
    ) -> tuple[CapabilityDependency, ...]:
        del context
        return self._declarations

    def evaluate_availability(
        self,
        dependency: CapabilityDependency,
        context: CapabilityDependencyValidationContext,
    ) -> tuple[CapabilityDependencyAvailabilityStatus, str]:
        del context
        return self._availability[dependency.dedup_key]


def _env(
    *,
    tools: tuple[str, ...] = (),
    skills: tuple[str, ...] = (),
    registry: SkillRegistry | None = None,
) -> tuple[ApplicationEnvironmentProfile, SkillRegistry | None]:
    profile = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "tool_profile": ToolProfile(enabled=list(tools)),
            "skill_profile": SkillProfile(enabled=list(skills)),
        },
    )
    return profile, registry


def _skill_tool_env(
    skill_id: str,
    tool_id: str,
    *,
    host_tools: tuple[str, ...],
) -> tuple[ApplicationEnvironmentProfile, SkillRegistry]:
    registry = SkillRegistry()
    registry.register(
        SkillManifest(skill_id=skill_id, description=skill_id, tool_ids=(tool_id,)),
    )
    profile = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "tool_profile": ToolProfile(enabled=list(host_tools)),
            "skill_profile": SkillProfile(enabled=[skill_id]),
        },
    )
    return profile, registry


def _same_edge_declaration(
    *,
    source_domain: str,
    requirement: CapabilityDependencyRequirement = CapabilityDependencyRequirement.REQUIRED,
) -> CapabilityDependency:
    return CapabilityDependency(
        owner=CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a"),
        dependency=CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.b"),
        requirement=requirement,
        source_domains=(source_domain,),
    )


def test_required_dependency_available_happy_path() -> None:
    owner = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    dependency = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.b")
    declaration = CapabilityDependency(
        owner=owner,
        dependency=dependency,
        requirement=CapabilityDependencyRequirement.REQUIRED,
        source_domains=("synthetic",),
    )
    provider = _SyntheticDependencyProvider(
        source_domain="synthetic",
        declarations=(declaration,),
        availability={
            declaration.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "ok",
            ),
        },
    )
    env, _ = _env()
    result = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=env),
        providers=(provider,),
    )
    assert result.available is True
    assert result.required_failures == ()
    assert result.optional_degradations == ()


def test_required_dependency_missing_fails_closed() -> None:
    owner = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    dependency = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.missing")
    declaration = CapabilityDependency(
        owner=owner,
        dependency=dependency,
        requirement=CapabilityDependencyRequirement.REQUIRED,
        source_domains=("synthetic",),
    )
    provider = _SyntheticDependencyProvider(
        source_domain="synthetic",
        declarations=(declaration,),
        availability={
            declaration.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNAVAILABLE,
                "missing",
            ),
        },
    )
    env, _ = _env()
    result = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=env),
        providers=(provider,),
    )
    assert result.available is False
    failure = result.required_failures[0]
    assert failure.owner == owner
    assert failure.dependency == dependency
    assert failure.requirement is CapabilityDependencyRequirement.REQUIRED
    assert failure.status is CapabilityDependencyAvailabilityStatus.UNAVAILABLE


def test_optional_dependency_missing_degrades_without_blocking() -> None:
    owner = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    dependency = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.opt")
    declaration = CapabilityDependency(
        owner=owner,
        dependency=dependency,
        requirement=CapabilityDependencyRequirement.OPTIONAL,
        source_domains=("synthetic",),
    )
    provider = _SyntheticDependencyProvider(
        source_domain="synthetic",
        declarations=(declaration,),
        availability={
            declaration.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNAVAILABLE,
                "optional missing",
            ),
        },
    )
    env, _ = _env()
    result = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=env),
        providers=(provider,),
    )
    assert result.available is True
    assert result.degraded is True
    assert len(result.optional_degradations) == 1


def test_required_unknown_fails_closed_optional_unknown_degrades() -> None:
    owner = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    required = CapabilityDependency(
        owner=owner,
        dependency=CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.req"),
        requirement=CapabilityDependencyRequirement.REQUIRED,
        source_domains=("synthetic",),
    )
    optional = CapabilityDependency(
        owner=owner,
        dependency=CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.opt"),
        requirement=CapabilityDependencyRequirement.OPTIONAL,
        source_domains=("synthetic",),
    )
    provider = _SyntheticDependencyProvider(
        source_domain="synthetic",
        declarations=(required, optional),
        availability={
            required.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNKNOWN,
                "cannot determine",
            ),
            optional.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNKNOWN,
                "cannot determine optional",
            ),
        },
    )
    env, _ = _env()
    result = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=env),
        providers=(provider,),
    )
    assert result.available is False
    assert result.required_failures[0].status is CapabilityDependencyAvailabilityStatus.UNKNOWN
    assert result.optional_degradations[0].status is CapabilityDependencyAvailabilityStatus.UNKNOWN


def test_custom_provider_participates_without_core_changes() -> None:
    owner = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="plugin.skill")
    dependency = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="plugin.tool")
    declaration = CapabilityDependency(
        owner=owner,
        dependency=dependency,
        requirement=CapabilityDependencyRequirement.REQUIRED,
        source_domains=("plugin.domain",),
    )
    plugin = _SyntheticDependencyProvider(
        source_domain="plugin.domain",
        declarations=(declaration,),
        availability={
            declaration.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "plugin ok",
            ),
        },
    )
    env, _ = _env()
    result = CapabilityDependencyValidator(
        (SkillToolCapabilityDependencyProvider(), plugin),
    ).validate(CapabilityDependencyValidationContext(environment_profile=env))
    assert any("plugin.domain" in item.source_domains for item in result.declarations)


def test_duplicate_declarations_deduplicate_deterministically() -> None:
    owner = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    dependency = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.b")
    declaration = CapabilityDependency(
        owner=owner,
        dependency=dependency,
        requirement=CapabilityDependencyRequirement.REQUIRED,
        source_domains=("synthetic",),
    )
    provider = _SyntheticDependencyProvider(
        source_domain="synthetic",
        declarations=(declaration, declaration),
        availability={
            declaration.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "ok",
            ),
        },
    )
    env, _ = _env()
    result = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=env),
        providers=(provider,),
    )
    assert len(result.declarations) == 1
    first = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=env),
        providers=(provider,),
    )
    second = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=env),
        providers=(provider,),
    )
    assert first == second


def test_required_dominates_optional_on_same_edge() -> None:
    owner = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    dependency = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.b")
    required = CapabilityDependency(
        owner=owner,
        dependency=dependency,
        requirement=CapabilityDependencyRequirement.REQUIRED,
        source_domains=("synthetic",),
    )
    optional = CapabilityDependency(
        owner=owner,
        dependency=dependency,
        requirement=CapabilityDependencyRequirement.OPTIONAL,
        source_domains=("synthetic",),
    )
    provider = _SyntheticDependencyProvider(
        source_domain="synthetic",
        declarations=(optional, required),
        availability={
            required.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNAVAILABLE,
                "missing",
            ),
        },
    )
    env, _ = _env()
    result = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=env),
        providers=(provider,),
    )
    merged = result.declarations[0]
    assert merged.requirement is CapabilityDependencyRequirement.REQUIRED
    assert result.available is False


def test_skill_requires_tool_unavailable_blocks_skill_capability() -> None:
    profile, registry = _skill_tool_env(
        "research.web_evidence",
        "websearch.query",
        host_tools=("rag.retrieve",),
    )
    result = validate_capability_dependencies(
        CapabilityDependencyValidationContext(
            environment_profile=profile,
            skill_registry=registry,
        ),
        providers=(SkillToolCapabilityDependencyProvider(),),
    )
    assert result.available is False
    failure = result.required_failures[0]
    assert failure.owner.capability_id == "research.web_evidence"
    assert failure.dependency.capability_id == "websearch.query"
    assert failure.source_domain == "skill_tool_contract"
    assert failure.source_domains == ("skill_tool_contract",)


def test_skill_requirements_do_not_expand_host_tool_profile() -> None:
    profile, registry = _skill_tool_env(
        "skill.missing",
        "tool.b",
        host_tools=("tool.a",),
    )
    original_enabled = list(profile.tool_profile.enabled)
    validate_capability_dependencies(
        CapabilityDependencyValidationContext(
            environment_profile=profile,
            skill_registry=registry,
        ),
        providers=(SkillToolCapabilityDependencyProvider(),),
    )
    assert profile.tool_profile.enabled == original_enabled
    assert "tool.b" not in profile.tool_profile.enabled


def test_wire_application_environment_blocks_before_meaningful_work() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "tool_profile": ToolProfile(enabled=["rag.retrieve"]),
            "skill_profile": SkillProfile(enabled_bundles=["legal"]),
        },
    )
    manifest = ApplicationManifest(
        app_id="skill_tool_guard",
        name="Skill Tool Guard",
        route_prefix="/v1/skill_tool_guard",
        env_prefix="SKILL_TOOL_GUARD_",
        agents=[],
    )
    with pytest.raises(RequiredCapabilityDependencyUnavailableError) as exc_info:
        wire_application_environment(manifest, env, conformance_check=False)
    assert exc_info.value.result.required_failures


def test_profile_resolution_carries_dependency_evidence() -> None:
    profile, registry = _skill_tool_env(
        "research.web_evidence",
        "websearch.query",
        host_tools=("rag.retrieve",),
    )
    resolution = enrich_profile_resolution_with_capability_dependencies(
        resolve_profile(profile),
        skill_registry=registry,
    )
    assert resolution.dependency_failures
    assert resolution.dependency_failures[0].dependency_id == "websearch.query"
    assert resolution.dependency_failures[0].source_domain == "skill_tool_contract"


def test_optional_degradation_does_not_block_unrelated_capability() -> None:
    owner_a = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    owner_b = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.b")
    optional = CapabilityDependency(
        owner=owner_a,
        dependency=CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.opt"),
        requirement=CapabilityDependencyRequirement.OPTIONAL,
        source_domains=("synthetic",),
    )
    required_b = CapabilityDependency(
        owner=owner_b,
        dependency=CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.ok"),
        requirement=CapabilityDependencyRequirement.REQUIRED,
        source_domains=("synthetic",),
    )
    provider = _SyntheticDependencyProvider(
        source_domain="synthetic",
        declarations=(optional, required_b),
        availability={
            optional.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNAVAILABLE,
                "optional missing",
            ),
            required_b.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "ok",
            ),
        },
    )
    env, _ = _env()
    result = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=env),
        providers=(provider,),
    )
    assert result.available is True
    outcome_by_owner = {item.owner.capability_id: item for item in result.outcomes}
    assert outcome_by_owner["skill.a"].degraded is True
    assert outcome_by_owner["skill.b"].available is True
    assert outcome_by_owner["skill.b"].degraded is False


def test_validate_capability_dependencies_for_environment_raises_on_required_missing() -> None:
    profile, registry = _skill_tool_env(
        "skill.x",
        "tool.missing",
        host_tools=(),
    )
    with pytest.raises(RequiredCapabilityDependencyUnavailableError):
        validate_capability_dependencies_for_environment(
            profile,
            skill_registry=registry,
        )


def test_one_hop_scope_no_transitive_cycle_traversal() -> None:
    """P1.3 validates declared edges only — no recursive graph walk."""
    declarations = (
        CapabilityDependency(
            owner=CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a"),
            dependency=CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.a"),
            requirement=CapabilityDependencyRequirement.REQUIRED,
            source_domains=("synthetic",),
        ),
    )
    provider = _SyntheticDependencyProvider(
        source_domain="synthetic",
        declarations=declarations,
        availability={
            declarations[0].dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "one hop only",
            ),
        },
    )
    env, _ = _env()
    result = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=env),
        providers=(provider,),
    )
    assert len(result.evaluations) == 1


def test_merged_provenance_does_not_create_synthetic_unknown() -> None:
    declaration_a = _same_edge_declaration(source_domain="domain_a")
    declaration_b = _same_edge_declaration(source_domain="domain_b")
    provider_a = _SyntheticDependencyProvider(
        source_domain="domain_a",
        declarations=(declaration_a,),
        availability={
            declaration_a.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "available from a",
            ),
        },
    )
    provider_b = _SyntheticDependencyProvider(
        source_domain="domain_b",
        declarations=(declaration_b,),
        availability={
            declaration_b.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "available from b",
            ),
        },
    )
    env, _ = _env()
    result = CapabilityDependencyValidator(
        (provider_a, provider_b),
    ).validate(CapabilityDependencyValidationContext(environment_profile=env))
    assert result.available is True
    assert result.required_failures == ()
    assert len(result.declarations) == 1
    assert result.declarations[0].source_domains == ("domain_a", "domain_b")
    assert result.evaluations[0].status is CapabilityDependencyAvailabilityStatus.AVAILABLE


def test_required_dominates_optional_across_providers_with_valid_evaluation() -> None:
    required = _same_edge_declaration(
        source_domain="domain_a",
        requirement=CapabilityDependencyRequirement.REQUIRED,
    )
    optional = _same_edge_declaration(
        source_domain="domain_b",
        requirement=CapabilityDependencyRequirement.OPTIONAL,
    )
    provider_a = _SyntheticDependencyProvider(
        source_domain="domain_a",
        declarations=(required,),
        availability={
            required.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "required available",
            ),
        },
    )
    provider_b = _SyntheticDependencyProvider(
        source_domain="domain_b",
        declarations=(optional,),
        availability={
            optional.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "optional available",
            ),
        },
    )
    env, _ = _env()
    result = CapabilityDependencyValidator(
        (provider_a, provider_b),
    ).validate(CapabilityDependencyValidationContext(environment_profile=env))
    merged = result.declarations[0]
    assert merged.requirement is CapabilityDependencyRequirement.REQUIRED
    assert merged.source_domains == ("domain_a", "domain_b")
    assert result.available is True
    assert result.required_failures == ()
    assert result.evaluations[0].status is CapabilityDependencyAvailabilityStatus.AVAILABLE


def test_same_required_edge_from_multiple_sources_retains_provenance() -> None:
    declaration_a = _same_edge_declaration(source_domain="domain_a")
    declaration_b = _same_edge_declaration(source_domain="domain_b")
    provider_a = _SyntheticDependencyProvider(
        source_domain="domain_a",
        declarations=(declaration_a,),
        availability={
            declaration_a.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "from a",
            ),
        },
    )
    provider_b = _SyntheticDependencyProvider(
        source_domain="domain_b",
        declarations=(declaration_b,),
        availability={
            declaration_b.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "from b",
            ),
        },
    )
    env, _ = _env()
    result = CapabilityDependencyValidator(
        (provider_a, provider_b),
    ).validate(CapabilityDependencyValidationContext(environment_profile=env))
    assert len(result.declarations) == 1
    assert result.declarations[0].source_domains == ("domain_a", "domain_b")
    assert result.declarations[0].requirement is CapabilityDependencyRequirement.REQUIRED


def test_duplicate_provider_identity_fails_closed() -> None:
    provider_a = _SyntheticDependencyProvider(
        provider_id="shared.id",
        source_domain="domain_a",
        declarations=(),
        availability={},
    )
    provider_b = _SyntheticDependencyProvider(
        provider_id="shared.id",
        source_domain="domain_b",
        declarations=(),
        availability={},
    )
    with pytest.raises(CapabilityDependencyProviderConflictError) as exc_info:
        CapabilityDependencyValidator((provider_a, provider_b))
    assert exc_info.value.provider_id == "shared.id"


def test_provider_order_does_not_change_semantic_result() -> None:
    declaration_a = _same_edge_declaration(source_domain="domain_a")
    declaration_b = _same_edge_declaration(source_domain="domain_b")
    provider_a = _SyntheticDependencyProvider(
        source_domain="domain_a",
        declarations=(declaration_a,),
        availability={
            declaration_a.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "from a",
            ),
        },
    )
    provider_b = _SyntheticDependencyProvider(
        source_domain="domain_b",
        declarations=(declaration_b,),
        availability={
            declaration_b.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "from b",
            ),
        },
    )
    env, _ = _env()
    context = CapabilityDependencyValidationContext(environment_profile=env)
    first = CapabilityDependencyValidator((provider_a, provider_b)).validate(context)
    second = CapabilityDependencyValidator((provider_b, provider_a)).validate(context)
    assert first == second


def test_evaluation_disagreement_prefers_unavailable_over_available() -> None:
    declaration_a = _same_edge_declaration(source_domain="domain_a")
    declaration_b = _same_edge_declaration(source_domain="domain_b")
    provider_a = _SyntheticDependencyProvider(
        source_domain="domain_a",
        declarations=(declaration_a,),
        availability={
            declaration_a.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "available from a",
            ),
        },
    )
    provider_b = _SyntheticDependencyProvider(
        source_domain="domain_b",
        declarations=(declaration_b,),
        availability={
            declaration_b.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNAVAILABLE,
                "missing from b",
            ),
        },
    )
    env, _ = _env()
    result = CapabilityDependencyValidator(
        (provider_a, provider_b),
    ).validate(CapabilityDependencyValidationContext(environment_profile=env))
    assert result.available is False
    assert result.evaluations[0].status is CapabilityDependencyAvailabilityStatus.UNAVAILABLE
    assert "domain_a" in result.evaluations[0].reason
    assert "domain_b" in result.evaluations[0].reason


def test_profile_resolution_multi_source_provenance_is_truthful() -> None:
    declaration_a = _same_edge_declaration(source_domain="domain_a")
    declaration_b = _same_edge_declaration(source_domain="domain_b")
    provider_a = _SyntheticDependencyProvider(
        source_domain="domain_a",
        declarations=(declaration_a,),
        availability={
            declaration_a.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNAVAILABLE,
                "missing from a",
            ),
        },
    )
    provider_b = _SyntheticDependencyProvider(
        source_domain="domain_b",
        declarations=(declaration_b,),
        availability={
            declaration_b.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNAVAILABLE,
                "missing from b",
            ),
        },
    )
    env, _ = _env()
    validation = CapabilityDependencyValidator(
        (provider_a, provider_b),
    ).validate(CapabilityDependencyValidationContext(environment_profile=env))
    resolution = enrich_profile_resolution_with_capability_dependencies(
        resolve_profile(env),
        providers=(provider_a, provider_b),
    )
    assert resolution.dependency_failures
    assert resolution.dependency_failures[0].source_domain == "domain_a, domain_b"
    assert validation.required_failures[0].source_domains == ("domain_a", "domain_b")
