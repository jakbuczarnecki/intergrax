# © Artur Czarnecki. All rights reserved.

"""P1.5 — effective capability health projection."""

from __future__ import annotations

import json

import pytest

from intergrax.applications._shared.capability_dependency import (
    SkillToolCapabilityDependencyProvider,
    validate_capability_dependencies,
)
from intergrax.applications._shared.capability_health import (
    EffectiveCapabilityHealthProjector,
    default_capability_health_providers,
    project_effective_capability_health,
    project_status_from_facts,
)
from intergrax.applications._shared.capability_health.redaction import (
    safe_effective_capability_health_view,
)
from intergrax.applications._shared.runtime_inspection import (
    RuntimeInspectionService,
    profile_contains_no_raw_secrets,
)
from intergrax.applications.contracts.capability_dependency import (
    CapabilityDependency,
    CapabilityDependencyAvailabilityStatus,
    CapabilityDependencyKind,
    CapabilityDependencyRequirement,
    CapabilityDependencyValidationContext,
)
from intergrax.applications.contracts.capability_dependency.dependency import CapabilityRef
from intergrax.applications.contracts.capability_health import (
    CapabilityHealthConditionKind,
    CapabilityHealthFact,
    CapabilityHealthFactStatus,
    CapabilityHealthProjectionContext,
    CapabilityHealthProvider,
    CapabilityHealthProviderConflictError,
    CapabilityHealthReason,
    CapabilityHealthStatus,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.runtime_inspection import InspectionCompleteness
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_RAW_SECRET = "RAW_HEALTH_SECRET_456"


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


class _CustomHealthProvider:
    def __init__(
        self,
        *,
        provider_id: str,
        source_provenance: str,
        facts: tuple[CapabilityHealthFact, ...] = (),
        fail: bool = False,
    ) -> None:
        self._provider_id = provider_id
        self._source_provenance = source_provenance
        self._facts = facts
        self._fail = fail

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def source_provenance(self) -> str:
        return self._source_provenance

    def health_facts_for(
        self,
        context: CapabilityHealthProjectionContext,
    ) -> tuple[CapabilityHealthFact, ...]:
        if self._fail:
            raise RuntimeError("provider exploded")
        return self._facts


def _fact(
    *,
    capability: CapabilityRef,
    condition_kind: CapabilityHealthConditionKind,
    condition_ref: str,
    status: CapabilityHealthFactStatus,
    blocking: bool,
    provider_id: str = "test",
    scope_application_id: str | None = None,
    scope_tenant_id: str | None = None,
) -> CapabilityHealthFact:
    return CapabilityHealthFact(
        capability=capability,
        source="test",
        condition_kind=condition_kind,
        condition_ref=condition_ref,
        scope_application_id=scope_application_id,
        scope_tenant_id=scope_tenant_id,
        status=status,
        blocking=blocking,
        reason=CapabilityHealthReason(
            reason_code=f"test.{status.value}",
            source="test",
            subject_ref=condition_ref,
            detail=f"detail for {condition_ref}",
        ),
        provider_id=provider_id,
    )


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


def _validation(
    env: ApplicationEnvironmentProfile,
    *,
    providers: tuple[object, ...] | None = None,
    registry: SkillRegistry | None = None,
):
    return validate_capability_dependencies(
        CapabilityDependencyValidationContext(
            environment_profile=env,
            skill_registry=registry,
        ),
        providers=providers,
    )


def test_pure_zero_facts_projects_unavailable() -> None:
    assert project_status_from_facts(()) is CapabilityHealthStatus.UNAVAILABLE


def test_projector_zero_facts_unavailable_with_missing_evidence_reason() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.a")
    projector = EffectiveCapabilityHealthProjector(
        (
            _CustomHealthProvider(
                provider_id="empty.plugin",
                source_provenance="empty.plugin",
                facts=(),
            ),
        ),
    )
    health = projector.project(CapabilityHealthProjectionContext(capability=capability))
    assert health.status is CapabilityHealthStatus.UNAVAILABLE
    assert any(
        item.reason_code == "capability.health.evidence_missing"
        for item in health.reasons
    )
    assert any(
        item.condition_kind is CapabilityHealthConditionKind.READINESS_EVIDENCE
        for item in health.facts
    )
    assert health.facts[0].provider_id == "capability_health_projection"


def test_unsupported_capability_kind_no_evidence_unavailable() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.orphan")
    health = EffectiveCapabilityHealthProjector(default_capability_health_providers()).project(
        CapabilityHealthProjectionContext(
            capability=capability,
            environment_profile=ApplicationEnvironmentProfile.lab_defaults(),
        ),
    )
    assert health.status is CapabilityHealthStatus.UNAVAILABLE
    assert any(
        item.reason_code == "capability.health.evidence_missing"
        for item in health.reasons
    )


def test_all_facts_satisfied_projects_ready() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.a")
    facts = (
        _fact(
            capability=capability,
            condition_kind=CapabilityHealthConditionKind.TOOL_EFFECTIVE_AVAILABILITY,
            condition_ref="tool.a",
            status=CapabilityHealthFactStatus.SATISFIED,
            blocking=True,
        ),
    )
    health = project_effective_capability_health(capability=capability, facts=facts)
    assert health.status is CapabilityHealthStatus.READY


def test_required_missing_projects_unavailable() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    facts = (
        _fact(
            capability=capability,
            condition_kind=CapabilityHealthConditionKind.DEPENDENCY_REQUIRED,
            condition_ref="tool:tool.b",
            status=CapabilityHealthFactStatus.UNSATISFIED,
            blocking=True,
        ),
    )
    assert project_status_from_facts(facts) is CapabilityHealthStatus.UNAVAILABLE


def test_required_unknown_projects_unavailable() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    facts = (
        _fact(
            capability=capability,
            condition_kind=CapabilityHealthConditionKind.DEPENDENCY_REQUIRED,
            condition_ref="tool:tool.b",
            status=CapabilityHealthFactStatus.UNKNOWN,
            blocking=True,
        ),
    )
    assert project_status_from_facts(facts) is CapabilityHealthStatus.UNAVAILABLE


def test_optional_missing_projects_degraded() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    facts = (
        _fact(
            capability=capability,
            condition_kind=CapabilityHealthConditionKind.DEPENDENCY_OPTIONAL,
            condition_ref="tool:tool.opt",
            status=CapabilityHealthFactStatus.UNSATISFIED,
            blocking=False,
        ),
    )
    assert project_status_from_facts(facts) is CapabilityHealthStatus.DEGRADED


def test_optional_unknown_projects_degraded() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    facts = (
        _fact(
            capability=capability,
            condition_kind=CapabilityHealthConditionKind.DEPENDENCY_OPTIONAL,
            condition_ref="tool:tool.opt",
            status=CapabilityHealthFactStatus.UNKNOWN,
            blocking=False,
        ),
    )
    assert project_status_from_facts(facts) is CapabilityHealthStatus.DEGRADED


def test_dominance_unavailable_over_degraded_and_ready() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.a")
    facts = (
        _fact(
            capability=capability,
            condition_kind=CapabilityHealthConditionKind.DEPENDENCY_OPTIONAL,
            condition_ref="tool:opt",
            status=CapabilityHealthFactStatus.UNSATISFIED,
            blocking=False,
        ),
        _fact(
            capability=capability,
            condition_kind=CapabilityHealthConditionKind.TOOL_EFFECTIVE_AVAILABILITY,
            condition_ref="tool.a",
            status=CapabilityHealthFactStatus.SATISFIED,
            blocking=True,
        ),
        _fact(
            capability=capability,
            condition_kind=CapabilityHealthConditionKind.DEPENDENCY_REQUIRED,
            condition_ref="tool:req",
            status=CapabilityHealthFactStatus.UNSATISFIED,
            blocking=True,
        ),
    )
    assert project_status_from_facts(facts) is CapabilityHealthStatus.UNAVAILABLE
    reversed_health = project_effective_capability_health(
        capability=capability,
        facts=tuple(reversed(facts)),
    )
    assert reversed_health.status is CapabilityHealthStatus.UNAVAILABLE


def test_duplicate_provider_id_fails_fast() -> None:
    provider_a = _CustomHealthProvider(provider_id="dup", source_provenance="a")
    provider_b = _CustomHealthProvider(provider_id="dup", source_provenance="b")
    with pytest.raises(CapabilityHealthProviderConflictError, match="dup"):
        EffectiveCapabilityHealthProjector((provider_a, provider_b))


def test_plugin_provider_contributes_without_core_modification() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.a")
    custom_fact = _fact(
        capability=capability,
        condition_kind=CapabilityHealthConditionKind.TOOL_EFFECTIVE_AVAILABILITY,
        condition_ref="custom.marker",
        status=CapabilityHealthFactStatus.DEGRADED,
        blocking=False,
        provider_id="custom.plugin",
    )
    projector = EffectiveCapabilityHealthProjector(
        (
            *_default_providers_without_tool(),
            _CustomHealthProvider(
                provider_id="custom.plugin",
                source_provenance="custom.plugin",
                facts=(custom_fact,),
            ),
        ),
    )
    health = projector.project(
        CapabilityHealthProjectionContext(
            capability=capability,
            environment_profile=ApplicationEnvironmentProfile.lab_defaults(),
        ),
    )
    assert health.status is CapabilityHealthStatus.DEGRADED
    assert any(item.provider_id == "custom.plugin" for item in health.facts)


def test_provider_failure_is_conservative_unavailable() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.a")
    projector = EffectiveCapabilityHealthProjector(
        (
            _CustomHealthProvider(
                provider_id="failing.plugin",
                source_provenance="failing.plugin",
                fail=True,
            ),
        ),
    )
    health = projector.project(CapabilityHealthProjectionContext(capability=capability))
    assert health.status is CapabilityHealthStatus.UNAVAILABLE
    assert len(health.provider_failures) == 1
    assert any(
        item.condition_kind is CapabilityHealthConditionKind.PROVIDER_FAILURE
        for item in health.facts
    )


def test_real_p1_3_required_failure_unavailable() -> None:
    env, registry = _skill_tool_env("skill.x", "tool.y", host_tools=())
    validation = _validation(env, registry=registry)
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.x")
    health = EffectiveCapabilityHealthProjector(default_capability_health_providers()).project(
        CapabilityHealthProjectionContext(
            capability=capability,
            validation=validation,
            environment_profile=env,
        ),
    )
    assert health.status is CapabilityHealthStatus.UNAVAILABLE
    assert any(
        item.condition_kind is CapabilityHealthConditionKind.DEPENDENCY_REQUIRED
        for item in health.facts
    )


def test_real_p1_3_optional_degradation_degraded() -> None:
    owner = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.x")
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
    env = ApplicationEnvironmentProfile.lab_defaults()
    validation = _validation(env, providers=(provider,))
    health = EffectiveCapabilityHealthProjector(default_capability_health_providers()).project(
        CapabilityHealthProjectionContext(
            capability=owner,
            validation=validation,
            environment_profile=env,
        ),
    )
    assert health.status is CapabilityHealthStatus.DEGRADED


def test_real_skill_tool_ready_when_tool_available() -> None:
    env, registry = _skill_tool_env("skill.x", "tool.y", host_tools=("tool.y",))
    validation = _validation(
        env,
        providers=(SkillToolCapabilityDependencyProvider(),),
        registry=registry,
    )
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.x")
    health = EffectiveCapabilityHealthProjector(default_capability_health_providers()).project(
        CapabilityHealthProjectionContext(
            capability=capability,
            validation=validation,
            environment_profile=env,
        ),
    )
    assert health.status is CapabilityHealthStatus.READY


def test_real_skill_tool_unavailable_when_tool_missing() -> None:
    env, registry = _skill_tool_env("skill.x", "tool.y", host_tools=())
    validation = _validation(
        env,
        providers=(SkillToolCapabilityDependencyProvider(),),
        registry=registry,
    )
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.x")
    health = EffectiveCapabilityHealthProjector(default_capability_health_providers()).project(
        CapabilityHealthProjectionContext(
            capability=capability,
            validation=validation,
            environment_profile=env,
        ),
    )
    assert health.status is CapabilityHealthStatus.UNAVAILABLE


def test_tool_capability_ready_when_on_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={"tool_profile": ToolProfile(enabled=["search"])},
    )
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="search")
    health = EffectiveCapabilityHealthProjector(default_capability_health_providers()).project(
        CapabilityHealthProjectionContext(
            capability=capability,
            environment_profile=env,
        ),
    )
    assert health.status is CapabilityHealthStatus.READY


def test_tool_capability_unavailable_when_not_on_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={"tool_profile": ToolProfile(enabled=["other"])},
    )
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="search")
    health = EffectiveCapabilityHealthProjector(default_capability_health_providers()).project(
        CapabilityHealthProjectionContext(
            capability=capability,
            environment_profile=env,
        ),
    )
    assert health.status is CapabilityHealthStatus.UNAVAILABLE


def test_no_authority_expansion_on_projection() -> None:
    env, registry = _skill_tool_env("skill.x", "tool.y", host_tools=("tool.y",))
    before_tool = env.tool_profile.model_dump(mode="json")
    before_skill = env.skill_profile.model_dump(mode="json")
    validation = _validation(
        env,
        providers=(SkillToolCapabilityDependencyProvider(),),
        registry=registry,
    )
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.x")
    EffectiveCapabilityHealthProjector(default_capability_health_providers()).project(
        CapabilityHealthProjectionContext(
            capability=capability,
            validation=validation,
            environment_profile=env,
        ),
    )
    assert env.tool_profile.model_dump(mode="json") == before_tool
    assert env.skill_profile.model_dump(mode="json") == before_skill
    assert validation.required_failures == ()


def test_health_not_authorization() -> None:
    env, registry = _skill_tool_env("skill.x", "tool.y", host_tools=("tool.y",))
    validation = _validation(
        env,
        providers=(SkillToolCapabilityDependencyProvider(),),
        registry=registry,
    )
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.x")
    health = EffectiveCapabilityHealthProjector(default_capability_health_providers()).project(
        CapabilityHealthProjectionContext(
            capability=capability,
            validation=validation,
            environment_profile=env,
        ),
    )
    assert health.status is CapabilityHealthStatus.READY
    assert env.governance.model_dump(mode="json") == env.governance.model_dump(mode="json")


def test_runtime_inspection_exposes_health() -> None:
    env, registry = _skill_tool_env("skill.x", "tool.y", host_tools=())
    validation = _validation(
        env,
        providers=(SkillToolCapabilityDependencyProvider(),),
        registry=registry,
    )
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.x")
    result = RuntimeInspectionService().inspect_capability(
        capability,
        validation,
        environment_profile=env,
    )
    assert result.health.status is CapabilityHealthStatus.UNAVAILABLE
    assert result.safe_health.status is CapabilityHealthStatus.UNAVAILABLE
    assert result.health.reasons


def test_inspection_completeness_not_health_status() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="search")
    validation = _validation(env)
    result = RuntimeInspectionService().inspect_capability(
        capability,
        validation,
        environment_profile=env,
    )
    assert result.completeness is InspectionCompleteness.COMPLETE
    assert result.health.status in {
        CapabilityHealthStatus.READY,
        CapabilityHealthStatus.DEGRADED,
        CapabilityHealthStatus.UNAVAILABLE,
    }


def test_safe_serialization_redacts_secret_like_reason() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.a")
    fact = CapabilityHealthFact(
        capability=capability,
        source="test",
        condition_kind=CapabilityHealthConditionKind.TOOL_EFFECTIVE_AVAILABILITY,
        condition_ref="tool.a",
        status=CapabilityHealthFactStatus.UNSATISFIED,
        blocking=True,
        reason=CapabilityHealthReason(
            reason_code="tool.availability.unsatisfied",
            source="test",
            subject_ref="tool.a",
            detail=f"api_key={_RAW_SECRET}",
        ),
        provider_id="test",
    )
    health = project_effective_capability_health(capability=capability, facts=(fact,))
    safe = safe_effective_capability_health_view(health)
    dumped = safe.model_dump(mode="json")
    assert profile_contains_no_raw_secrets(dumped, raw_secret=_RAW_SECRET)
    assert profile_contains_no_raw_secrets(safe.model_dump_json(), raw_secret=_RAW_SECRET)


def test_deterministic_output_provider_order_independent() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.a")
    fact_a = _fact(
        capability=capability,
        condition_kind=CapabilityHealthConditionKind.TOOL_EFFECTIVE_AVAILABILITY,
        condition_ref="a",
        status=CapabilityHealthFactStatus.SATISFIED,
        blocking=True,
        provider_id="provider.a",
    )
    fact_b = _fact(
        capability=capability,
        condition_kind=CapabilityHealthConditionKind.DEPENDENCY_OPTIONAL,
        condition_ref="b",
        status=CapabilityHealthFactStatus.UNSATISFIED,
        blocking=False,
        provider_id="provider.b",
    )
    projector_ab = EffectiveCapabilityHealthProjector(
        (
            _CustomHealthProvider(
                provider_id="provider.a",
                source_provenance="a",
                facts=(fact_a,),
            ),
            _CustomHealthProvider(
                provider_id="provider.b",
                source_provenance="b",
                facts=(fact_b,),
            ),
        ),
    )
    projector_ba = EffectiveCapabilityHealthProjector(
        (
            _CustomHealthProvider(
                provider_id="provider.b",
                source_provenance="b",
                facts=(fact_b,),
            ),
            _CustomHealthProvider(
                provider_id="provider.a",
                source_provenance="a",
                facts=(fact_a,),
            ),
        ),
    )
    context = CapabilityHealthProjectionContext(capability=capability)
    assert projector_ab.project(context).model_dump(
        mode="json",
    ) == projector_ba.project(context).model_dump(mode="json")


def test_tenant_scope_isolation() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.a")
    tenant_a_fact = _fact(
        capability=capability,
        condition_kind=CapabilityHealthConditionKind.TOOL_EFFECTIVE_AVAILABILITY,
        condition_ref="tool.a",
        status=CapabilityHealthFactStatus.UNSATISFIED,
        blocking=True,
        scope_tenant_id="tenant-a",
    )
    projector = EffectiveCapabilityHealthProjector(
        (
            _CustomHealthProvider(
                provider_id="tenant.facts",
                source_provenance="tenant.facts",
                facts=(tenant_a_fact,),
            ),
        ),
    )
    health_b = projector.project(
        CapabilityHealthProjectionContext(
            capability=capability,
            scope_tenant_id="tenant-b",
        ),
    )
    assert health_b.status is CapabilityHealthStatus.UNAVAILABLE
    assert any(
        item.reason_code == "capability.health.evidence_missing"
        for item in health_b.reasons
    )
    health_a = projector.project(
        CapabilityHealthProjectionContext(
            capability=capability,
            scope_tenant_id="tenant-a",
        ),
    )
    assert health_a.status is CapabilityHealthStatus.UNAVAILABLE


def test_provider_failure_no_duplicate_missing_evidence_reason() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.a")
    projector = EffectiveCapabilityHealthProjector(
        (
            _CustomHealthProvider(
                provider_id="failing.plugin",
                source_provenance="failing.plugin",
                fail=True,
            ),
        ),
    )
    health = projector.project(CapabilityHealthProjectionContext(capability=capability))
    assert health.status is CapabilityHealthStatus.UNAVAILABLE
    reason_codes = {item.reason_code for item in health.reasons}
    assert "provider.failure" in reason_codes
    assert "capability.health.evidence_missing" not in reason_codes


def test_capability_mismatch_fact_filtered_out() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.a")
    other_capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.b")
    foreign_fact = _fact(
        capability=other_capability,
        condition_kind=CapabilityHealthConditionKind.TOOL_EFFECTIVE_AVAILABILITY,
        condition_ref="tool.b",
        status=CapabilityHealthFactStatus.SATISFIED,
        blocking=True,
    )
    projector = EffectiveCapabilityHealthProjector(
        (
            _CustomHealthProvider(
                provider_id="foreign.facts",
                source_provenance="foreign.facts",
                facts=(foreign_fact,),
            ),
        ),
    )
    health = projector.project(CapabilityHealthProjectionContext(capability=capability))
    assert health.status is CapabilityHealthStatus.UNAVAILABLE
    assert any(
        item.reason_code == "capability.health.evidence_missing"
        for item in health.reasons
    )


def test_conflicting_facts_unsatisfied_dominates() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.a")
    satisfied = _fact(
        capability=capability,
        condition_kind=CapabilityHealthConditionKind.TOOL_EFFECTIVE_AVAILABILITY,
        condition_ref="tool.a",
        status=CapabilityHealthFactStatus.SATISFIED,
        blocking=True,
    )
    unsatisfied = satisfied.model_copy(
        update={"status": CapabilityHealthFactStatus.UNSATISFIED},
    )
    health = project_effective_capability_health(
        capability=capability,
        facts=(satisfied, unsatisfied),
    )
    assert health.status is CapabilityHealthStatus.UNAVAILABLE
    merged = [item for item in health.facts if item.condition_ref == "tool.a"]
    assert len(merged) == 1
    assert merged[0].status is CapabilityHealthFactStatus.UNSATISFIED


def _default_providers_without_tool() -> tuple[CapabilityHealthProvider, ...]:
    return tuple(
        provider
        for provider in default_capability_health_providers()
        if provider.provider_id != "tool_effective_availability"
    )
