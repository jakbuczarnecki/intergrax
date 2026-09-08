# © Artur Czarnecki. All rights reserved.

"""P1.10 — skill version binding, provenance, and execution pinning."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.runtime_inspection.service import RuntimeInspectionService
from intergrax.skills.registry.tool_requirements import SkillToolRequirementError
from intergrax.skills.contribution_provenance import (
    SkillContributionKind,
    build_skill_contribution_provenance,
    contributors_for,
)
from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier
from intergrax.skills.core.version_binding import (
    ResolvedSkillRole,
    SkillVersionResolutionMode,
)
from intergrax.skills.execution_binding import (
    InMemorySkillExecutionPinningStore,
    bind_resolved_skill_pack,
    binding_from_composition,
    resolve_bound_skill_pack,
)
from intergrax.skills.providers.harness.manifests import (
    HARNESS_POLICY_SMOKE,
    HARNESS_STACK_DEMO,
    HARNESS_TOOL_SMOKE,
)
from intergrax.skills.registry import SkillProfile, build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import (
    ResolvedSkillComposition,
    SkillResolutionError,
    SkillResolver,
)
from intergrax.skills.snapshot_digest import compute_resolved_skill_pack_digest
from intergrax.skills.version_validation import validate_skill_version
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.tools.registry.profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_harness_catalog() -> None:
    reset_default_skills_for_tests()
    register_default_skills()
    yield
    reset_default_skills_for_tests()


def _manifest(
    skill_id: str,
    version: str = "1.0.0",
    *,
    requires_skills: tuple[str, ...] = (),
    tool_ids: tuple[str, ...] = (),
    prompt_instruction_ids: tuple[str, ...] = (),
    policy_fragment_id: str | None = None,
    risk_tier: SkillRiskTier = SkillRiskTier.LOW,
) -> SkillManifest:
    return SkillManifest(
        skill_id=skill_id,
        description=skill_id,
        version=version,
        requires_skills=requires_skills,
        tool_ids=tool_ids,
        prompt_instruction_ids=prompt_instruction_ids,
        policy_fragment_id=policy_fragment_id,
        risk_tier=risk_tier,
    )


def _harness_registry() -> SkillRegistry:
    return build_registry_from_profile(SkillProfile(enabled_bundles=["harness"]))


def test_version_validation_rejects_sentinels() -> None:
    with pytest.raises(ValueError, match="explicit"):
        validate_skill_version("latest")
    with pytest.raises(ValueError, match="explicit"):
        validate_skill_version("unknown")
    assert validate_skill_version("2.3.4-dev") == "2.3.4-dev"


def test_manifest_version_validation_rejects_latest() -> None:
    with pytest.raises(ValueError, match="explicit"):
        SkillManifest(skill_id="a.pack", description="x", version="latest")


def test_root_replacement_e1_stays_e2_gets_new() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.x",)))
    resolver = SkillResolver(registry)
    e1_pack = resolver.resolve_composition(["a.pack"])
    pinning = InMemorySkillExecutionPinningStore()
    e1_id = mint_execution_id()
    e2_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=e1_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=e1_pack,
    )
    registry.register_or_replace(_manifest("a.pack", "2.0.0", tool_ids=("tool.y",)))
    e2_pack = resolver.resolve_composition(["a.pack"])
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=e2_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=e2_pack,
    )
    assert pinning.get(tenant_id="tenant-a", execution_id=e1_id).resolved_pack.snapshot_digest != (
        pinning.get(tenant_id="tenant-a", execution_id=e2_id).resolved_pack.snapshot_digest
    )
    assert pinning.get(tenant_id="tenant-a", execution_id=e1_id).resolved_pack.resolved_skills[0].version == "1.0.0"
    assert pinning.get(tenant_id="tenant-a", execution_id=e2_id).resolved_pack.resolved_skills[0].version == "2.0.0"


def test_transitive_replacement_e1_stays_b1_e2_gets_b2() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("b.pack", "1.0.0", tool_ids=("tool.b",)))
    registry.register(
        _manifest("a.pack", "1.0.0", requires_skills=("b.pack",), tool_ids=("tool.a",)),
    )
    e1_composition = SkillResolver(registry).resolve_composition(["a.pack"])
    pinning = InMemorySkillExecutionPinningStore()
    e1_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=e1_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=e1_composition,
    )
    registry.register_or_replace(_manifest("b.pack", "2.0.0", tool_ids=("tool.b2",)))
    e2_composition = SkillResolver(registry).resolve_composition(["a.pack"])
    e2_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=e2_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=e2_composition,
    )
    e1_refs = pinning.get(tenant_id="tenant-a", execution_id=e1_id).resolved_pack.resolved_skills
    e2_refs = pinning.get(tenant_id="tenant-a", execution_id=e2_id).resolved_pack.resolved_skills
    e1_b = next(ref for ref in e1_refs if ref.skill_id == "b.pack")
    e2_b = next(ref for ref in e2_refs if ref.skill_id == "b.pack")
    assert e1_b.version == "1.0.0"
    assert e2_b.version == "2.0.0"


def test_unregister_does_not_alter_e1_new_resolve_fails() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    pinning = InMemorySkillExecutionPinningStore()
    e1_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=e1_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    registry.unregister("a.pack")
    assert pinning.get(tenant_id="tenant-a", execution_id=e1_id) is not None
    with pytest.raises(SkillResolutionError, match="Unknown skill_id"):
        SkillResolver(registry).resolve(["a.pack"])


def test_no_downstream_re_resolution_after_bind() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.x",)))
    profile = SkillProfile(enabled=["a.pack"])
    pinning = InMemorySkillExecutionPinningStore()
    e1_id = mint_execution_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=e1_id,
    )
    try:
        get_calls = 0
        original_get = registry.get

        def counting_get(skill_id: str):
            nonlocal get_calls
            get_calls += 1
            return original_get(skill_id)

        spy_registry = MagicMock(wraps=registry)
        spy_registry.get = counting_get
        spy_registry.has = registry.has
        spy_registry.skill_ids = registry.skill_ids
        spy_registry.list = registry.list
        first = resolve_bound_skill_pack(
            tenant_id="tenant-a",
            skill_profile=profile,
            skill_registry=spy_registry,
            pinning_store=pinning,
        )
        get_calls = 0
        second = resolve_bound_skill_pack(
            tenant_id="tenant-a",
            skill_profile=profile,
            skill_registry=spy_registry,
            pinning_store=pinning,
        )
        assert first.snapshot_digest == second.snapshot_digest
        assert get_calls == 0
    finally:
        reset_active_execution_identity(token)


def test_contribution_lineage_exact_qualified_ids() -> None:
    registry = SkillRegistry()
    registry.register(
        _manifest(
            "b.pack",
            "2.0.0",
            tool_ids=("tool.x",),
            prompt_instruction_ids=("prompt.p",),
            policy_fragment_id="policy.g",
        ),
    )
    registry.register(
        _manifest(
            "a.pack",
            "1.0.0",
            requires_skills=("b.pack",),
            tool_ids=("tool.a",),
        ),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    manifests = composition.manifest_by_skill_id()
    provenance = build_skill_contribution_provenance(composition.pack, manifests)
    assert contributors_for(
        provenance,
        contribution_kind=SkillContributionKind.TOOL_REQUIREMENT,
        contribution_id="tool.x",
    ) == ("b.pack@2.0.0",)
    assert contributors_for(
        provenance,
        contribution_kind=SkillContributionKind.PROMPT_INSTRUCTION,
        contribution_id="prompt.p",
    ) == ("b.pack@2.0.0",)
    assert contributors_for(
        provenance,
        contribution_kind=SkillContributionKind.POLICY_FRAGMENT,
        contribution_id="policy.g",
    ) == ("b.pack@2.0.0",)


def test_duplicate_tool_lineage_retains_both_skills() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.x",)))
    registry.register(_manifest("b.pack", "1.0.0", tool_ids=("tool.x",)))
    composition = SkillResolver(registry).resolve_composition(["a.pack", "b.pack"])
    provenance = build_skill_contribution_provenance(
        composition.pack,
        composition.manifest_by_skill_id(),
    )
    assert set(
        contributors_for(
            provenance,
            contribution_kind=SkillContributionKind.TOOL_REQUIREMENT,
            contribution_id="tool.x",
        ),
    ) == {"a.pack@1.0.0", "b.pack@1.0.0"}


def test_digest_proofs() -> None:
    ref_a1 = _manifest("a.pack", "1.0.0")
    ref_a2 = _manifest("a.pack", "2.0.0")
    registry = SkillRegistry()
    registry.register(ref_a1)
    pack1 = SkillResolver(registry).resolve(["a.pack"])
    pack1b = SkillResolver(registry).resolve(["a.pack"])
    assert pack1.snapshot_digest == pack1b.snapshot_digest
    registry.register_or_replace(ref_a2)
    pack2 = SkillResolver(registry).resolve(["a.pack"])
    assert pack1.snapshot_digest != pack2.snapshot_digest

    registry = SkillRegistry()
    registry.register(_manifest("b.pack", "1.0.0"))
    registry.register(_manifest("a.pack", "1.0.0", requires_skills=("b.pack",)))
    pack_t1 = SkillResolver(registry).resolve(["a.pack"])
    registry.register_or_replace(_manifest("b.pack", "2.0.0"))
    pack_t2 = SkillResolver(registry).resolve(["a.pack"])
    assert pack_t1.snapshot_digest != pack_t2.snapshot_digest


def test_tool_authority_not_widened_by_skill_requirement() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("forbidden.tool",)))
    profile = SkillProfile(enabled=["a.pack"])
    tool_profile = ToolProfile(enabled=["allowed.tool"])
    with pytest.raises(SkillToolRequirementError) as exc_info:
        from intergrax.applications._shared.skill_tool_profile import (
            assert_skill_tool_requirements_for_profile,
        )

        assert_skill_tool_requirements_for_profile(
            tool_profile,
            profile,
            skill_registry=registry,
        )
    assert "forbidden.tool" in exc_info.value.resolution.missing_tool_ids


def test_real_harness_stack_demo_transitive_adoption() -> None:
    registry = _harness_registry()
    pack = SkillResolver(registry).resolve([HARNESS_STACK_DEMO.skill_id])
    assert HARNESS_TOOL_SMOKE.skill_id in pack.skill_ids
    assert HARNESS_STACK_DEMO.skill_id in pack.skill_ids
    stack_ref = next(ref for ref in pack.resolved_skills if ref.skill_id == HARNESS_STACK_DEMO.skill_id)
    smoke_ref = next(ref for ref in pack.resolved_skills if ref.skill_id == HARNESS_TOOL_SMOKE.skill_id)
    assert stack_ref.role is ResolvedSkillRole.ROOT
    assert smoke_ref.role is ResolvedSkillRole.TRANSITIVE
    assert smoke_ref.resolution_mode is SkillVersionResolutionMode.MATERIALIZED


def test_real_harness_policy_fragment_provenance() -> None:
    registry = _harness_registry()
    composition = SkillResolver(registry).resolve_composition([HARNESS_POLICY_SMOKE.skill_id])
    provenance = build_skill_contribution_provenance(
        composition.pack,
        composition.manifest_by_skill_id(),
    )
    assert contributors_for(
        provenance,
        contribution_kind=SkillContributionKind.POLICY_FRAGMENT,
        contribution_id="harness.policy_smoke",
    ) == (f"{HARNESS_POLICY_SMOKE.skill_id}@{HARNESS_POLICY_SMOKE.version}",)


def test_inspection_stable_after_registry_mutation() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.x",)))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    pinning = InMemorySkillExecutionPinningStore()
    e1_id = mint_execution_id()
    e2_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=e1_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    service = RuntimeInspectionService(skill_pinning_store=pinning)
    before = service.inspect_execution(tenant_id="tenant-a", execution_id=e1_id, scope_application_id="app")
    registry.register_or_replace(_manifest("a.pack", "9.9.9", tool_ids=("tool.z",)))
    e2_composition = SkillResolver(registry).resolve_composition(["a.pack"])
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=e2_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=e2_composition,
    )
    after_e1 = service.inspect_execution(tenant_id="tenant-a", execution_id=e1_id, scope_application_id="app")
    after_e2 = service.inspect_execution(tenant_id="tenant-a", execution_id=e2_id, scope_application_id="app")
    d1_before = next(
        item.payload["skill_pack_digest"]
        for item in before.extension_evidence
        if item.subject == "skill_pack"
    )
    d1_after = next(
        item.payload["skill_pack_digest"]
        for item in after_e1.extension_evidence
        if item.subject == "skill_pack"
    )
    d2 = next(
        item.payload["skill_pack_digest"]
        for item in after_e2.extension_evidence
        if item.subject == "skill_pack"
    )
    assert d1_before == d1_after
    assert d1_before != d2


def test_configured_vs_effective_distinction_in_binding() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("b.pack", "1.0.0"))
    registry.register(_manifest("a.pack", "1.0.0", requires_skills=("b.pack",)))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    binding = binding_from_composition(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        composition=composition,
    )
    assert binding.configured_skill_ids == ("a.pack",)
    assert set(binding.resolved_pack.skill_ids) == {"a.pack", "b.pack"}


def test_root_replacement_race_coherent_provenance() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.old",)))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    registry.register_or_replace(_manifest("a.pack", "2.0.0", tool_ids=("tool.new",)))
    binding = binding_from_composition(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        composition=composition,
    )
    assert binding.resolved_pack.resolved_skills[0].version == "1.0.0"
    assert contributors_for(
        binding.contribution_provenance,
        contribution_kind=SkillContributionKind.TOOL_REQUIREMENT,
        contribution_id="tool.old",
    ) == ("a.pack@1.0.0",)
    assert contributors_for(
        binding.contribution_provenance,
        contribution_kind=SkillContributionKind.TOOL_REQUIREMENT,
        contribution_id="tool.new",
    ) == ()


def test_transitive_replacement_race_coherent_provenance() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("b.pack", "1.0.0", tool_ids=("tool.old",)))
    registry.register(
        _manifest("a.pack", "1.0.0", requires_skills=("b.pack",), tool_ids=("tool.a",)),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    registry.register_or_replace(_manifest("b.pack", "2.0.0", tool_ids=("tool.new",)))
    binding = binding_from_composition(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        composition=composition,
    )
    b_ref = next(ref for ref in binding.resolved_pack.resolved_skills if ref.skill_id == "b.pack")
    assert b_ref.version == "1.0.0"
    assert contributors_for(
        binding.contribution_provenance,
        contribution_kind=SkillContributionKind.TOOL_REQUIREMENT,
        contribution_id="tool.old",
    ) == ("b.pack@1.0.0",)
    assert contributors_for(
        binding.contribution_provenance,
        contribution_kind=SkillContributionKind.TOOL_REQUIREMENT,
        contribution_id="tool.new",
    ) == ()


def test_composition_rejects_missing_manifest_for_ref() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    with pytest.raises(SkillResolutionError, match="missing observed manifest"):
        ResolvedSkillComposition(
            pack=composition.pack,
            observed_manifests=(),
        )


def test_composition_rejects_version_mismatch() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    with pytest.raises(SkillResolutionError, match="manifest version mismatch"):
        ResolvedSkillComposition(
            pack=composition.pack,
            observed_manifests=(_manifest("a.pack", "2.0.0"),),
        )


def test_composition_rejects_duplicate_manifest_for_same_skill_id() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0"))
    registry.register(_manifest("b.pack", "1.0.0"))
    composition = SkillResolver(registry).resolve_composition(["a.pack", "b.pack"])
    duplicate = _manifest("a.pack", "1.0.0")
    with pytest.raises(SkillResolutionError, match="duplicate observed manifest"):
        ResolvedSkillComposition(
            pack=composition.pack,
            observed_manifests=(duplicate, duplicate),
        )


def test_provenance_rejects_missing_manifest() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.x",)))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    with pytest.raises(SkillResolutionError, match="missing observed manifest"):
        build_skill_contribution_provenance(composition.pack, {})


def test_provenance_rejects_version_mismatch() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.x",)))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    mismatched = {
        "a.pack": _manifest("a.pack", "2.0.0", tool_ids=("tool.y",)),
    }
    with pytest.raises(SkillResolutionError, match="manifest identity mismatch"):
        build_skill_contribution_provenance(composition.pack, mismatched)


def test_execution_binding_rejects_rebind_with_different_pack() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.x",)))
    pinning = InMemorySkillExecutionPinningStore()
    execution_id = mint_execution_id()
    profile = SkillProfile(enabled=["a.pack"])
    first = SkillResolver(registry).resolve_composition(["a.pack"])
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=profile,
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=first,
    )
    registry.register_or_replace(_manifest("a.pack", "2.0.0", tool_ids=("tool.y",)))
    second = SkillResolver(registry).resolve_composition(["a.pack"])
    with pytest.raises(SkillResolutionError, match="already pinned"):
        bind_resolved_skill_pack(
            tenant_id="tenant-a",
            execution_id=execution_id,
            skill_profile=profile,
            skill_registry=registry,
            pinning_store=pinning,
            resolved_composition=second,
        )


def test_bind_does_not_reread_registry_for_provenance() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.old",)))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    registry.register_or_replace(_manifest("a.pack", "2.0.0", tool_ids=("tool.new",)))
    get_calls = 0
    original_get = registry.get

    def counting_get(skill_id: str):
        nonlocal get_calls
        get_calls += 1
        return original_get(skill_id)

    spy_registry = MagicMock(wraps=registry)
    spy_registry.get = counting_get
    spy_registry.has = registry.has
    spy_registry.skill_ids = registry.skill_ids
    spy_registry.list = registry.list
    pinning = InMemorySkillExecutionPinningStore()
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=spy_registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    assert get_calls == 0
    assert contributors_for(
        binding.contribution_provenance,
        contribution_kind=SkillContributionKind.TOOL_REQUIREMENT,
        contribution_id="tool.old",
    ) == ("a.pack@1.0.0",)
