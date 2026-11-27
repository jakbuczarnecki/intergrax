# © Artur Czarnecki. All rights reserved.

"""EC-W3-07 reference host policy fragment declarations and HITL symmetry."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from intergrax.applications._shared.governance_permission_preset import (
    expand_governance_permission_preset,
)
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications._shared.skill_bridge_wiring import (
    SkillBridgeCompositionError,
    merge_skill_policy_fragments,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.presets import (
    HARNESS_REFERENCE_POLICY_FRAGMENT_ID,
    LEGAL_REFERENCE_POLICY_FRAGMENT_ID,
    reference_harness_policy_fragment_payload,
    reference_legal_contract_review_policy_fragment_payload,
)
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.skills.instruction_policy_pins import (
    collect_skill_policy_fragment_pins,
    verify_pinned_policy_fragments_on_resume,
)
from intergrax.skills.providers.harness.manifests import HARNESS_POLICY_SMOKE
from intergrax.skills.providers.legal.manifests import LEGAL_CONTRACT_REVIEW
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolver

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _runtime_fragments(profile: ApplicationEnvironmentProfile) -> dict:
    return profile.domain_policy_fragments


def test_lab_defaults_declares_both_reference_fragments() -> None:
    profile = ApplicationEnvironmentProfile.lab_defaults()
    fragments = _runtime_fragments(profile)
    assert fragments[HARNESS_REFERENCE_POLICY_FRAGMENT_ID] == reference_harness_policy_fragment_payload()
    assert fragments[LEGAL_REFERENCE_POLICY_FRAGMENT_ID] == (reference_legal_contract_review_policy_fragment_payload())


def test_harness_production_defaults_declares_both_reference_fragments() -> None:
    profile = ApplicationEnvironmentProfile.harness_production_defaults()
    fragments = _runtime_fragments(profile)
    assert HARNESS_REFERENCE_POLICY_FRAGMENT_ID in fragments
    assert LEGAL_REFERENCE_POLICY_FRAGMENT_ID in fragments


def test_product_defaults_with_harness_bundle_declares_harness_fragment() -> None:
    profile = ApplicationEnvironmentProfile.product_defaults(skill_bundles=["harness"])
    assert HARNESS_REFERENCE_POLICY_FRAGMENT_ID in _runtime_fragments(profile)


def test_product_defaults_without_harness_bundle_omits_harness_fragment() -> None:
    profile = ApplicationEnvironmentProfile.product_defaults(skill_bundles=["rag"])
    assert HARNESS_REFERENCE_POLICY_FRAGMENT_ID not in _runtime_fragments(profile)


def test_product_defaults_preserves_caller_domain_fragments() -> None:
    caller = {"custom.domain.policy": {"marker": "caller"}}
    profile = ApplicationEnvironmentProfile.product_defaults(
        skill_bundles=["rag"],
        domain_fragments=caller,
    )
    assert _runtime_fragments(profile)["custom.domain.policy"] == {"marker": "caller"}


def test_product_defaults_caller_fragment_overrides_reference_default() -> None:
    caller = {HARNESS_REFERENCE_POLICY_FRAGMENT_ID: {"require_human_on_critical": True, "caller": True}}
    profile = ApplicationEnvironmentProfile.product_defaults(
        skill_bundles=["harness"],
        domain_fragments=caller,
    )
    payload = _runtime_fragments(profile)[HARNESS_REFERENCE_POLICY_FRAGMENT_ID]
    assert payload.get("caller") is True


def test_product_defaults_build_does_not_mutate_caller_dict() -> None:
    caller = {"custom.domain.policy": {"marker": "caller"}}
    snapshot = deepcopy(caller)
    profile = ApplicationEnvironmentProfile.product_defaults(
        skill_bundles=["harness"],
        domain_fragments=caller,
    )
    assert caller == snapshot
    assert HARNESS_REFERENCE_POLICY_FRAGMENT_ID not in caller
    assert HARNESS_REFERENCE_POLICY_FRAGMENT_ID in _runtime_fragments(profile)


def test_reference_payload_helpers_return_independent_dicts() -> None:
    left = reference_harness_policy_fragment_payload()
    right = reference_harness_policy_fragment_payload()
    left["probe"] = True
    assert "probe" not in right
    legal_left = reference_legal_contract_review_policy_fragment_payload()
    legal_right = reference_legal_contract_review_policy_fragment_payload()
    legal_left["probe"] = True
    assert "probe" not in legal_right
    profile_a = ApplicationEnvironmentProfile.lab_defaults()
    profile_b = ApplicationEnvironmentProfile.lab_defaults()
    fragments_a = dict(_runtime_fragments(profile_a))
    fragments_b = dict(_runtime_fragments(profile_b))
    fragments_a[HARNESS_REFERENCE_POLICY_FRAGMENT_ID]["probe"] = True
    assert "probe" not in _runtime_fragments(profile_b)[HARNESS_REFERENCE_POLICY_FRAGMENT_ID]
    assert fragments_b == dict(_runtime_fragments(profile_b))


def test_harness_policy_bundle_and_skill_composition_passes() -> None:
    profile = expand_governance_permission_preset(
        ApplicationEnvironmentProfile.harness_production_defaults(),
    )
    bundle = wire_policy_bundle(profile)
    registry = SkillRegistry()
    registry.register(HARNESS_POLICY_SMOKE)
    composition = SkillResolver(registry).resolve_composition([HARNESS_POLICY_SMOKE.skill_id])
    merged = merge_skill_policy_fragments(bundle, composition.pack)
    assert HARNESS_REFERENCE_POLICY_FRAGMENT_ID in merged.domain_fragments


def test_lab_harness_and_legal_skill_fragment_composition_passes() -> None:
    profile = expand_governance_permission_preset(ApplicationEnvironmentProfile.lab_defaults())
    bundle = wire_policy_bundle(profile)
    registry = SkillRegistry()
    registry.register(LEGAL_CONTRACT_REVIEW)
    registry.register(HARNESS_POLICY_SMOKE)
    composition = SkillResolver(registry).resolve_composition(
        [HARNESS_POLICY_SMOKE.skill_id, LEGAL_CONTRACT_REVIEW.skill_id],
    )
    assert HARNESS_REFERENCE_POLICY_FRAGMENT_ID in composition.pack.policy_fragment_ids
    assert LEGAL_REFERENCE_POLICY_FRAGMENT_ID in composition.pack.policy_fragment_ids
    rules_before = len(bundle.declarative_policy_runtime.rules) if bundle.declarative_policy_runtime else 0
    tools_before = bundle.tool_access
    hitl_before = bundle.require_human_on_critical
    merged = merge_skill_policy_fragments(bundle, composition.pack)
    assert HARNESS_REFERENCE_POLICY_FRAGMENT_ID in merged.domain_fragments
    assert LEGAL_REFERENCE_POLICY_FRAGMENT_ID in merged.domain_fragments
    assert merged.require_human_on_critical == hitl_before
    rules_after = len(merged.declarative_policy_runtime.rules) if merged.declarative_policy_runtime else 0
    assert rules_after == rules_before
    assert merged.tool_access == tools_before


def test_missing_required_fragment_still_raises() -> None:
    registry = SkillRegistry()
    registry.register(HARNESS_POLICY_SMOKE)
    composition = SkillResolver(registry).resolve_composition([HARNESS_POLICY_SMOKE.skill_id])
    with pytest.raises(SkillBridgeCompositionError, match="host catalog"):
        merge_skill_policy_fragments(RuntimePolicyBundle(), composition.pack)


def test_fragment_hitl_true_with_host_hitl_false_raises() -> None:
    registry = SkillRegistry()
    registry.register(HARNESS_POLICY_SMOKE)
    composition = SkillResolver(registry).resolve_composition([HARNESS_POLICY_SMOKE.skill_id])
    bundle = RuntimePolicyBundle(
        require_human_on_critical=False,
        domain_fragments={
            HARNESS_REFERENCE_POLICY_FRAGMENT_ID: reference_harness_policy_fragment_payload(),
        },
    )
    with pytest.raises(SkillBridgeCompositionError, match="requires require_human_on_critical"):
        merge_skill_policy_fragments(bundle, composition.pack)


def test_fragment_cannot_clear_required_hitl() -> None:
    registry = SkillRegistry()
    registry.register(HARNESS_POLICY_SMOKE)
    composition = SkillResolver(registry).resolve_composition([HARNESS_POLICY_SMOKE.skill_id])
    bundle = RuntimePolicyBundle(
        require_human_on_critical=True,
        domain_fragments={
            HARNESS_REFERENCE_POLICY_FRAGMENT_ID: {"require_human_on_critical": False},
        },
    )
    with pytest.raises(SkillBridgeCompositionError, match="cannot clear require_human_on_critical"):
        merge_skill_policy_fragments(bundle, composition.pack)


def test_merge_does_not_add_allow_rules_or_tools() -> None:
    profile = expand_governance_permission_preset(ApplicationEnvironmentProfile.lab_defaults())
    bundle = wire_policy_bundle(profile)
    registry = SkillRegistry()
    registry.register(HARNESS_POLICY_SMOKE)
    composition = SkillResolver(registry).resolve_composition([HARNESS_POLICY_SMOKE.skill_id])
    rules_before = len(bundle.declarative_policy_runtime.rules) if bundle.declarative_policy_runtime else 0
    tools_before = bundle.tool_access
    merged = merge_skill_policy_fragments(bundle, composition.pack)
    rules_after = len(merged.declarative_policy_runtime.rules) if merged.declarative_policy_runtime else 0
    assert rules_after == rules_before
    assert merged.tool_access == tools_before


def test_pinned_fragment_digest_drift_refused_on_resume() -> None:
    profile = expand_governance_permission_preset(ApplicationEnvironmentProfile.lab_defaults())
    host_bundle = wire_policy_bundle(profile)
    registry = SkillRegistry()
    registry.register(HARNESS_POLICY_SMOKE)
    composition = SkillResolver(registry).resolve_composition([HARNESS_POLICY_SMOKE.skill_id])
    merged = merge_skill_policy_fragments(host_bundle, composition.pack)
    pins = collect_skill_policy_fragment_pins(merged, composition.pack)
    drifted = replace(
        merged,
        domain_fragments={
            **merged.domain_fragments,
            HARNESS_REFERENCE_POLICY_FRAGMENT_ID: {
                **reference_harness_policy_fragment_payload(),
                "drift": True,
            },
        },
    )
    with pytest.raises(SkillBridgeCompositionError, match="digest mismatch"):
        verify_pinned_policy_fragments_on_resume(drifted, pins)
