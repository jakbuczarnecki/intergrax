# © Artur Czarnecki. All rights reserved.

"""Skill prompt/policy runtime bridge (SK-BRIDGE.*)."""

from __future__ import annotations

from typing import Any

from intergrax.prompts.storage.models import LoadedPrompt
from intergrax.runtime.policy.policy_bundle import DeclarativePolicyRuntime, RuntimePolicyBundle
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule, PolicyRuleAction
from intergrax.skills.instruction_policy_pins import (
    SkillBridgeCompositionError,
    SkillPolicyFragmentPin,
    SkillPromptInstructionPin,
    apply_skill_bridge_at_bind,
    host_policy_fragment_payload,
    resolve_pinned_prompt_instructions,
    verify_pinned_policy_fragments_on_resume,
    verify_skill_bridge_on_resume,
)
from intergrax.skills.resolver import ResolvedSkillComposition, ResolvedSkillPack

_SKILL_FRAGMENT_PAYLOAD_SCHEMA = "skill_policy_fragment.v1"
_ACTION_STRICTNESS = {
    PolicyRuleAction.DENY: 3,
    PolicyRuleAction.REQUIRE_HITL: 2,
    PolicyRuleAction.ALLOW: 1,
}

__all__ = [
    "SkillBridgeCompositionError",
    "SkillPolicyFragmentPin",
    "SkillPromptInstructionPin",
    "apply_skill_bridge_at_bind",
    "merge_skill_policy_fragments",
    "resolve_pinned_prompt_instructions",
    "skill_prompt_metadata",
    "verify_pinned_policy_fragments_on_resume",
    "verify_skill_bridge_on_resume",
]


def skill_prompt_metadata(
    pack: ResolvedSkillPack,
    *,
    resolved_prompts: dict[str, LoadedPrompt] | None = None,
) -> dict[str, list[str]]:
    """SK-BRIDGE.1 — prompt instruction ids and optional pinned bodies for hosts."""
    if not pack.prompt_instruction_ids:
        return {}
    metadata: dict[str, list[str]] = {
        "skill_prompt_instruction_ids": sorted(pack.prompt_instruction_ids),
    }
    if resolved_prompts:
        bodies = [
            resolved_prompts[prompt_id].document.locales["en"].system
            for prompt_id in sorted(pack.prompt_instruction_ids)
            if prompt_id in resolved_prompts
        ]
        if bodies:
            metadata["skill_prompt_instruction_bodies"] = bodies
    return metadata


def _extract_declarative_rules(payload: dict[str, Any]) -> tuple[DeclarativePolicyRule, ...]:
    rules_raw = payload.get("declarative_rules")
    if not rules_raw:
        return ()
    if not isinstance(rules_raw, list):
        raise SkillBridgeCompositionError("declarative_rules must be a list when present")
    rules: list[DeclarativePolicyRule] = []
    for item in rules_raw:
        rules.append(DeclarativePolicyRule.model_validate(item))
    return tuple(rules)


def _rule_identity(rule: DeclarativePolicyRule) -> tuple[str, str, str]:
    return (rule.rule_id, rule.resource_kind, rule.resource_id)


def _host_rule_index(
    bundle: RuntimePolicyBundle,
) -> dict[tuple[str, str, str], PolicyRuleAction]:
    runtime = bundle.declarative_policy_runtime
    if runtime is None:
        return {}
    return {_rule_identity(rule): rule.action for rule in runtime.rules}


def _assert_skill_fragment_admissible(
    bundle: RuntimePolicyBundle,
    fragment_id: str,
    payload: dict[str, Any],
) -> tuple[DeclarativePolicyRule, ...]:
    rules = _extract_declarative_rules(payload)
    host_rules = _host_rule_index(bundle)

    if payload.get("require_human_on_critical") is False and bundle.require_human_on_critical:
        raise SkillBridgeCompositionError(
            f"skill fragment cannot clear require_human_on_critical: {fragment_id}",
        )

    host_tenants = payload.get("allowed_tenants")
    skill_tenants = payload.get("skill_allowed_tenants")
    if isinstance(host_tenants, list) and isinstance(skill_tenants, list):
        widened = set(skill_tenants) - set(host_tenants)
        if widened:
            raise SkillBridgeCompositionError(
                f"skill fragment widens tenant scope: {fragment_id}",
            )

    for rule in rules:
        host_action = host_rules.get(_rule_identity(rule))
        if host_action is None:
            continue
        if _ACTION_STRICTNESS[rule.action] < _ACTION_STRICTNESS[host_action]:
            raise SkillBridgeCompositionError(
                f"skill fragment weakens host policy rule {rule.rule_id}: {fragment_id}",
            )
    return rules


def _collect_skill_rule_conflicts(
    fragment_rules: list[tuple[str, DeclarativePolicyRule]],
) -> None:
    seen: dict[tuple[str, str, str], DeclarativePolicyRule] = {}
    for _fragment_id, rule in fragment_rules:
        key = _rule_identity(rule)
        previous = seen.get(key)
        if previous is not None and previous.model_dump() != rule.model_dump():
            raise SkillBridgeCompositionError(
                f"conflicting policy rule values across skill fragments for {rule.rule_id}",
            )
        seen[key] = rule


def merge_skill_policy_fragments(
    bundle: RuntimePolicyBundle,
    pack: ResolvedSkillPack,
    *,
    composition: ResolvedSkillComposition | None = None,
) -> RuntimePolicyBundle:
    """SK-BRIDGE.2 — merge skill catalog policy fragments into the host bundle."""
    del composition
    if not pack.policy_fragment_ids:
        return bundle

    fragments = dict(bundle.domain_fragments)
    appended_rules: list[DeclarativePolicyRule] = []
    skill_rules_with_source: list[tuple[str, DeclarativePolicyRule]] = []

    for fragment_id in sorted(pack.policy_fragment_ids):
        payload = host_policy_fragment_payload(bundle, fragment_id)
        rules = _assert_skill_fragment_admissible(bundle, fragment_id, payload)
        skill_rules_with_source.extend((fragment_id, rule) for rule in rules)
        fragments[fragment_id] = {
            "schema_version": _SKILL_FRAGMENT_PAYLOAD_SCHEMA,
            "fragment_id": fragment_id,
            "composition_layer": "skill",
            "catalog_payload": payload,
        }

    _collect_skill_rule_conflicts(skill_rules_with_source)
    appended_rules.extend(rule for _, rule in skill_rules_with_source)

    declarative_runtime = bundle.declarative_policy_runtime
    if appended_rules:
        if declarative_runtime is None:
            raise SkillBridgeCompositionError(
                "skill policy fragments require declarative_policy_runtime on host bundle",
            )
        merged_rules = tuple(declarative_runtime.rules) + tuple(appended_rules)
        declarative_runtime = DeclarativePolicyRuntime(
            registry=declarative_runtime.registry,
            rules=merged_rules,
            load_report=declarative_runtime.load_report,
            provenance=declarative_runtime.provenance,
            enforcement_mode=declarative_runtime.enforcement_mode,
        )

    return RuntimePolicyBundle(
        tool_access=bundle.tool_access,
        require_human_on_critical=bundle.require_human_on_critical,
        domain_fragments=fragments,
        policy_catalog=bundle.policy_catalog,
        configuration_contract_registry=bundle.configuration_contract_registry,
        declarative_policy_runtime=declarative_runtime,
    )
