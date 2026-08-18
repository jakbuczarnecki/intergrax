# © Artur Czarnecki. All rights reserved.

"""Skill prompt/policy runtime bridge (SK-BRIDGE.*)."""

from __future__ import annotations

from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.skills.resolver import ResolvedSkillPack


def skill_prompt_metadata(pack: ResolvedSkillPack) -> dict[str, list[str]]:
    """SK-BRIDGE.1 — prompt instruction ids for ContextManager / prompt registry."""
    if not pack.prompt_instruction_ids:
        return {}
    return {"skill_prompt_instruction_ids": list(pack.prompt_instruction_ids)}


def merge_skill_policy_fragments(
    bundle: RuntimePolicyBundle,
    pack: ResolvedSkillPack,
) -> RuntimePolicyBundle:
    """SK-BRIDGE.2 — merge skill policy fragments into runtime policy bundle."""
    if not pack.policy_fragment_ids:
        return bundle
    fragments = dict(bundle.domain_fragments)
    for fragment_id in pack.policy_fragment_ids:
        fragments[fragment_id] = {"source": "skill_pack", "id": fragment_id}
    return RuntimePolicyBundle(
        tool_access=bundle.tool_access,
        budget=bundle.budget,
        plan_loop=bundle.plan_loop,
        require_human_on_critical=bundle.require_human_on_critical,
        domain_fragments=fragments,
        policy_catalog=bundle.policy_catalog,
        configuration_contract_registry=bundle.configuration_contract_registry,
        declarative_policy_runtime=bundle.declarative_policy_runtime,
    )
