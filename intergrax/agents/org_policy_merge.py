# © Artur Czarnecki. All rights reserved.

"""Merge Tier-3 organizational envelope into runtime context (ACP-ORG-2)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.agent_run_binding import AgentRunBinding
from intergrax.contracts.org_policy import (
    OrganizationalPolicyContext,
    OrganizationalPolicyEnvelope,
    ScenarioBinding,
)


def _resolve_scenario_id(
    envelope: OrganizationalPolicyEnvelope,
    *,
    metadata: dict[str, Any],
    capability: str | None,
) -> str | None:
    explicit = metadata.get("scenario_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    for binding in envelope.scenario_bindings:
        triggers = (
            [binding.trigger]
            if isinstance(binding.trigger, str)
            else list(binding.trigger)
        )
        for trigger in triggers:
            if metadata.get(trigger) is not None:
                return binding.scenario_id
            if capability and capability == trigger:
                return binding.scenario_id
    return None


def _playbook_ids_for_scenario(
    envelope: OrganizationalPolicyEnvelope,
    scenario_id: str | None,
) -> list[str]:
    if scenario_id is None:
        return []
    playbooks: list[str] = []
    for binding in envelope.scenario_bindings:
        if binding.scenario_id == scenario_id:
            playbooks.append(binding.required_playbook_id)
    return playbooks


def merge_organizational_policy_context(
    *,
    envelope: OrganizationalPolicyEnvelope | None,
    binding: AgentRunBinding | None = None,
    metadata: dict[str, Any] | None = None,
    capability: str | None = None,
) -> OrganizationalPolicyContext | None:
    """Materialize ``OrganizationalPolicyContext`` for ``merge_environment``."""
    if envelope is None:
        return None

    meta = dict(metadata or {})
    scenario_id = _resolve_scenario_id(envelope, metadata=meta, capability=capability)
    playbooks = _playbook_ids_for_scenario(envelope, scenario_id)

    tool_denies: list[str] = []
    if envelope.tool_policy_overlay is not None:
        tool_denies.extend(envelope.tool_policy_overlay.deny_patterns)
    if binding is not None:
        tool_denies.extend(binding.tool_denylist)

    prompt_overlays = list(envelope.communication_rules.required_disclosures)
    if envelope.communication_rules.tone:
        prompt_overlays.append(f"org.tone.{envelope.communication_rules.tone}")

    return OrganizationalPolicyContext(
        organization_id=envelope.organization_id,
        org_role_id=binding.org_role_id if binding is not None else None,
        active_scenario_id=scenario_id,
        active_playbook_ids=playbooks,
        channel_policy=envelope.channel_policy.model_copy(deep=True),
        effective_tool_denies=sorted(set(tool_denies)),
        prompt_overlay_ids=prompt_overlays,
        observability_labels=dict(envelope.observability_labels),
        execution_mode=envelope.execution_mode,
        domain_fragments={
            "compliance_profile_id": envelope.compliance_profile_id,
            "rag_playbook_collection": envelope.rag_playbook_collection,
        },
    )
