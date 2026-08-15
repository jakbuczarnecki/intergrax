# © Artur Czarnecki. All rights reserved.

"""Fail-closed policy composition boundary (COLLAB-WORK-1E).

Combines pre-evaluated collaborative authority, workspace, resource, and runtime/tool
policy decisions into one final enforcement ``PolicyDecision``. Does not evaluate rules,
does not execute side effects, and does not invent missing policy evaluators.
"""

from __future__ import annotations

from typing import Any

from intergrax.contracts.collaborative_work import (
    PolicyCompositionInput,
    PolicyCompositionLayer,
    PolicyCompositionResult,
    PolicyLayerApplicability,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

_POLICY_RULE_COMPOSITION_ALLOW = "collaborative_work.policy_composition.allow"
_POLICY_RULE_UNSUPPORTED_MODIFY = "collaborative_work.policy_composition.unsupported_modify"
_POLICY_RULE_MISSING_PREFIX = "collaborative_work.policy_composition.missing"
_POLICY_RULE_APPLICABILITY_UNRESOLVED_PREFIX = (
    "collaborative_work.policy_composition.applicability_unresolved"
)

_LAYER_ORDER: tuple[PolicyCompositionLayer, ...] = (
    PolicyCompositionLayer.COLLABORATIVE_AUTHORITY,
    PolicyCompositionLayer.WORKSPACE_POLICY,
    PolicyCompositionLayer.RESOURCE_POLICY,
    PolicyCompositionLayer.RUNTIME_POLICY,
)

_NON_EXECUTABLE_ACTIONS = frozenset({PolicyAction.REQUIRE_HUMAN, PolicyAction.ESCALATE})


def _action_precedence_rank(action: PolicyAction) -> int:
    """Explicit strictness ordering — not enum declaration order."""
    if action is PolicyAction.DENY:
        return 0
    if action in _NON_EXECUTABLE_ACTIONS:
        return 1
    if action is PolicyAction.MODIFY:
        return 2
    if action is PolicyAction.ALLOW:
        return 3
    return 0


def _missing_mandatory_decision(layer: PolicyCompositionLayer) -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.DENY,
        reason=f"mandatory {layer.value} decision missing",
        policy_rule_id=f"{_POLICY_RULE_MISSING_PREFIX}.{layer.value}",
        audit_payload={
            "layer": layer.value,
            "cause": "missing_mandatory_decision",
        },
    )


def _unresolved_applicability_decision(layer: PolicyCompositionLayer) -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.DENY,
        reason=f"{layer.value} applicability unresolved",
        policy_rule_id=f"{_POLICY_RULE_APPLICABILITY_UNRESOLVED_PREFIX}.{layer.value}",
        audit_payload={
            "layer": layer.value,
            "cause": "applicability_unresolved",
        },
    )


def _normalize_modify(decision: PolicyDecision, layer: PolicyCompositionLayer) -> PolicyDecision:
    if decision.action is not PolicyAction.MODIFY:
        return decision
    return PolicyDecision(
        action=PolicyAction.DENY,
        reason="modify_not_supported_at_enforcement_composition_boundary",
        policy_rule_id=_POLICY_RULE_UNSUPPORTED_MODIFY,
        audit_payload={
            "layer": layer.value,
            "original_action": PolicyAction.MODIFY.value,
            "original_reason": decision.reason,
            "original_policy_rule_id": decision.policy_rule_id,
        },
    )


def _decision_snapshot(decision: PolicyDecision) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "action": decision.action.value,
        "reason": decision.reason,
        "policy_rule_id": decision.policy_rule_id,
    }
    if decision.has_attested_policy_bundle_refs():
        snapshot["policy_bundle_id"] = decision.policy_bundle_id
        snapshot["policy_bundle_version"] = decision.policy_bundle_version
        snapshot["policy_bundle_digest"] = decision.policy_bundle_digest
    return snapshot


def _layer_entries(
    input: PolicyCompositionInput,
) -> list[tuple[PolicyCompositionLayer, PolicyDecision]]:
    applicability = input.applicability
    entries: list[tuple[PolicyCompositionLayer, PolicyDecision]] = [
        (
            PolicyCompositionLayer.COLLABORATIVE_AUTHORITY,
            _normalize_modify(
                input.collaborative_authority,
                PolicyCompositionLayer.COLLABORATIVE_AUTHORITY,
            ),
        ),
    ]
    optional_layers: tuple[
        tuple[PolicyCompositionLayer, PolicyDecision | None, PolicyLayerApplicability],
        ...,
    ] = (
        (
            PolicyCompositionLayer.WORKSPACE_POLICY,
            input.workspace_policy,
            applicability.workspace_policy,
        ),
        (
            PolicyCompositionLayer.RESOURCE_POLICY,
            input.resource_policy,
            applicability.resource_policy,
        ),
        (
            PolicyCompositionLayer.RUNTIME_POLICY,
            input.runtime_policy,
            applicability.runtime_policy,
        ),
    )
    for layer, decision, layer_applicability in optional_layers:
        if layer_applicability is PolicyLayerApplicability.NOT_APPLICABLE:
            continue
        if layer_applicability is PolicyLayerApplicability.UNKNOWN:
            entries.append((layer, _unresolved_applicability_decision(layer)))
            continue
        if decision is None:
            entries.append((layer, _missing_mandatory_decision(layer)))
        else:
            entries.append((layer, _normalize_modify(decision, layer)))
    return entries


def _select_determining_entry(
    entries: list[tuple[PolicyCompositionLayer, PolicyDecision]],
) -> tuple[PolicyCompositionLayer, PolicyDecision]:
    best_layer = entries[0][0]
    best_decision = entries[0][1]
    best_rank = _action_precedence_rank(best_decision.action)
    for layer, decision in entries[1:]:
        rank = _action_precedence_rank(decision.action)
        if rank < best_rank:
            best_layer = layer
            best_decision = decision
            best_rank = rank
    return best_layer, best_decision


def compose_policy_decisions(input: PolicyCompositionInput) -> PolicyCompositionResult:
    """Compose mandatory policy layers into one fail-closed enforcement decision."""
    entries = _layer_entries(input)
    determining_layer, determining_decision = _select_determining_entry(entries)

    contributing: dict[str, dict[str, Any]] = {
        layer.value: _decision_snapshot(decision) for layer, decision in entries
    }
    non_allow_layers = [
        layer.value for layer, decision in entries if decision.action is not PolicyAction.ALLOW
    ]

    if determining_decision.action is PolicyAction.ALLOW:
        final = PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="all mandatory policy layers allow",
            policy_rule_id=_POLICY_RULE_COMPOSITION_ALLOW,
            audit_payload={
                "schema": "policy_composition_audit.v1",
                "contributing_layers": contributing,
                "non_allow_layers": non_allow_layers,
            },
        )
        return PolicyCompositionResult(
            decision=final,
            collaborative_authority=input.collaborative_authority,
            workspace_policy=input.workspace_policy,
            resource_policy=input.resource_policy,
            runtime_policy=input.runtime_policy,
            determining_layer=None,
        )

    audit_payload: dict[str, Any] = {
        "schema": "policy_composition_audit.v1",
        "determining_layer": determining_layer.value,
        "contributing_layers": contributing,
        "non_allow_layers": non_allow_layers,
    }
    if determining_decision.audit_payload:
        audit_payload["determining_layer_audit"] = dict(determining_decision.audit_payload)

    final = PolicyDecision(
        action=determining_decision.action,
        reason=determining_decision.reason,
        modified_decision=determining_decision.modified_decision,
        enforcement_level=determining_decision.enforcement_level,
        policy_rule_id=determining_decision.policy_rule_id,
        policy_bundle_id=determining_decision.policy_bundle_id,
        policy_bundle_version=determining_decision.policy_bundle_version,
        policy_bundle_digest=determining_decision.policy_bundle_digest,
        decision_id=determining_decision.decision_id,
        audit_payload=audit_payload,
    )
    return PolicyCompositionResult(
        decision=final,
        collaborative_authority=input.collaborative_authority,
        workspace_policy=input.workspace_policy,
        resource_policy=input.resource_policy,
        runtime_policy=input.runtime_policy,
        determining_layer=determining_layer,
    )
