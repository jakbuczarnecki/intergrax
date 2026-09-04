# © Artur Czarnecki. All rights reserved.

"""Legacy ``critic_profile`` → ``DecisionProfile`` migration boundary (DS-MIG-05)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_LEGACY_CRITIC_MIGRATION_GUIDANCE = (
    "Migrate legacy critic_profile to cognition.decision / decision_profile. "
    "Removed fields (L2, require_critic_on_completion, node_partial, "
    "evaluator_loop_max_iterations, judge_threshold) have canonical Decision owners."
)


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError("legacy critic payload must be a mapping")


def _reject_unsafe_legacy_critic_fields(raw: Mapping[str, Any]) -> None:
    if raw.get("l2_human_required") is True:
        raise ValueError(
            f"{_LEGACY_CRITIC_MIGRATION_GUIDANCE} l2_human_required is not migratable.",
        )
    margin = raw.get("l2_borderline_margin", 0.05)
    if margin is not None and float(margin) != 0.05:
        raise ValueError(
            f"{_LEGACY_CRITIC_MIGRATION_GUIDANCE} l2_borderline_margin is not migratable.",
        )
    if raw.get("require_critic_on_completion") is True:
        raise ValueError(
            f"{_LEGACY_CRITIC_MIGRATION_GUIDANCE} require_critic_on_completion is not migratable.",
        )
    if "evaluator_loop_max_iterations" in raw:
        raise ValueError(
            f"{_LEGACY_CRITIC_MIGRATION_GUIDANCE} evaluator_loop_max_iterations is not migratable.",
        )
    if "judge_threshold" in raw:
        raise ValueError(
            f"{_LEGACY_CRITIC_MIGRATION_GUIDANCE} judge_threshold belongs on semantic rubrics.",
        )
    scopes = raw.get("scopes")
    if isinstance(scopes, Mapping) and scopes.get("node_partial") is True:
        raise ValueError(
            f"{_LEGACY_CRITIC_MIGRATION_GUIDANCE} scopes.node_partial has no canonical equivalent.",
        )


def migrate_legacy_critic_payload_to_decision(raw: object) -> dict[str, Any]:
    """Translate safe legacy critic fields into canonical decision wire shape."""
    mapping = _as_mapping(raw)
    _reject_unsafe_legacy_critic_fields(mapping)
    scopes = mapping.get("scopes")
    scope_mapping: Mapping[str, Any] = scopes if isinstance(scopes, Mapping) else {}
    verification: dict[str, Any] = {
        "semantic_enabled": bool(mapping.get("semantic_judge_enabled", False)),
        "trajectory_enabled": bool(mapping.get("trajectory_eval_enabled", False)),
    }
    if mapping.get("critic_llm_profile_ref") is not None:
        verification["verifier_llm_profile_ref"] = mapping["critic_llm_profile_ref"]
    if mapping.get("critic_llm_profile") is not None:
        verification["verifier_llm_profile"] = mapping["critic_llm_profile"]
    if mapping.get("default_rubric_ref") is not None:
        verification["semantic_rubric_ref"] = mapping["default_rubric_ref"]
    flow: dict[str, Any] = {
        "verify_graph_final": bool(scope_mapping.get("graph_final", True)),
        "verify_uaep_step": bool(scope_mapping.get("uaep_step", False)),
        "max_revisions": 0,
    }
    return {"verification": verification, "flow": flow}


def migrate_cognition_wire_data(cognition: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize cognition bundle wire data to canonical ``decision`` storage."""
    if "decision" in cognition and "critic" in cognition:
        raise ValueError("cognition cannot contain both decision and critic")
    if "critic" not in cognition:
        return dict(cognition)
    migrated = dict(cognition)
    critic_payload = migrated.pop("critic")
    migrated["decision"] = migrate_legacy_critic_payload_to_decision(critic_payload)
    return migrated


def migrate_environment_profile_wire(data: Mapping[str, Any]) -> dict[str, Any]:
    """Apply legacy critic migration on nested environment profile wire data."""
    payload = dict(data)
    cognition = payload.get("cognition")
    if isinstance(cognition, Mapping):
        payload["cognition"] = migrate_cognition_wire_data(cognition)
    return payload
