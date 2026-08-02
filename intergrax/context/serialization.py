# © Artur Czarnecki. All rights reserved.

"""Telemetry-safe serialization for context planning contracts (CTX-UCL-3)."""

from __future__ import annotations

from typing import Any

from intergrax.context.planning import (
    ContextArtifactLookupInputs,
    ContextArtifactRequirement,
    ContextPlan,
    ContextSourceGroup,
)
from intergrax.context.session_history import SessionHistorySnapshot

_RAW_CONTENT_INCLUDED = False


def _compression_target_to_dict(target: object) -> dict[str, Any]:
    from intergrax.runtime.context_lifecycle.contracts import ArtifactCompressionTarget

    if not isinstance(target, ArtifactCompressionTarget):
        raise ValueError("compression_target must be ArtifactCompressionTarget")
    if target.target_tokens is not None:
        return {"target_tokens": target.target_tokens}
    return {"budget_class": target.budget_class}


def serialize_context_source_group_safe(group: ContextSourceGroup) -> dict[str, Any]:
    return {
        "compressible": group.compressible,
        "droppable": group.droppable,
        "end_sequence": group.end_sequence,
        "group_id": group.group_id,
        "protected": group.protected,
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "required": group.required,
        "source": group.source.value,
        "source_content_hash": group.source_content_hash,
        "source_ref_count": len(group.source_refs),
        "start_sequence": group.start_sequence,
        "token_estimate": group.token_estimate,
        "trim_safe": group.trim_safe,
    }


def serialize_context_artifact_lookup_inputs_safe(
    inputs: ContextArtifactLookupInputs,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_type": inputs.artifact_type.value,
        "compression_target": _compression_target_to_dict(inputs.compression_target),
        "lossiness_profile": inputs.lossiness_profile,
        "model_family": inputs.model_family,
        "locale": inputs.locale,
        "protected_region_policy_version": inputs.protected_region_policy_version,
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "source_content_hash": inputs.source_content_hash,
        "source_ref_count": len(inputs.source_refs),
        "source_range_present": inputs.source_range is not None,
    }
    return payload


def serialize_context_artifact_requirement_safe(
    requirement: ContextArtifactRequirement,
) -> dict[str, Any]:
    return {
        "allowed_strategy_ids": list(requirement.allowed_strategy_ids),
        "lookup_inputs": serialize_context_artifact_lookup_inputs_safe(requirement.lookup_inputs),
        "minimum_preservation": {
            "preserve_message_ids": requirement.minimum_preservation.preserve_message_ids,
            "preserve_message_order": requirement.minimum_preservation.preserve_message_order,
            "preserve_recent_tail_messages": (
                requirement.minimum_preservation.preserve_recent_tail_messages
            ),
            "preserve_roles": requirement.minimum_preservation.preserve_roles,
            "preserve_tool_call_links": requirement.minimum_preservation.preserve_tool_call_links,
            "protected_group_count": len(requirement.minimum_preservation.protected_group_ids),
            "required_group_count": len(requirement.minimum_preservation.required_group_ids),
        },
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "source_group_count": len(requirement.source_group_ids),
    }


def serialize_context_plan_safe(plan: ContextPlan) -> dict[str, Any]:
    from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope

    scope = plan.execution_scope
    if not isinstance(scope, ModelCallExecutionScope):
        raise ValueError("execution_scope must be ModelCallExecutionScope")
    return {
        "artifact_requirement": (
            serialize_context_artifact_requirement_safe(plan.artifact_requirement)
            if plan.artifact_requirement is not None
            else None
        ),
        "budget_class": plan.budget_class.value,
        "compressible_group_count": len(plan.compressible_group_ids),
        "droppable_group_count": len(plan.droppable_group_ids),
        "estimated_total_tokens": plan.estimated_total_tokens,
        "execution_scope": scope.value,
        "excluded_group_count": len(plan.excluded_group_ids),
        "final_validation_requirements": list(plan.final_validation_requirements),
        "optimization_required": plan.optimization_required,
        "protected_group_count": len(plan.protected_group_ids),
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "required_group_count": len(plan.required_group_ids),
        "resolved_global_budget_tokens": plan.resolved_global_budget_tokens,
        "selected_group_count": len(plan.selected_group_ids),
        "source_allocations": [
            {
                "allocated_tokens": allocation.allocated_tokens,
                "excluded_group_count": len(allocation.excluded_group_ids),
                "selected_group_count": len(allocation.selected_group_ids),
                "source": allocation.source.value,
            }
            for allocation in plan.source_allocations
        ],
        "source_group_count": len(plan.source_groups),
        "source_groups": [
            serialize_context_source_group_safe(group) for group in plan.source_groups
        ],
        "trim_safe_group_count": len(plan.trim_safe_group_ids),
    }


def serialize_session_history_snapshot_safe(
    snapshot: SessionHistorySnapshot,
) -> dict[str, Any]:
    return {
        "message_count": len(snapshot.messages),
        "raw_content_included": _RAW_CONTENT_INCLUDED,
        "revision_id": snapshot.revision_id,
        "source_content_hash": snapshot.source_content_hash,
        "source_ref_count": len(snapshot.source_refs),
    }
