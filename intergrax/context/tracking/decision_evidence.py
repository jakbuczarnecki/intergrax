# © Artur Czarnecki. All rights reserved.

"""Canonical context-decision evidence fingerprint (Context Engineering ownership)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from intergrax.context.budget.compaction import ContextCompactionProvenance
from intergrax.context.budget.contracts import ResolvedModelContextBudget
from intergrax.context.contracts import (
    AssembledContext,
    ContextAssemblyProvenance,
    ContextConflictDecision,
    ContextPolicyDecision,
    ContextProviderCollectionOutcome,
    ContextProviderDescriptor,
    ContextProviderSetSnapshot,
    ContextSemanticDedupDecision,
)
from intergrax.context.serialization import serialize_context_plan_safe


def _provenance_entry(provenance: ContextAssemblyProvenance) -> dict[str, Any]:
    return {
        "content_hash": provenance.content_hash,
        "fragment_id": provenance.fragment_id,
        "provider_id": provenance.provider_id,
        "provider_origin": provenance.provider_origin,
        "provider_version": provenance.provider_version,
        "source_id": provenance.source_id,
        "source_type": provenance.source_type,
    }


def _provider_descriptor_entry(descriptor: ContextProviderDescriptor) -> dict[str, Any]:
    return {
        "allowed_authority_classes": sorted(c.value for c in descriptor.allowed_authority_classes),
        "origin": descriptor.origin,
        "provider_id": descriptor.provider_id,
        "provider_version": descriptor.provider_version,
        "supported_sources": sorted(s.value for s in descriptor.supported_sources),
        "trusted_authority_class": (
            descriptor.trusted_authority_class.value
            if descriptor.trusted_authority_class is not None
            else None
        ),
    }


def _provider_set_snapshot(snapshot: ContextProviderSetSnapshot) -> dict[str, Any]:
    return {
        "engine_id": snapshot.engine_id,
        "fingerprint": snapshot.fingerprint,
        "providers": [
            _provider_descriptor_entry(descriptor) for descriptor in snapshot.providers
        ],
    }


def _provider_outcome(outcome: ContextProviderCollectionOutcome) -> dict[str, Any]:
    return {
        "failure_reason": outcome.failure_reason,
        "fragment_count": outcome.fragment_count,
        "provider": _provider_descriptor_entry(outcome.descriptor),
        "reason_code": outcome.reason_code,
        "status": outcome.status.value,
    }


def _policy_decision(decision: ContextPolicyDecision) -> dict[str, Any]:
    return {
        "detail": decision.detail,
        "input_fragment_ids": list(decision.input_fragment_ids),
        "output_fragment_ids": list(decision.output_fragment_ids),
        "reason_code": decision.reason_code.value,
        "stage": decision.stage.value,
        "strategy_id": decision.strategy_id,
    }


def _semantic_dedup(decision: ContextSemanticDedupDecision) -> dict[str, Any]:
    return {
        "kept_fragment_id": decision.kept_fragment_id,
        "reason_code": decision.reason_code.value,
        "strategy_id": decision.strategy_id,
        "suppressed_fragment_ids": list(decision.suppressed_fragment_ids),
    }


def _conflict_decision(decision: ContextConflictDecision) -> dict[str, Any]:
    return {
        "action": decision.action.value,
        "kept_fragment_ids": list(decision.kept_fragment_ids),
        "left_fragment_id": decision.left_fragment_id,
        "reason_code": decision.reason_code.value,
        "right_fragment_id": decision.right_fragment_id,
        "strategy_id": decision.strategy_id,
    }


def _compaction_provenance(record: ContextCompactionProvenance) -> dict[str, Any]:
    return {
        "input_content_hash": record.input_content_hash,
        "input_token_estimate": record.input_token_estimate,
        "output_content_hash": record.output_content_hash,
        "output_token_estimate": record.output_token_estimate,
        "reason_code": record.reason_code,
        "source_fragment_ids": list(record.source_fragment_ids),
        "strategy_id": record.strategy_id,
    }


def _resolved_model_budget(budget: ResolvedModelContextBudget) -> dict[str, Any]:
    return {
        "allocatable_tokens": budget.allocatable_tokens,
        "available_input_tokens": budget.available_input_tokens,
        "mandatory_reserve_tokens": budget.mandatory_reserve_tokens,
        "model_context_window": budget.model_context_window,
        "platform_margin_tokens": budget.platform_margin_tokens,
        "policy_id": budget.policy_id,
        "policy_version": budget.policy_version,
        "request_cap_tokens": budget.request_cap_tokens,
        "reserved_output_tokens": budget.reserved_output_tokens,
    }


def build_context_decision_evidence_canonical(assembled: AssembledContext) -> dict[str, Any]:
    """Privacy-bounded canonical document for decision semantics (no raw message bodies)."""
    plan_payload: dict[str, Any] | None = None
    if assembled.context_plan is not None:
        plan_payload = serialize_context_plan_safe(assembled.context_plan)
    snapshot_payload: dict[str, Any] | None = None
    if assembled.provider_set_snapshot is not None:
        snapshot_payload = _provider_set_snapshot(assembled.provider_set_snapshot)
    budget_payload: dict[str, Any] | None = None
    if assembled.resolved_model_budget is not None:
        budget_payload = _resolved_model_budget(assembled.resolved_model_budget)
    return {
        "compaction_provenance": [
            _compaction_provenance(record) for record in assembled.compaction_provenance
        ],
        "compaction_strategy_id": assembled.compaction_strategy_id,
        "context_plan": plan_payload,
        "degradation_policy_id": assembled.degradation_policy_id,
        "degradation_steps": list(assembled.degradation_steps),
        "policy_conflicts": [_conflict_decision(d) for d in assembled.policy_conflicts],
        "policy_decisions": [_policy_decision(d) for d in assembled.policy_decisions],
        "policy_semantic_dedup": [_semantic_dedup(d) for d in assembled.policy_semantic_dedup],
        "provider_outcomes": [_provider_outcome(o) for o in assembled.provider_outcomes],
        "provider_set_snapshot": snapshot_payload,
        "provenance": [_provenance_entry(p) for p in assembled.provenance],
        "resolved_model_budget": budget_payload,
        "token_counter_strategy_id": assembled.token_counter_strategy_id,
    }


def compute_context_decision_evidence_fingerprint(assembled: AssembledContext) -> str:
    """Deterministic SHA-256 over canonical context-decision semantics."""
    canonical = build_context_decision_evidence_canonical(assembled)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_context_decision_evidence_fingerprint_from_metadata(
    metadata: Mapping[str, Any],
    *,
    degradation_steps: tuple[str, ...] = (),
) -> str:
    """Legacy assembly path fingerprint from bounded metadata (no raw content)."""
    canonical = {
        "compaction_strategy_id": str(metadata.get("compaction_strategy_id") or ""),
        "degradation_policy_id": str(metadata.get("degradation_policy_id") or ""),
        "degradation_steps": list(degradation_steps or metadata.get("degradation_steps") or ()),
        "engine_id": str(metadata.get("engine_id") or ""),
        "summary_tier": metadata.get("summary_tier"),
        "token_counter_strategy_id": str(metadata.get("token_counter_strategy_id") or ""),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "build_context_decision_evidence_canonical",
    "compute_context_decision_evidence_fingerprint",
    "compute_context_decision_evidence_fingerprint_from_metadata",
]
