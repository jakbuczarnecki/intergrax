# © Artur Czarnecki. All rights reserved.

"""Hard invariant envelope for replaceable context policy (MEM-XINT-5-R2)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextConflictDecision,
    ContextFragment,
    ContextFragmentInvariantSnapshot,
    ContextPolicyInvariantViolationCode,
    ContextPolicyPipelineResult,
    ContextSemanticDedupDecision,
    content_hash_for_text,
)
from intergrax.context.errors import ContextPolicyInvariantViolationError


def build_fragment_invariant_snapshots(
    fragments: list[ContextFragment],
    *,
    pipeline_id: str = "platform.invariant_snapshot.v1",
) -> dict[str, ContextFragmentInvariantSnapshot]:
    snapshots: dict[str, ContextFragmentInvariantSnapshot] = {}
    for fragment in fragments:
        fragment_id = fragment.fragment_id
        if fragment_id in snapshots:
            raise ContextPolicyInvariantViolationError(
                fragment_id=fragment_id,
                invariant=ContextPolicyInvariantViolationCode.DUPLICATE_FRAGMENT_ID,
                pipeline_id=pipeline_id,
            )
        raw_signal = fragment.raw_relevance_signal
        if raw_signal is None:
            raw_signal = fragment.relevance_score
        snapshots[fragment_id] = ContextFragmentInvariantSnapshot(
            fragment_id=fragment_id,
            source=fragment.source,
            source_id=fragment.source_id,
            provider_provenance=fragment.provider_provenance,
            authority_class=fragment.authority_class,
            sensitivity=fragment.sensitivity,
            scope_ref=fragment.scope_ref,
            canonical_content_hash=content_hash_for_text(fragment.content),
            raw_relevance_signal=raw_signal,
        )
    return snapshots


def _raise_violation(
    *,
    fragment_id: str,
    invariant: ContextPolicyInvariantViolationCode,
    pipeline_id: str,
    expected: str = "",
    actual: str = "",
) -> None:
    raise ContextPolicyInvariantViolationError(
        fragment_id=fragment_id,
        invariant=invariant,
        pipeline_id=pipeline_id,
        expected=expected,
        actual=actual,
    )


def _validate_fragment_against_snapshot(
    fragment: ContextFragment,
    snapshot: ContextFragmentInvariantSnapshot,
    *,
    pipeline_id: str,
) -> None:
    fragment_id = fragment.fragment_id
    if fragment.source is not snapshot.source:
        _raise_violation(
            fragment_id=fragment_id,
            invariant=ContextPolicyInvariantViolationCode.SOURCE_CHANGED,
            pipeline_id=pipeline_id,
            expected=snapshot.source.value,
            actual=fragment.source.value,
        )
    if fragment.source_id != snapshot.source_id:
        _raise_violation(
            fragment_id=fragment_id,
            invariant=ContextPolicyInvariantViolationCode.SOURCE_ID_CHANGED,
            pipeline_id=pipeline_id,
            expected=snapshot.source_id,
            actual=fragment.source_id,
        )
    if fragment.provider_provenance != snapshot.provider_provenance:
        _raise_violation(
            fragment_id=fragment_id,
            invariant=ContextPolicyInvariantViolationCode.PROVENANCE_CHANGED,
            pipeline_id=pipeline_id,
        )
    if fragment.authority_class is not snapshot.authority_class:
        _raise_violation(
            fragment_id=fragment_id,
            invariant=ContextPolicyInvariantViolationCode.AUTHORITY_CHANGED,
            pipeline_id=pipeline_id,
            expected=snapshot.authority_class.value,
            actual=fragment.authority_class.value,
        )
    if fragment.sensitivity is not snapshot.sensitivity:
        _raise_violation(
            fragment_id=fragment_id,
            invariant=ContextPolicyInvariantViolationCode.SENSITIVITY_CHANGED,
            pipeline_id=pipeline_id,
            expected=snapshot.sensitivity.value,
            actual=fragment.sensitivity.value,
        )
    if fragment.scope_ref != snapshot.scope_ref:
        _raise_violation(
            fragment_id=fragment_id,
            invariant=ContextPolicyInvariantViolationCode.SCOPE_CHANGED,
            pipeline_id=pipeline_id,
        )
    recomputed_hash = content_hash_for_text(fragment.content)
    if recomputed_hash != snapshot.canonical_content_hash:
        _raise_violation(
            fragment_id=fragment_id,
            invariant=ContextPolicyInvariantViolationCode.CONTENT_CHANGED,
            pipeline_id=pipeline_id,
        )
    raw_signal = fragment.raw_relevance_signal
    if raw_signal is None:
        raw_signal = fragment.relevance_score
    if raw_signal != snapshot.raw_relevance_signal:
        _raise_violation(
            fragment_id=fragment_id,
            invariant=ContextPolicyInvariantViolationCode.RAW_RELEVANCE_CHANGED,
            pipeline_id=pipeline_id,
        )


def _validate_known_fragment_id(
    fragment_id: str,
    known_ids: frozenset[str],
    *,
    pipeline_id: str,
) -> None:
    if fragment_id not in known_ids:
        _raise_violation(
            fragment_id=fragment_id,
            invariant=ContextPolicyInvariantViolationCode.INVALID_DECISION_REFERENCE,
            pipeline_id=pipeline_id,
        )


def _validate_semantic_dedup_decisions(
    decisions: tuple[ContextSemanticDedupDecision, ...],
    known_ids: frozenset[str],
    *,
    pipeline_id: str,
) -> None:
    for decision in decisions:
        _validate_known_fragment_id(decision.kept_fragment_id, known_ids, pipeline_id=pipeline_id)
        for suppressed_id in decision.suppressed_fragment_ids:
            _validate_known_fragment_id(suppressed_id, known_ids, pipeline_id=pipeline_id)


def _validate_conflict_decisions(
    decisions: tuple[ContextConflictDecision, ...],
    known_ids: frozenset[str],
    *,
    pipeline_id: str,
) -> None:
    for decision in decisions:
        _validate_known_fragment_id(decision.left_fragment_id, known_ids, pipeline_id=pipeline_id)
        _validate_known_fragment_id(decision.right_fragment_id, known_ids, pipeline_id=pipeline_id)
        for kept_id in decision.kept_fragment_ids:
            _validate_known_fragment_id(kept_id, known_ids, pipeline_id=pipeline_id)


def validate_policy_pipeline_result(
    snapshots: dict[str, ContextFragmentInvariantSnapshot],
    result: ContextPolicyPipelineResult,
    *,
    pipeline_id: str,
) -> None:
    """Fail closed when replaceable policy mutates immutable source facts."""
    known_ids = frozenset(snapshots.keys())

    for fragment in result.fragments:
        snapshot = snapshots.get(fragment.fragment_id)
        if snapshot is None:
            _raise_violation(
                fragment_id=fragment.fragment_id,
                invariant=ContextPolicyInvariantViolationCode.UNKNOWN_FRAGMENT,
                pipeline_id=pipeline_id,
            )
        _validate_fragment_against_snapshot(fragment, snapshot, pipeline_id=pipeline_id)

    for excluded_fragment, _reason in result.excluded:
        snapshot = snapshots.get(excluded_fragment.fragment_id)
        if snapshot is None:
            _raise_violation(
                fragment_id=excluded_fragment.fragment_id,
                invariant=ContextPolicyInvariantViolationCode.UNKNOWN_FRAGMENT,
                pipeline_id=pipeline_id,
            )
        _validate_fragment_against_snapshot(excluded_fragment, snapshot, pipeline_id=pipeline_id)

    for decision in result.decisions:
        for fragment_id in decision.input_fragment_ids:
            _validate_known_fragment_id(fragment_id, known_ids, pipeline_id=pipeline_id)
        for fragment_id in decision.output_fragment_ids:
            _validate_known_fragment_id(fragment_id, known_ids, pipeline_id=pipeline_id)

    _validate_semantic_dedup_decisions(
        result.semantic_dedup_decisions,
        known_ids,
        pipeline_id=pipeline_id,
    )
    _validate_conflict_decisions(
        result.conflict_decisions,
        known_ids,
        pipeline_id=pipeline_id,
    )
