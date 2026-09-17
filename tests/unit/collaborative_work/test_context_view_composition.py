# © Artur Czarnecki. All rights reserved.

"""MP-5E — default ContextView composer tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.collaborative_work.context_view_composition import (
    DefaultContextViewCategoryOrderingStrategy,
    DefaultContextViewComposer,
    Sha256ContextViewEntryIdentityStrategy,
    Sha256ContextViewIdentityStrategy,
)
from intergrax.contracts.context_view import (
    ContextView,
    ContextViewCategory,
    ContextViewMemorySourceRef,
    ContextViewOperationScope,
    ContextViewRequest,
    ContextViewScope,
    ContextViewVisibilityClass,
)
from intergrax.contracts.context_view_composition import (
    ContextViewCandidateOrderingStrategy,
    ContextViewCompositionCandidateIsolationError,
    ContextViewCompositionPolicyDeniedError,
    ContextViewCompositionRequest,
    ContextViewCompositionRequestAlignmentError,
    ContextViewCompositionSourceFailureError,
    ContextViewCompositionValidatedCandidate,
    ContextViewComposer,
    ContextViewEntryIdentityStrategy,
    ContextViewIdentityStrategy,
    effective_scope_within_request_scope,
    validate_context_view_matches_composition_request,
)
from intergrax.contracts.context_view_source_ports import (
    ContextViewKnowledgeSourceCandidatesResult,
    ContextViewMemorySourceCandidate,
    ContextViewMemorySourceCandidatesResult,
    ContextViewMemorySourceRequest,
    ContextViewSourceOutcome,
    MemoryContextSourcePort,
)
from intergrax.contracts.context_view_visibility_policy import (
    ContextViewPolicyDecision,
    ContextViewPolicyDenialReason,
    ContextViewPolicyOutcome,
    DEFAULT_CONTEXT_VIEW_VISIBILITY_POLICY_ID,
    fail_closed_context_view_policy_decision,
)

pytestmark = pytest.mark.unit


def _scope(**overrides: object) -> ContextViewScope:
    payload = {"tenant_id": "tenant-a", "workspace_id": "ws-1"}
    payload.update(overrides)
    return ContextViewScope(**payload)


def _request(**overrides: object) -> ContextViewRequest:
    payload = {
        "scope": _scope(),
        "acting_principal_id": "principal-1",
        "operation_id": "op.read_context",
        "requested_categories": (ContextViewCategory.MEMORY,),
    }
    payload.update(overrides)
    return ContextViewRequest(**payload)


def _request_for_scope(scope: ContextViewScope, **overrides: object) -> ContextViewRequest:
    operation_id = (
        scope.operation_scope.operation_id
        if scope.operation_scope is not None
        else "op.read_context"
    )
    return _request(scope=scope, operation_id=operation_id, **overrides)


def _allow_decision(
    request: ContextViewRequest,
    *,
    eligible_categories: tuple[ContextViewCategory, ...] = (ContextViewCategory.MEMORY,),
    effective_scope: ContextViewScope | None = None,
) -> ContextViewPolicyDecision:
    scope = effective_scope or request.scope
    visibility = (ContextViewVisibilityClass.PRIVATE_TO_PRINCIPAL,)
    private_id = request.acting_principal_id
    return ContextViewPolicyDecision(
        outcome=ContextViewPolicyOutcome.ALLOW,
        policy_id=DEFAULT_CONTEXT_VIEW_VISIBILITY_POLICY_ID,
        effective_scope=scope,
        eligible_categories=eligible_categories,
        eligible_visibility_classes=visibility,
        private_visibility_principal_id=private_id,
    )


@dataclass
class _RecordingMemoryPort(MemoryContextSourcePort):
    calls: list[ContextViewMemorySourceRequest] = field(default_factory=list)
    result: ContextViewMemorySourceCandidatesResult = field(
        default_factory=lambda: ContextViewMemorySourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=(),
        ),
    )

    def list_candidates(
        self,
        request: ContextViewMemorySourceRequest,
    ) -> ContextViewMemorySourceCandidatesResult:
        self.calls.append(request)
        return self.result


@dataclass
class _RecordingKnowledgePort:
    calls: list[object] = field(default_factory=list)

    def list_candidates(self, request: object) -> ContextViewKnowledgeSourceCandidatesResult:
        self.calls.append(request)
        return ContextViewKnowledgeSourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=(),
        )


class _ReverseOrderingStrategy:
    def order(
        self,
        *,
        eligible_category_order: tuple[ContextViewCategory, ...],
        candidates: tuple[ContextViewCompositionValidatedCandidate, ...],
    ) -> tuple[ContextViewCompositionValidatedCandidate, ...]:
        return tuple(reversed(candidates))


class _FixedEntryIdStrategy:
    def __init__(self, entry_id: str) -> None:
        self._entry_id = entry_id

    def entry_id_for_candidate(
        self,
        *,
        candidate: ContextViewCompositionValidatedCandidate,
        composition_request: ContextViewCompositionRequest,
    ) -> str:
        return self._entry_id


class _FixedViewIdStrategy:
    def __init__(self, view_id: str) -> None:
        self._view_id = view_id

    def view_id_for_composition(
        self,
        *,
        composition_request: ContextViewCompositionRequest,
        entry_ids: tuple[str, ...],
    ) -> str:
        return self._view_id


class _StubComposer:
    def compose(self, composition_request: ContextViewCompositionRequest) -> ContextView:
        return ContextView(
            view_id="stub-view",
            scope=composition_request.policy_decision.effective_scope,
            acting_principal_id=composition_request.request.acting_principal_id,
        )


def _memory_candidate(**overrides: object) -> ContextViewMemorySourceCandidate:
    payload = {
        "source_ref": ContextViewMemorySourceRef(tenant_id="tenant-a", record_ref="mem-1"),
        "candidate_scope": _scope(),
        "suggested_visibility": ContextViewVisibilityClass.PRIVATE_TO_PRINCIPAL,
    }
    payload.update(overrides)
    return ContextViewMemorySourceCandidate(**payload)


def test_deny_invokes_zero_source_ports() -> None:
    memory = _RecordingMemoryPort()
    knowledge = _RecordingKnowledgePort()
    request = _request()
    deny = fail_closed_context_view_policy_decision(
        policy_id=DEFAULT_CONTEXT_VIEW_VISIBILITY_POLICY_ID,
        effective_scope=request.scope,
        denial_reason=ContextViewPolicyDenialReason.AUTHORITY_DENIED,
    )
    composer = DefaultContextViewComposer(memory_source=memory, knowledge_source=knowledge)
    with pytest.raises(ContextViewCompositionPolicyDeniedError):
        composer.compose(ContextViewCompositionRequest(request=request, policy_decision=deny))
    assert memory.calls == []
    assert knowledge.calls == []


def test_only_eligible_category_port_invoked() -> None:
    memory = _RecordingMemoryPort()
    knowledge = _RecordingKnowledgePort()
    request = _request(
        requested_categories=(ContextViewCategory.MEMORY, ContextViewCategory.KNOWLEDGE),
    )
    decision = _allow_decision(request, eligible_categories=(ContextViewCategory.MEMORY,))
    DefaultContextViewComposer(memory_source=memory, knowledge_source=knowledge).compose(
        ContextViewCompositionRequest(request=request, policy_decision=decision),
    )
    assert len(memory.calls) == 1
    assert knowledge.calls == []


def test_source_request_uses_policy_effective_scope() -> None:
    memory = _RecordingMemoryPort()
    narrowed = _scope(
        operation_scope=ContextViewOperationScope(
            operation_id="op.read_context",
            resource_scope="res-a",
        ),
    )
    request = _request(
        scope=_scope(
            operation_scope=ContextViewOperationScope(
                operation_id="op.read_context",
                resource_scope="res-a",
            ),
        ),
    )
    decision = _allow_decision(request, effective_scope=narrowed)
    DefaultContextViewComposer(memory_source=memory).compose(
        ContextViewCompositionRequest(request=request, policy_decision=decision),
    )
    assert memory.calls[0].scope == narrowed


def test_source_unavailable_fail_closed() -> None:
    memory = _RecordingMemoryPort(
        result=ContextViewMemorySourceCandidatesResult(
            outcome=ContextViewSourceOutcome.SOURCE_UNAVAILABLE,
        ),
    )
    request = _request()
    decision = _allow_decision(request)
    composer = DefaultContextViewComposer(memory_source=memory)
    with pytest.raises(ContextViewCompositionSourceFailureError, match="source_unavailable"):
        composer.compose(ContextViewCompositionRequest(request=request, policy_decision=decision))


def test_ok_empty_candidates_produces_empty_view() -> None:
    memory = _RecordingMemoryPort()
    request = _request()
    decision = _allow_decision(request)
    view = DefaultContextViewComposer(memory_source=memory).compose(
        ContextViewCompositionRequest(request=request, policy_decision=decision),
    )
    assert view.entries == ()
    assert view.scope == decision.effective_scope
    validate_context_view_matches_composition_request(view=view, composition_request=ContextViewCompositionRequest(request=request, policy_decision=decision))


def test_missing_port_for_eligible_category_fail_closed() -> None:
    request = _request()
    decision = _allow_decision(request)
    composer = DefaultContextViewComposer(memory_source=None)
    with pytest.raises(ContextViewCompositionSourceFailureError, match="missing source port"):
        composer.compose(ContextViewCompositionRequest(request=request, policy_decision=decision))


def test_malicious_cross_tenant_candidate_fails() -> None:
    malicious = ContextViewMemorySourceCandidate.model_construct(
        category=ContextViewCategory.MEMORY,
        source_ref=ContextViewMemorySourceRef(tenant_id="tenant-b", record_ref="mem-1"),
        candidate_scope=_scope(),
        suggested_visibility=ContextViewVisibilityClass.PRIVATE_TO_PRINCIPAL,
    )
    malicious_result = ContextViewMemorySourceCandidatesResult.model_construct(
        outcome=ContextViewSourceOutcome.OK,
        candidates=(malicious,),
    )
    memory = _RecordingMemoryPort(result=malicious_result)
    request = _request()
    decision = _allow_decision(request)
    with pytest.raises(ContextViewCompositionCandidateIsolationError):
        DefaultContextViewComposer(memory_source=memory).compose(
            ContextViewCompositionRequest(request=request, policy_decision=decision),
        )


def test_dedupe_same_source_ref() -> None:
    candidate = _memory_candidate()
    memory = _RecordingMemoryPort(
        result=ContextViewMemorySourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=(candidate, candidate),
        ),
    )
    request = _request()
    decision = _allow_decision(request)
    view = DefaultContextViewComposer(
        memory_source=memory,
        entry_identity_strategy=_FixedEntryIdStrategy("entry-fixed"),
        view_identity_strategy=_FixedViewIdStrategy("view-fixed"),
    ).compose(ContextViewCompositionRequest(request=request, policy_decision=decision))
    assert len(view.entries) == 1


def test_deterministic_composition() -> None:
    candidate = _memory_candidate()
    memory = _RecordingMemoryPort(
        result=ContextViewMemorySourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=(candidate,),
        ),
    )
    request = _request()
    decision = _allow_decision(request)
    composer = DefaultContextViewComposer(memory_source=memory)
    first = composer.compose(ContextViewCompositionRequest(request=request, policy_decision=decision))
    second = composer.compose(ContextViewCompositionRequest(request=request, policy_decision=decision))
    assert first == second


def test_custom_ordering_strategy() -> None:
    candidate_a = _memory_candidate(
        source_ref=ContextViewMemorySourceRef(tenant_id="tenant-a", record_ref="mem-a"),
    )
    candidate_b = _memory_candidate(
        source_ref=ContextViewMemorySourceRef(tenant_id="tenant-a", record_ref="mem-b"),
    )
    memory = _RecordingMemoryPort(
        result=ContextViewMemorySourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=(candidate_a, candidate_b),
        ),
    )
    request = _request()
    decision = _allow_decision(request)
    view = DefaultContextViewComposer(
        memory_source=memory,
        ordering_strategy=_ReverseOrderingStrategy(),
        entry_identity_strategy=Sha256ContextViewEntryIdentityStrategy(),
        view_identity_strategy=_FixedViewIdStrategy("view-fixed"),
    ).compose(ContextViewCompositionRequest(request=request, policy_decision=decision))
    refs = [entry.source_ref.record_ref for entry in view.entries]
    assert refs == ["mem-b", "mem-a"]


def test_composer_protocol_structural_substitution() -> None:
    request = _request()
    decision = _allow_decision(request)
    view = _StubComposer().compose(ContextViewCompositionRequest(request=request, policy_decision=decision))
    assert isinstance(view, ContextView)
    assert isinstance(_StubComposer(), ContextViewComposer)


def test_default_category_ordering_strategy_is_replaceable() -> None:
    strategy = DefaultContextViewCategoryOrderingStrategy()
    assert isinstance(strategy, ContextViewCandidateOrderingStrategy)


def test_identity_strategies_are_replaceable() -> None:
    assert isinstance(Sha256ContextViewEntryIdentityStrategy(), ContextViewEntryIdentityStrategy)
    assert isinstance(Sha256ContextViewIdentityStrategy(), ContextViewIdentityStrategy)


def _compose_with_effective_scope(
    *,
    request_scope: ContextViewScope,
    effective_scope: ContextViewScope,
) -> None:
    memory = _RecordingMemoryPort()
    request = _request_for_scope(request_scope)
    decision = _allow_decision(request, effective_scope=effective_scope)
    DefaultContextViewComposer(memory_source=memory).compose(
        ContextViewCompositionRequest(request=request, policy_decision=decision),
    )


def _compose_with_effective_scope_expect_alignment_error(
    *,
    request_scope: ContextViewScope,
    effective_scope: ContextViewScope,
) -> None:
    memory = _RecordingMemoryPort()
    request = _request_for_scope(request_scope)
    decision = _allow_decision(request, effective_scope=effective_scope)
    with pytest.raises(ContextViewCompositionRequestAlignmentError):
        DefaultContextViewComposer(memory_source=memory).compose(
            ContextViewCompositionRequest(request=request, policy_decision=decision),
        )
    assert memory.calls == []


def test_work_item_broadening_rejected_zero_source_calls() -> None:
    _compose_with_effective_scope_expect_alignment_error(
        request_scope=_scope(work_item_id="wi-1"),
        effective_scope=_scope(),
    )


def test_work_item_narrowing_passes_source_gets_narrowed_work_item() -> None:
    memory = _RecordingMemoryPort()
    narrowed = _scope(work_item_id="wi-1")
    request = _request(scope=_scope())
    decision = _allow_decision(request, effective_scope=narrowed)
    DefaultContextViewComposer(memory_source=memory).compose(
        ContextViewCompositionRequest(request=request, policy_decision=decision),
    )
    assert memory.calls[0].scope.work_item_id == "wi-1"


def test_different_work_item_rejected() -> None:
    _compose_with_effective_scope_expect_alignment_error(
        request_scope=_scope(work_item_id="wi-1"),
        effective_scope=_scope(work_item_id="wi-2"),
    )


def test_operation_broadening_rejected() -> None:
    op = ContextViewOperationScope(operation_id="op-1")
    _compose_with_effective_scope_expect_alignment_error(
        request_scope=_scope(operation_scope=op),
        effective_scope=_scope(operation_scope=None),
    )


def test_operation_narrowing_passes() -> None:
    op = ContextViewOperationScope(operation_id="op-1")
    _compose_with_effective_scope(
        request_scope=_scope(),
        effective_scope=_scope(operation_scope=op),
    )


def test_operation_id_mismatch_rejected() -> None:
    _compose_with_effective_scope_expect_alignment_error(
        request_scope=_scope(
            operation_scope=ContextViewOperationScope(operation_id="op-1"),
        ),
        effective_scope=_scope(
            operation_scope=ContextViewOperationScope(operation_id="op-2"),
        ),
    )


def test_resource_narrowing_passes() -> None:
    op = ContextViewOperationScope(
        operation_id="op.read_context",
        resource_scope="res-1",
    )
    base_op = ContextViewOperationScope(operation_id="op.read_context")
    _compose_with_effective_scope(
        request_scope=_scope(operation_scope=base_op),
        effective_scope=_scope(operation_scope=op),
    )


def test_resource_broadening_rejected() -> None:
    op_with = ContextViewOperationScope(
        operation_id="op.read_context",
        resource_scope="res-1",
    )
    op_without = ContextViewOperationScope(operation_id="op.read_context")
    _compose_with_effective_scope_expect_alignment_error(
        request_scope=_scope(operation_scope=op_with),
        effective_scope=_scope(operation_scope=op_without),
    )


def test_resource_mismatch_rejected() -> None:
    _compose_with_effective_scope_expect_alignment_error(
        request_scope=_scope(
            operation_scope=ContextViewOperationScope(
                operation_id="op.read_context",
                resource_scope="res-1",
            ),
        ),
        effective_scope=_scope(
            operation_scope=ContextViewOperationScope(
                operation_id="op.read_context",
                resource_scope="res-2",
            ),
        ),
    )


@pytest.mark.parametrize(
    ("request_overrides", "effective_overrides", "expected_within"),
    [
        pytest.param({}, {}, True, id="workspace-unchanged"),
        pytest.param({}, {"work_item_id": "wi-1"}, True, id="workspace-to-work-item"),
        pytest.param({"work_item_id": "wi-1"}, {}, False, id="work-item-to-workspace"),
        pytest.param(
            {},
            {"operation_scope": ContextViewOperationScope(operation_id="op-1")},
            True,
            id="no-op-to-op",
        ),
        pytest.param(
            {"operation_scope": ContextViewOperationScope(operation_id="op-1")},
            {},
            False,
            id="op-to-no-op",
        ),
        pytest.param(
            {},
            {
                "operation_scope": ContextViewOperationScope(
                    operation_id="op.read_context",
                    resource_scope="res-1",
                ),
            },
            True,
            id="no-resource-to-resource",
        ),
        pytest.param(
            {
                "operation_scope": ContextViewOperationScope(
                    operation_id="op.read_context",
                    resource_scope="res-1",
                ),
            },
            {
                "operation_scope": ContextViewOperationScope(
                    operation_id="op.read_context",
                ),
            },
            False,
            id="resource-to-no-resource",
        ),
        pytest.param(
            {"work_item_id": "wi-1"},
            {"work_item_id": "wi-2"},
            False,
            id="work-item-mismatch",
        ),
        pytest.param(
            {
                "operation_scope": ContextViewOperationScope(operation_id="op-1"),
            },
            {
                "operation_scope": ContextViewOperationScope(operation_id="op-2"),
            },
            False,
            id="operation-mismatch",
        ),
        pytest.param(
            {"tenant_id": "tenant-a"},
            {"tenant_id": "tenant-b"},
            False,
            id="cross-tenant"),
        pytest.param(
            {"workspace_id": "ws-1"},
            {"workspace_id": "ws-2"},
            False,
            id="cross-workspace"),
    ],
)
def test_effective_scope_subset_relation_table(
    request_overrides: dict[str, object],
    effective_overrides: dict[str, object],
    expected_within: bool,
) -> None:
    request_scope = _scope(**request_overrides)
    effective_scope = _scope(**effective_overrides)
    assert (
        effective_scope_within_request_scope(
            request_scope=request_scope,
            effective_scope=effective_scope,
        )
        is expected_within
    )
