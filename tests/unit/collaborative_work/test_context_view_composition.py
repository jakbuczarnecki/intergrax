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
    ContextViewCompositionSourceFailureError,
    ContextViewCompositionValidatedCandidate,
    ContextViewComposer,
    ContextViewEntryIdentityStrategy,
    ContextViewIdentityStrategy,
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
