# © Artur Czarnecki. All rights reserved.

"""MP-5G-C1-R1 — pluggable ContextViewScopeCompatibilityPolicy runtime wiring."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.collaborative_work.context_view_composition import DefaultContextViewComposer
from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.contracts.context_view import (
    ContextViewCategory,
    ContextViewMemorySourceRef,
    ContextViewOperationScope,
    ContextViewRequest,
    ContextViewScope,
    ContextViewVisibilityClass,
)
from intergrax.contracts.context_view_composition import (
    ContextViewCompositionCandidateIsolationError,
    ContextViewCompositionRequest,
)
from intergrax.contracts.context_view_scope_compatibility import (
    ContextViewScopeCompatibilityPolicy,
    DefaultContextViewScopeCompatibilityPolicy,
)
from intergrax.contracts.context_view_source_ports import (
    ContextViewMemorySourceCandidate,
    ContextViewMemorySourceCandidatesResult,
    ContextViewMemorySourceRequest,
    ContextViewSourceOutcome,
    MemoryContextSourcePort,
)
from intergrax.contracts.context_view_visibility_policy import (
    ContextViewPolicyDecision,
    ContextViewPolicyOutcome,
    DEFAULT_CONTEXT_VIEW_VISIBILITY_POLICY_ID,
)
from tests.unit.collaborative_work.mp5g_e2e_harness import (
    OP_COMPOSE,
    PRINCIPAL_A,
    TENANT_A,
    WI_A1,
    WS_A,
    build_mp5g_harness,
    context_view_request,
    context_view_scope,
    principal_identity,
    run_qualified_flow,
)
from tests.unit.collaborative_work.test_mp5g_context_view_e2e_qualification import (
    _ALL_CATEGORIES,
    _scope_ucl_context,
)

pytestmark = pytest.mark.unit


def _scope(**overrides: object) -> ContextViewScope:
    payload = {"tenant_id": TENANT_A, "workspace_id": WS_A}
    payload.update(overrides)
    return ContextViewScope(**payload)


def _principal_identity(request: ContextViewRequest) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=request.scope.tenant_id,
        user_id=request.acting_principal_id,
        principal_type=PrincipalType.USER,
        auth_subject=request.acting_principal_id,
    )


def _allow_decision(request: ContextViewRequest) -> ContextViewPolicyDecision:
    return ContextViewPolicyDecision(
        outcome=ContextViewPolicyOutcome.ALLOW,
        policy_id=DEFAULT_CONTEXT_VIEW_VISIBILITY_POLICY_ID,
        effective_scope=request.scope,
        eligible_categories=(ContextViewCategory.MEMORY,),
        eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
    )


@dataclass
class _RecordingMemoryPort(MemoryContextSourcePort):
    result: ContextViewMemorySourceCandidatesResult
    calls: list[ContextViewMemorySourceRequest] = field(default_factory=list)

    def list_candidates(
        self,
        request: ContextViewMemorySourceRequest,
    ) -> ContextViewMemorySourceCandidatesResult:
        self.calls.append(request)
        return self.result


class _StrictExactScopeCompatibilityPolicy:
    def candidate_scope_compatible(
        self,
        *,
        category: ContextViewCategory,
        request_scope: ContextViewScope,
        candidate_scope: ContextViewScope,
    ) -> bool:
        return request_scope == candidate_scope


class _RejectMemoryScopeCompatibilityPolicy:
    def candidate_scope_compatible(
        self,
        *,
        category: ContextViewCategory,
        request_scope: ContextViewScope,
        candidate_scope: ContextViewScope,
    ) -> bool:
        if category is ContextViewCategory.MEMORY:
            return False
        return DefaultContextViewScopeCompatibilityPolicy().candidate_scope_compatible(
            category=category,
            request_scope=request_scope,
            candidate_scope=candidate_scope,
        )


class _ExplodingScopeCompatibilityPolicy:
    def candidate_scope_compatible(
        self,
        *,
        category: ContextViewCategory,
        request_scope: ContextViewScope,
        candidate_scope: ContextViewScope,
    ) -> bool:
        raise RuntimeError("policy internal failure")


@dataclass
class _RecordingScopeCompatibilityPolicy:
    inner: ContextViewScopeCompatibilityPolicy
    calls: list[tuple[ContextViewCategory, ContextViewScope, ContextViewScope]] = field(
        default_factory=list,
    )

    def candidate_scope_compatible(
        self,
        *,
        category: ContextViewCategory,
        request_scope: ContextViewScope,
        candidate_scope: ContextViewScope,
    ) -> bool:
        self.calls.append((category, request_scope, candidate_scope))
        return self.inner.candidate_scope_compatible(
            category=category,
            request_scope=request_scope,
            candidate_scope=candidate_scope,
        )


def _work_item_memory_composition_request() -> tuple[
    ContextViewRequest,
    _RecordingMemoryPort,
    ContextViewPolicyDecision,
]:
    op_scope = ContextViewOperationScope(operation_id=OP_COMPOSE, resource_scope="doc-1")
    scope = _scope(work_item_id=WI_A1, operation_scope=op_scope)
    request = ContextViewRequest(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        operation_id=OP_COMPOSE,
        requested_categories=(ContextViewCategory.MEMORY,),
    )
    workspace_only_candidate = ContextViewMemorySourceCandidate(
        source_ref=ContextViewMemorySourceRef(tenant_id=TENANT_A, record_ref="mem-1"),
        candidate_scope=_scope(),
        suggested_visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
    )
    memory = _RecordingMemoryPort(
        result=ContextViewMemorySourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=(workspace_only_candidate,),
        ),
    )
    return request, memory, _allow_decision(request)


def test_mp5g_c1_r1_strict_policy_rejects_memory_workspace_candidate_in_work_item_view() -> None:
    request, memory, decision = _work_item_memory_composition_request()
    composer = DefaultContextViewComposer(
        memory_source=memory,
        scope_compatibility_policy=_StrictExactScopeCompatibilityPolicy(),
    )
    with pytest.raises(ContextViewCompositionCandidateIsolationError, match="compatible"):
        composer.compose(
            ContextViewCompositionRequest(
                request=request,
                policy_decision=decision,
                principal_identity=_principal_identity(request),
            ),
        )


def test_mp5g_c1_r1_default_policy_admits_memory_workspace_candidate_in_work_item_view() -> None:
    request, memory, decision = _work_item_memory_composition_request()
    composer = DefaultContextViewComposer(memory_source=memory)
    view = composer.compose(
        ContextViewCompositionRequest(
            request=request,
            policy_decision=decision,
            principal_identity=_principal_identity(request),
        ),
    )
    assert len(view.entries) == 1
    assert view.entries[0].entry_scope.work_item_id is None


def test_mp5g_c1_r1_custom_policy_exception_fail_closed() -> None:
    request, memory, decision = _work_item_memory_composition_request()
    composer = DefaultContextViewComposer(
        memory_source=memory,
        scope_compatibility_policy=_ExplodingScopeCompatibilityPolicy(),
    )
    with pytest.raises(ContextViewCompositionCandidateIsolationError, match="policy failed"):
        composer.compose(
            ContextViewCompositionRequest(
                request=request,
                policy_decision=decision,
                principal_identity=_principal_identity(request),
            ),
        )


def test_mp5g_c1_r1_reject_memory_custom_policy_fail_closed() -> None:
    request, memory, decision = _work_item_memory_composition_request()
    composer = DefaultContextViewComposer(
        memory_source=memory,
        scope_compatibility_policy=_RejectMemoryScopeCompatibilityPolicy(),
    )
    with pytest.raises(ContextViewCompositionCandidateIsolationError):
        composer.compose(
            ContextViewCompositionRequest(
                request=request,
                policy_decision=decision,
                principal_identity=_principal_identity(request),
            ),
        )


def test_mp5g_c1_r1_recording_policy_invoked_once_per_category_in_four_source_e2e() -> None:
    inner = DefaultContextViewScopeCompatibilityPolicy()
    recording = _RecordingScopeCompatibilityPolicy(inner=inner)
    harness = build_mp5g_harness(scope_compatibility_policy=recording)
    scope = _scope_ucl_context(work_item_id=WI_A1)
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=_ALL_CATEGORIES,
    )
    identity = principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_A)
    run_qualified_flow(harness, request=request, identity=identity)

    categories_called = {call[0] for call in recording.calls}
    assert categories_called == set(_ALL_CATEGORIES)
    assert len(recording.calls) >= len(_ALL_CATEGORIES)
    for category, request_scope, _candidate_scope in recording.calls:
        if category is ContextViewCategory.MEMORY:
            assert request_scope.work_item_id == WI_A1
        assert request_scope.tenant_id == TENANT_A
