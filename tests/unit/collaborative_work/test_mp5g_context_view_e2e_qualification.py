# © Artur Czarnecki. All rights reserved.

"""MP-5G — end-to-end ContextView isolation and contract-chain qualification."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest
from pydantic import ValidationError

from intergrax.collaborative_work.context_view_composition import DefaultContextViewComposer
from intergrax.collaborative_work.context_view_source_adapters import (
    DefaultCollaborativeWorkContextSource,
    DefaultKnowledgeContextSource,
    DefaultMemoryContextSource,
)
from intergrax.collaborative_work.context_view_source_wiring import (
    DefaultContextViewAsyncReferenceReadRunner,
)
from intergrax.collaborative_work.contracts.collaborative_work_reference_read import (
    CollaborativeWorkArtifactCanonicalRef,
    CollaborativeWorkItemCanonicalRef,
    CollaborativeWorkReferenceReadOutcome,
    CollaborativeWorkReferenceReadResult,
    CollaborativeWorkReferenceReadScope,
)
from intergrax.contracts.agent_run import PrincipalType
from intergrax.contracts.context_view import (
    ContextViewCategory,
    ContextViewCollaborativeWorkSourceRef,
    ContextViewKnowledgeSourceRef,
    ContextViewMemorySourceRef,
    ContextViewOperationScope,
    ContextViewUclSourceRef,
)
from intergrax.contracts.context_view_composition import (
    ContextViewCompositionPolicyDeniedError,
    ContextViewCompositionRequest,
    ContextViewCompositionRequestAlignmentError,
    ContextViewCompositionSourceFailureError,
)
from intergrax.contracts.context_view_source_ports import (
    ContextViewMemorySourceCandidatesResult,
    ContextViewSourceOutcome,
    MemoryContextSourcePort,
)
from intergrax.contracts.context_view_visibility_policy import (
    CONTEXT_VIEW_READ_AUTHORITY_SCOPE,
    ContextViewCategoryDenialReason,
    ContextViewPolicyOutcome,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.knowledge.contracts.knowledge_reference_read import (
    KnowledgeChunkCanonicalRef,
    KnowledgeReferenceReadOutcome,
    KnowledgeReferenceReadResult,
    KnowledgeReferenceReadScope,
)
from intergrax.memory.contracts.memory_reference_read import (
    MemoryRecordCanonicalRef,
    MemoryReferenceReadOutcome,
    MemoryReferenceReadResult,
    MemoryReferenceReadScope,
)
from intergrax.ucl.contracts.ucl_reference_read import (
    UclOptimizationArtifactCanonicalRef,
    UclReferenceReadOutcome,
    UclReferenceReadResult,
    UclReferenceReadScope,
)

from tests.unit.collaborative_work.mp5g_e2e_harness import (
    CTX_A,
    CTX_B,
    CTX_X,
    DOC_A1,
    DOC_A2,
    DOC_B1,
    DOC_X1,
    KNOWLEDGE_QUERY,
    OP_COMPOSE,
    PRINCIPAL_A,
    PRINCIPAL_B,
    PRINCIPAL_C,
    SERVICE_PRINCIPAL,
    TENANT_A,
    TENANT_B,
    WI_A1,
    WI_A2,
    WI_B1,
    WI_X1,
    WS_A,
    WS_B,
    WS_X,
    ImmediateAsyncRunner,
    Mp5gHarness,
    assert_no_payload_fields,
    build_mp5g_harness,
    context_view_request,
    context_view_scope,
    principal_identity,
    run_qualified_flow,
)

pytestmark = pytest.mark.unit

_ALL_CATEGORIES = (
    ContextViewCategory.MEMORY,
    ContextViewCategory.KNOWLEDGE,
    ContextViewCategory.UCL_CONTEXT_LIFECYCLE,
    ContextViewCategory.COLLABORATIVE_WORK,
)


@pytest.fixture(scope="module")
def mp5g_harness() -> Mp5gHarness:
    return build_mp5g_harness()


def _scope_ucl_context(
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WS_A,
    context_id: str = CTX_A,
    work_item_id: str | None = None,
) -> object:
    return context_view_scope(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        work_item_id=work_item_id,
        operation_id=OP_COMPOSE,
        resource_scope=context_id,
    )


def _scope_knowledge_document(
    document_id: str,
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WS_A,
    work_item_id: str | None = None,
) -> object:
    return context_view_scope(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        work_item_id=work_item_id,
        operation_id=OP_COMPOSE,
        resource_scope=document_id,
    )


def test_mp5g_e2e_happy_path_authority_policy_composer_four_categories(
    mp5g_harness: Mp5gHarness,
) -> None:
    """Full chain: evaluator → composer → B5 adapters → default readers (workspace scope)."""
    scope = _scope_ucl_context()
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=_ALL_CATEGORIES,
    )
    identity = principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_A)
    decision, view = run_qualified_flow(mp5g_harness, request=request, identity=identity)

    assert decision.outcome is ContextViewPolicyOutcome.ALLOW
    assert view.scope.tenant_id == TENANT_A
    assert view.scope.workspace_id == WS_A
    assert view.acting_principal_id == PRINCIPAL_A

    def _entry_category(entry: object) -> ContextViewCategory:
        ref = entry.source_ref  # type: ignore[attr-defined]
        if isinstance(ref, ContextViewMemorySourceRef):
            return ContextViewCategory.MEMORY
        if isinstance(ref, ContextViewKnowledgeSourceRef):
            return ContextViewCategory.KNOWLEDGE
        if isinstance(ref, ContextViewUclSourceRef):
            return ContextViewCategory.UCL_CONTEXT_LIFECYCLE
        return ContextViewCategory.COLLABORATIVE_WORK

    categories_present = {_entry_category(entry) for entry in view.entries}
    assert ContextViewCategory.MEMORY in categories_present
    assert ContextViewCategory.UCL_CONTEXT_LIFECYCLE in categories_present
    assert ContextViewCategory.COLLABORATIVE_WORK in categories_present

    for entry in view.entries:
        assert entry.entry_scope.tenant_id == TENANT_A
        assert entry.entry_scope.workspace_id == WS_A
        assert_no_payload_fields(entry.source_ref)
        if isinstance(entry.source_ref, ContextViewMemorySourceRef):
            assert entry.source_ref.tenant_id == TENANT_A
        if isinstance(entry.source_ref, ContextViewUclSourceRef):
            assert entry.source_ref.tenant_id == TENANT_A
        if isinstance(entry.source_ref, ContextViewCollaborativeWorkSourceRef):
            assert entry.source_ref.workspace_id == WS_A

    tenant_b_leaks = [
        e
        for e in view.entries
        if getattr(e.source_ref, "tenant_id", TENANT_A) == TENANT_B
        or getattr(e.source_ref, "workspace_id", WS_A) in {WS_B, WS_X}
    ]
    assert not tenant_b_leaks

    assert len(mp5g_harness.memory_port.calls) == 1
    assert len(mp5g_harness.knowledge_port.calls) == 1
    assert len(mp5g_harness.ucl_port.calls) == 1
    assert len(mp5g_harness.collaborative_work_port.calls) == 1
    assert mp5g_harness.memory_port.calls[0].scope == decision.effective_scope


def test_mp5g_e2e_deterministic_view_three_runs(mp5g_harness: Mp5gHarness) -> None:
    scope = _scope_ucl_context()
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=_ALL_CATEGORIES,
    )
    identity = principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_A)
    views = [
        run_qualified_flow(mp5g_harness, request=request, identity=identity)[1]
        for _ in range(3)
    ]
    assert views[0] == views[1] == views[2]


def test_mp5g_cross_tenant_isolation_principal_a_never_sees_tenant_b(
    mp5g_harness: Mp5gHarness,
) -> None:
    scope = _scope_ucl_context()
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=_ALL_CATEGORIES,
    )
    identity = principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_A)
    _, view = run_qualified_flow(mp5g_harness, request=request, identity=identity)
    for entry in view.entries:
        ref = entry.source_ref
        tenant = getattr(ref, "tenant_id", None)
        assert tenant != TENANT_B
        assert DOC_X1 not in str(ref)
        assert WI_X1 not in str(ref)


@pytest.mark.parametrize(
    ("workspace_id", "principal_id", "must_contain_ws"),
    [
        (WS_A, PRINCIPAL_A, WS_A),
        (WS_B, PRINCIPAL_B, WS_B),
    ],
)
def test_mp5g_cross_workspace_isolation_per_source(
    mp5g_harness: Mp5gHarness,
    workspace_id: str,
    principal_id: str,
    must_contain_ws: str,
) -> None:
    context_id = CTX_A if workspace_id == WS_A else CTX_B
    scope = _scope_ucl_context(workspace_id=workspace_id, context_id=context_id)
    request = context_view_request(
        scope=scope,
        acting_principal_id=principal_id,
        categories=_ALL_CATEGORIES,
    )
    identity = principal_identity(tenant_id=TENANT_A, principal_id=principal_id)
    _, view = run_qualified_flow(mp5g_harness, request=request, identity=identity)
    for entry in view.entries:
        ws = getattr(entry.source_ref, "workspace_id", workspace_id)
        if ws is not None:
            assert ws == must_contain_ws
        assert WS_X not in str(entry.source_ref)


def test_mp5g_work_item_isolation_collaborative_work(
    mp5g_harness: Mp5gHarness,
) -> None:
    scope = _scope_ucl_context(work_item_id=WI_A1)
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=(ContextViewCategory.COLLABORATIVE_WORK,),
    )
    identity = principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_A)
    _, view = run_qualified_flow(mp5g_harness, request=request, identity=identity)
    for entry in view.entries:
        ref = entry.source_ref
        assert isinstance(ref, ContextViewCollaborativeWorkSourceRef)
        if ref.work_item_id is not None:
            assert ref.work_item_id == WI_A1
        assert WI_A2 not in str(ref)


def test_mp5g_knowledge_document_resource_isolation(
    mp5g_harness: Mp5gHarness,
) -> None:
    scope = _scope_knowledge_document(DOC_A1)
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=(ContextViewCategory.KNOWLEDGE,),
    )
    identity = principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_A)
    _, view = run_qualified_flow(mp5g_harness, request=request, identity=identity)
    knowledge_refs = [
        e.source_ref.knowledge_ref
        for e in view.entries
        if isinstance(e.source_ref, ContextViewKnowledgeSourceRef)
    ]
    assert "vec-a1" in knowledge_refs
    assert "vec-a2" not in knowledge_refs


def test_mp5g_service_principal_type_preserved_to_memory_reader(
    mp5g_harness: Mp5gHarness,
) -> None:
    scope = context_view_scope(tenant_id=TENANT_A, workspace_id=WS_A)
    request = context_view_request(
        scope=scope,
        acting_principal_id=SERVICE_PRINCIPAL,
        categories=(ContextViewCategory.MEMORY,),
    )
    identity = principal_identity(
        tenant_id=TENANT_A,
        principal_id=SERVICE_PRINCIPAL,
        principal_type=PrincipalType.SERVICE,
    )
    run_qualified_flow(mp5g_harness, request=request, identity=identity)
    assert mp5g_harness.memory_reader.calls
    seen_identity = mp5g_harness.memory_reader.calls[-1][0]
    assert seen_identity.principal_type is PrincipalType.SERVICE


def test_mp5g_acting_principal_mismatch_fails_before_compose(
    mp5g_harness: Mp5gHarness,
) -> None:
    scope = context_view_scope(tenant_id=TENANT_A, workspace_id=WS_A)
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=(ContextViewCategory.MEMORY,),
    )
    identity = principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_B)
    with pytest.raises(ValidationError):
        ContextViewCompositionRequest(
            request=request,
            policy_decision=mp5g_harness.evaluator.evaluate(request),
            principal_identity=identity,
        )


def test_mp5g_identity_tenant_mismatch_scope_fails_closed(
    mp5g_harness: Mp5gHarness,
) -> None:
    scope = context_view_scope(tenant_id=TENANT_A, workspace_id=WS_A)
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_C,
        categories=(ContextViewCategory.MEMORY,),
    )
    identity = principal_identity(tenant_id=TENANT_B, principal_id=PRINCIPAL_C)
    with pytest.raises(ValidationError):
        ContextViewCompositionRequest(
            request=request,
            policy_decision=mp5g_harness.evaluator.evaluate(request),
            principal_identity=identity,
        )


def test_mp5g_policy_deny_never_invokes_source_ports(mp5g_harness: Mp5gHarness) -> None:
    scope = context_view_scope(tenant_id=TENANT_B, workspace_id=WS_X)
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=(ContextViewCategory.MEMORY,),
    )
    identity = principal_identity(tenant_id=TENANT_B, principal_id=PRINCIPAL_A)
    decision = mp5g_harness.evaluator.evaluate(request)
    assert decision.outcome is ContextViewPolicyOutcome.DENY
    memory_calls_before = len(mp5g_harness.memory_port.calls)
    with pytest.raises(ContextViewCompositionPolicyDeniedError):
        mp5g_harness.composer.compose(
            ContextViewCompositionRequest(
                request=request,
                policy_decision=decision,
                principal_identity=identity,
            ),
        )
    assert len(mp5g_harness.memory_port.calls) == memory_calls_before


def test_mp5g_partial_category_eligibility_skips_denied_adapter(
    mp5g_harness: Mp5gHarness,
) -> None:
    delegation_repo = mp5g_harness.delegation_repo
    membership_repo = mp5g_harness.membership_repo
    authority_repo = mp5g_harness.authority_repo
    delegator = "principal-delegator-mp5g"
    from intergrax.collaborative_work.repository import (
        CreateAuthorityDelegationCommand,
        CreatePrincipalAuthorityGrantCommand,
        CreateWorkspaceMembershipCommand,
    )

    delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            delegation_id="deleg-mp5g",
            delegator_principal_id=delegator,
            delegate_principal_id=PRINCIPAL_A,
            authority_scopes=(CONTEXT_VIEW_READ_AUTHORITY_SCOPE,),
        )
    )
    from intergrax.contracts.collaborative_work import WorkspaceMembershipRole

    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            membership_id="mem-delegator",
            principal_id=delegator,
            role=WorkspaceMembershipRole.MEMBER,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            authority_grant_id="grant-delegator",
            principal_id=delegator,
            authority_scopes=(CONTEXT_VIEW_READ_AUTHORITY_SCOPE,),
        )
    )
    locator = delegation_repo.get(tenant_id=TENANT_A, workspace_id=WS_A, delegation_id="deleg-mp5g")
    assert locator is not None
    scope = context_view_scope(tenant_id=TENANT_A, workspace_id=WS_A)
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=(ContextViewCategory.MEMORY, ContextViewCategory.KNOWLEDGE),
        delegator_principal_id=delegator,
        delegation=locator,
    )
    identity = principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_A)
    knowledge_calls_before = len(mp5g_harness.knowledge_port.calls)
    decision, view = run_qualified_flow(mp5g_harness, request=request, identity=identity)
    assert ContextViewCategory.MEMORY in decision.eligible_categories
    assert ContextViewCategory.KNOWLEDGE not in decision.eligible_categories
    assert len(mp5g_harness.knowledge_port.calls) == knowledge_calls_before
    assert all(isinstance(e.source_ref, ContextViewMemorySourceRef) for e in view.entries)


def test_mp5g_source_unavailable_fail_fast_memory(mp5g_harness: Mp5gHarness) -> None:
    class _UnavailableMemory:
        async def read_references(self, identity, request):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    runner = ImmediateAsyncRunner()
    port = DefaultMemoryContextSource(reader=_UnavailableMemory(), async_runner=runner)
    composer = DefaultContextViewComposer(
        memory_source=port,
        knowledge_source=None,
        ucl_source=None,
        collaborative_work_source=None,
    )
    scope = context_view_scope(tenant_id=TENANT_A, workspace_id=WS_A)
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=(ContextViewCategory.MEMORY,),
    )
    decision = mp5g_harness.evaluator.evaluate(request)
    identity = principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_A)
    with pytest.raises(ContextViewCompositionSourceFailureError):
        composer.compose(
            ContextViewCompositionRequest(
                request=request,
                policy_decision=decision,
                principal_identity=identity,
            ),
        )


def test_mp5g_malicious_memory_wrong_workspace_rejected_e2e(
    mp5g_harness: Mp5gHarness,
) -> None:
    class _MaliciousMemory:
        async def read_references(self, identity, request):  # type: ignore[no-untyped-def]
            return MemoryReferenceReadResult(
                outcome=MemoryReferenceReadOutcome.OK,
                references=(
                    MemoryRecordCanonicalRef(
                        tenant_id=TENANT_A,
                        memory_id="evil",
                        revision=1,
                    ),
                ),
                evaluated_scope=MemoryReferenceReadScope(
                    tenant_id=TENANT_A,
                    workspace_id=WS_B,
                    user_id=PRINCIPAL_A,
                ),
            )

    runner = ImmediateAsyncRunner()
    composer = DefaultContextViewComposer(
        memory_source=DefaultMemoryContextSource(reader=_MaliciousMemory(), async_runner=runner),
        knowledge_source=None,
        ucl_source=None,
        collaborative_work_source=None,
    )
    scope = context_view_scope(tenant_id=TENANT_A, workspace_id=WS_A)
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=(ContextViewCategory.MEMORY,),
    )
    decision = mp5g_harness.evaluator.evaluate(request)
    identity = principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_A)
    with pytest.raises(ContextViewCompositionSourceFailureError):
        composer.compose(
            ContextViewCompositionRequest(
                request=request,
                policy_decision=decision,
                principal_identity=identity,
            ),
        )


def test_mp5g_pluginability_custom_mp5d_port(mp5g_harness: Mp5gHarness) -> None:
    class _CustomPort(MemoryContextSourcePort):
        def list_candidates(self, request):  # type: ignore[no-untyped-def]
            return ContextViewMemorySourceCandidatesResult(outcome=ContextViewSourceOutcome.OK)

    composer = DefaultContextViewComposer(memory_source=_CustomPort())
    assert composer._memory_source is not None


def test_mp5g_pluginability_custom_source_reader_via_default_adapter(
    mp5g_harness: Mp5gHarness,
) -> None:
    class _CustomKnowledge:
        def read_references(self, identity, request):  # type: ignore[no-untyped-def]
            return KnowledgeReferenceReadResult(
                outcome=KnowledgeReferenceReadOutcome.OK,
                references=(
                    KnowledgeChunkCanonicalRef(
                        tenant_id=TENANT_A,
                        knowledge_ref="custom-k",
                        document_id=DOC_A1,
                    ),
                ),
                evaluated_scope=KnowledgeReferenceReadScope(
                    tenant_id=TENANT_A,
                    workspace_id=WS_A,
                ),
            )

    port = DefaultKnowledgeContextSource(reader=_CustomKnowledge())
    scope = context_view_scope(tenant_id=TENANT_A, workspace_id=WS_A)
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=(ContextViewCategory.KNOWLEDGE,),
    )
    decision = mp5g_harness.evaluator.evaluate(request)
    composer = DefaultContextViewComposer(
        knowledge_source=port,
        config=mp5g_harness.composer._config,
    )
    view = composer.compose(
        ContextViewCompositionRequest(
            request=request,
            policy_decision=decision,
            principal_identity=principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_A),
        ),
    )
    assert any(
        isinstance(e.source_ref, ContextViewKnowledgeSourceRef)
        and e.source_ref.knowledge_ref == "custom-k"
        for e in view.entries
    )


def test_mp5g_async_runner_rejects_active_event_loop() -> None:
    runner = DefaultContextViewAsyncReferenceReadRunner()

    async def _coro() -> int:
        return 1

    async def _inside_loop() -> None:
        with pytest.raises(RuntimeError, match="active event loop"):
            runner.run(_coro())

    asyncio.run(_inside_loop())


def test_mp5g_security_matrix_workspace_b_isolated_from_a(
    mp5g_harness: Mp5gHarness,
) -> None:
    scope = _scope_ucl_context(workspace_id=WS_A, context_id=CTX_A)
    request = context_view_request(
        scope=scope,
        acting_principal_id=PRINCIPAL_A,
        categories=_ALL_CATEGORIES,
    )
    identity = principal_identity(tenant_id=TENANT_A, principal_id=PRINCIPAL_A)
    _, view = run_qualified_flow(mp5g_harness, request=request, identity=identity)
    serialized = view.model_dump_json()
    for forbidden in (WI_B1, DOC_B1, CTX_B, WS_B, TENANT_B, WI_X1):
        assert forbidden not in serialized
