# © Artur Czarnecki. All rights reserved.

"""MP-5D — ContextView source composition port contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.collaborative_work import WorkArtifactVersionRef
from intergrax.contracts.context_view import (
    ContextViewCategory,
    ContextViewCollaborativeWorkSourceRef,
    ContextViewKnowledgeSourceRef,
    ContextViewMemorySourceRef,
    ContextViewOperationScope,
    ContextViewScope,
    ContextViewUclSourceRef,
    ContextViewVisibilityClass,
)
from intergrax.contracts.context_view_source_ports import (
    CollaborativeWorkContextSourcePort,
    ContextViewCollaborativeWorkSourceCandidate,
    ContextViewCollaborativeWorkSourceCandidatesResult,
    ContextViewCollaborativeWorkSourceRequest,
    ContextViewKnowledgeSourceCandidate,
    ContextViewKnowledgeSourceCandidatesResult,
    ContextViewKnowledgeSourceRequest,
    ContextViewMemorySourceCandidate,
    ContextViewMemorySourceCandidatesResult,
    ContextViewMemorySourceRequest,
    ContextViewSourceOutcome,
    ContextViewUclSourceCandidate,
    ContextViewUclSourceCandidatesResult,
    ContextViewUclSourceRequest,
    KnowledgeContextSourcePort,
    MemoryContextSourcePort,
    UclContextSourcePort,
    validate_collaborative_work_source_candidate_isolation,
    validate_knowledge_source_candidate_isolation,
    validate_memory_source_candidate_isolation,
    validate_ucl_source_candidate_isolation,
)

pytestmark = pytest.mark.unit


def _scope(**overrides: object) -> ContextViewScope:
    payload = {"tenant_id": "tenant-a", "workspace_id": "ws-1"}
    payload.update(overrides)
    return ContextViewScope(**payload)


def _memory_request(**overrides: object) -> ContextViewMemorySourceRequest:
    payload = {
        "scope": _scope(),
        "acting_principal_id": "principal-1",
        "eligible_visibility_classes": (ContextViewVisibilityClass.WORKSPACE_SHARED,),
    }
    payload.update(overrides)
    return ContextViewMemorySourceRequest(**payload)


def test_valid_memory_source_request_and_candidate() -> None:
    request = _memory_request()
    candidate = ContextViewMemorySourceCandidate(
        source_ref=ContextViewMemorySourceRef(tenant_id="tenant-a", record_ref="mem-1"),
        candidate_scope=_scope(),
        suggested_visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
    )
    validate_memory_source_candidate_isolation(request=request, candidate=candidate)
    result = ContextViewMemorySourceCandidatesResult(
        outcome=ContextViewSourceOutcome.OK,
        candidates=(candidate,),
    )
    assert result.category == ContextViewCategory.MEMORY


def test_immutability_and_extra_field_rejection() -> None:
    request = _memory_request()
    with pytest.raises(ValidationError):
        ContextViewMemorySourceRequest.model_validate(
            {**request.model_dump(), "payload": "secret"},
        )
    candidate = ContextViewMemorySourceCandidate(
        source_ref=ContextViewMemorySourceRef(tenant_id="tenant-a", record_ref="mem-1"),
        candidate_scope=_scope(),
        suggested_visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
    )
    with pytest.raises(ValidationError):
        ContextViewMemorySourceCandidate.model_validate(
            {**candidate.model_dump(), "record_body": "leak"},
        )


def test_serialization_round_trip_memory_request() -> None:
    request = _memory_request(scope=_scope(work_item_id="wi-1"))
    restored = ContextViewMemorySourceRequest.model_validate_json(request.model_dump_json())
    assert restored == request


def test_non_ok_result_rejects_candidates() -> None:
    candidate = ContextViewMemorySourceCandidate(
        source_ref=ContextViewMemorySourceRef(tenant_id="tenant-a", record_ref="mem-1"),
        candidate_scope=_scope(),
        suggested_visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
    )
    with pytest.raises(ValidationError):
        ContextViewMemorySourceCandidatesResult(
            outcome=ContextViewSourceOutcome.SOURCE_UNAVAILABLE,
            candidates=(candidate,),
        )


def test_cross_tenant_candidate_rejected() -> None:
    request = _memory_request()
    with pytest.raises(ValidationError):
        ContextViewMemorySourceCandidate(
            source_ref=ContextViewMemorySourceRef(tenant_id="tenant-b", record_ref="mem-1"),
            candidate_scope=_scope(),
            suggested_visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
        )
    candidate = ContextViewMemorySourceCandidate(
        source_ref=ContextViewMemorySourceRef(tenant_id="tenant-b", record_ref="mem-1"),
        candidate_scope=_scope(tenant_id="tenant-b"),
        suggested_visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
    )
    with pytest.raises(ValueError, match="tenant_id"):
        validate_memory_source_candidate_isolation(request=request, candidate=candidate)


def test_cross_workspace_collaborative_candidate_rejected() -> None:
    request = ContextViewCollaborativeWorkSourceRequest(
        scope=_scope(),
        acting_principal_id="principal-1",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORK_ITEM,),
    )
    candidate = ContextViewCollaborativeWorkSourceCandidate(
        source_ref=ContextViewCollaborativeWorkSourceRef(
            tenant_id="tenant-a",
            workspace_id="ws-other",
            work_item_id="wi-1",
        ),
        candidate_scope=_scope(workspace_id="ws-other"),
        suggested_visibility=ContextViewVisibilityClass.WORK_ITEM,
    )
    with pytest.raises(ValueError, match="workspace_id"):
        validate_collaborative_work_source_candidate_isolation(request=request, candidate=candidate)


def _collaborative_work_artifact_ref(
    *,
    work_item_id: str,
    tenant_id: str = "tenant-a",
    workspace_id: str = "ws-1",
) -> WorkArtifactVersionRef:
    return WorkArtifactVersionRef(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        work_item_id=work_item_id,
        work_artifact_id="art-1",
        work_artifact_version_id="v-1",
    )


def _collaborative_request(**scope_overrides: object) -> ContextViewCollaborativeWorkSourceRequest:
    return ContextViewCollaborativeWorkSourceRequest(
        scope=_scope(**scope_overrides),
        acting_principal_id="principal-1",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORK_ITEM,),
    )


def test_collaborative_wrapper_and_nested_work_item_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="work_item_id"):
        ContextViewCollaborativeWorkSourceRef(
            tenant_id="tenant-a",
            workspace_id="ws-1",
            work_item_id="wi-1",
            work_artifact_version=_collaborative_work_artifact_ref(work_item_id="wi-2"),
        )


def test_nested_artifact_work_item_mismatch_with_request_scope_rejected() -> None:
    request = _collaborative_request(work_item_id="wi-1")
    with pytest.raises(ValueError, match="work_item_id"):
        ContextViewCollaborativeWorkSourceCandidate(
            source_ref=ContextViewCollaborativeWorkSourceRef(
                tenant_id="tenant-a",
                workspace_id="ws-1",
                work_artifact_version=_collaborative_work_artifact_ref(work_item_id="wi-2"),
            ),
            candidate_scope=_scope(work_item_id="wi-1"),
            suggested_visibility=ContextViewVisibilityClass.WORK_ITEM,
        )
    ref = ContextViewCollaborativeWorkSourceRef(
        tenant_id="tenant-a",
        workspace_id="ws-1",
        work_artifact_version=_collaborative_work_artifact_ref(work_item_id="wi-2"),
    )
    bypass = ContextViewCollaborativeWorkSourceCandidate.model_construct(
        source_ref=ref,
        candidate_scope=_scope(work_item_id="wi-1"),
        suggested_visibility=ContextViewVisibilityClass.WORK_ITEM,
    )
    with pytest.raises(ValueError, match="work_item_id"):
        validate_collaborative_work_source_candidate_isolation(
            request=request,
            candidate=bypass,
        )


def test_nested_only_work_item_locator_passes_isolation() -> None:
    request = _collaborative_request(work_item_id="wi-1")
    candidate = ContextViewCollaborativeWorkSourceCandidate(
        source_ref=ContextViewCollaborativeWorkSourceRef(
            tenant_id="tenant-a",
            workspace_id="ws-1",
            work_artifact_version=_collaborative_work_artifact_ref(work_item_id="wi-1"),
        ),
        candidate_scope=_scope(work_item_id="wi-1"),
        suggested_visibility=ContextViewVisibilityClass.WORK_ITEM,
    )
    validate_collaborative_work_source_candidate_isolation(request=request, candidate=candidate)


def test_workspace_level_collaborative_candidate_with_nested_artifact_passes() -> None:
    request = ContextViewCollaborativeWorkSourceRequest(
        scope=_scope(),
        acting_principal_id="principal-1",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
    )
    candidate = ContextViewCollaborativeWorkSourceCandidate(
        source_ref=ContextViewCollaborativeWorkSourceRef(
            tenant_id="tenant-a",
            workspace_id="ws-1",
            work_artifact_version=_collaborative_work_artifact_ref(work_item_id="wi-specific"),
        ),
        candidate_scope=_scope(),
        suggested_visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
    )
    validate_collaborative_work_source_candidate_isolation(request=request, candidate=candidate)


def test_work_item_mismatch_rejected() -> None:
    request = ContextViewCollaborativeWorkSourceRequest(
        scope=_scope(work_item_id="wi-1"),
        acting_principal_id="principal-1",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORK_ITEM,),
    )
    candidate = ContextViewCollaborativeWorkSourceCandidate(
        source_ref=ContextViewCollaborativeWorkSourceRef(
            tenant_id="tenant-a",
            workspace_id="ws-1",
            work_item_id="wi-2",
        ),
        candidate_scope=_scope(work_item_id="wi-2"),
        suggested_visibility=ContextViewVisibilityClass.WORK_ITEM,
    )
    with pytest.raises(ValueError, match="work_item_id"):
        validate_collaborative_work_source_candidate_isolation(request=request, candidate=candidate)


def test_visibility_escalation_rejected() -> None:
    request = _memory_request(
        eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
    )
    candidate = ContextViewMemorySourceCandidate(
        source_ref=ContextViewMemorySourceRef(tenant_id="tenant-a", record_ref="mem-1"),
        candidate_scope=_scope(),
        suggested_visibility=ContextViewVisibilityClass.PLATFORM_VISIBLE,
    )
    with pytest.raises(ValueError, match="eligible_visibility_classes"):
        validate_memory_source_candidate_isolation(request=request, candidate=candidate)


def test_operation_scope_mismatch_rejected() -> None:
    op_scope = ContextViewOperationScope(operation_id="op-1", resource_scope="res-a")
    request = _memory_request(scope=_scope(operation_scope=op_scope))
    candidate = ContextViewMemorySourceCandidate(
        source_ref=ContextViewMemorySourceRef(tenant_id="tenant-a", record_ref="mem-1"),
        candidate_scope=_scope(
            operation_scope=ContextViewOperationScope(operation_id="op-2"),
        ),
        suggested_visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
    )
    with pytest.raises(ValueError, match="scope"):
        validate_memory_source_candidate_isolation(request=request, candidate=candidate)


class _CustomMemorySource:
    def list_candidates(
        self,
        request: ContextViewMemorySourceRequest,
    ) -> ContextViewMemorySourceCandidatesResult:
        candidate = ContextViewMemorySourceCandidate(
            source_ref=ContextViewMemorySourceRef(
                tenant_id=request.scope.tenant_id,
                record_ref="custom-mem",
            ),
            candidate_scope=request.scope,
            suggested_visibility=request.eligible_visibility_classes[0],
        )
        validate_memory_source_candidate_isolation(request=request, candidate=candidate)
        return ContextViewMemorySourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=(candidate,),
        )


class _CustomKnowledgeSource:
    def list_candidates(
        self,
        request: ContextViewKnowledgeSourceRequest,
    ) -> ContextViewKnowledgeSourceCandidatesResult:
        return ContextViewKnowledgeSourceCandidatesResult(outcome=ContextViewSourceOutcome.OK)


class _CustomUclSource:
    def list_candidates(
        self,
        request: ContextViewUclSourceRequest,
    ) -> ContextViewUclSourceCandidatesResult:
        return ContextViewUclSourceCandidatesResult(outcome=ContextViewSourceOutcome.OK)


class _CustomCollaborativeWorkSource:
    def list_candidates(
        self,
        request: ContextViewCollaborativeWorkSourceRequest,
    ) -> ContextViewCollaborativeWorkSourceCandidatesResult:
        ref = WorkArtifactVersionRef(
            tenant_id=request.scope.tenant_id,
            workspace_id=request.scope.workspace_id,
            work_item_id=request.scope.work_item_id or "wi-default",
            work_artifact_id="art-1",
            work_artifact_version_id="v-1",
        )
        candidate = ContextViewCollaborativeWorkSourceCandidate(
            source_ref=ContextViewCollaborativeWorkSourceRef(
                tenant_id=request.scope.tenant_id,
                workspace_id=request.scope.workspace_id,
                work_artifact_version=ref,
            ),
            candidate_scope=request.scope,
            suggested_visibility=request.eligible_visibility_classes[0],
        )
        validate_collaborative_work_source_candidate_isolation(request=request, candidate=candidate)
        return ContextViewCollaborativeWorkSourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=(candidate,),
        )


def test_pluginability_custom_memory_source() -> None:
    port: MemoryContextSourcePort = _CustomMemorySource()
    result = port.list_candidates(_memory_request())
    assert result.outcome is ContextViewSourceOutcome.OK
    assert len(result.candidates) == 1


def test_pluginability_custom_knowledge_source() -> None:
    port: KnowledgeContextSourcePort = _CustomKnowledgeSource()
    request = ContextViewKnowledgeSourceRequest(
        scope=_scope(),
        acting_principal_id="p",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
    )
    assert port.list_candidates(request).outcome is ContextViewSourceOutcome.OK


def test_pluginability_custom_ucl_source() -> None:
    port: UclContextSourcePort = _CustomUclSource()
    request = ContextViewUclSourceRequest(
        scope=_scope(),
        acting_principal_id="p",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
    )
    assert port.list_candidates(request).outcome is ContextViewSourceOutcome.OK


def test_pluginability_custom_collaborative_work_source() -> None:
    port: CollaborativeWorkContextSourcePort = _CustomCollaborativeWorkSource()
    request = ContextViewCollaborativeWorkSourceRequest(
        scope=_scope(work_item_id="wi-1"),
        acting_principal_id="p",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORK_ITEM,),
    )
    assert port.list_candidates(request).candidates


def test_knowledge_and_ucl_isolation_helpers() -> None:
    knowledge_request = ContextViewKnowledgeSourceRequest(
        scope=_scope(),
        acting_principal_id="p",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
    )
    knowledge_candidate = ContextViewKnowledgeSourceCandidate(
        source_ref=ContextViewKnowledgeSourceRef(tenant_id="tenant-a", knowledge_ref="k-1"),
        candidate_scope=_scope(),
        suggested_visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
    )
    validate_knowledge_source_candidate_isolation(
        request=knowledge_request,
        candidate=knowledge_candidate,
    )

    ucl_request = ContextViewUclSourceRequest(
        scope=_scope(),
        acting_principal_id="p",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
    )
    ucl_candidate = ContextViewUclSourceCandidate(
        source_ref=ContextViewUclSourceRef(tenant_id="tenant-a", ucl_artifact_ref="u-1"),
        candidate_scope=_scope(),
        suggested_visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
    )
    validate_ucl_source_candidate_isolation(request=ucl_request, candidate=ucl_candidate)
