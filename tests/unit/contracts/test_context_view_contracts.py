# © Artur Czarnecki. All rights reserved.

"""MP-5B — ContextView public contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.collaborative_work import EffectiveAuthorityRequest, WorkArtifactVersionRef
from intergrax.contracts.context_view import (
    ContextView,
    ContextViewCategory,
    ContextViewCollaborativeWorkSourceRef,
    ContextViewEntry,
    ContextViewKnowledgeSourceRef,
    ContextViewMemorySourceRef,
    ContextViewRequest,
    ContextViewScope,
    ContextViewUclSourceRef,
    ContextViewVisibilityClass,
    validate_context_view_matches_request,
)

pytestmark = pytest.mark.unit


def _scope(**overrides: object) -> ContextViewScope:
    payload = {
        "tenant_id": "tenant-a",
        "workspace_id": "ws-1",
    }
    payload.update(overrides)
    return ContextViewScope(**payload)


def _minimal_request(**overrides: object) -> ContextViewRequest:
    payload = {
        "scope": _scope(),
        "acting_principal_id": "principal-1",
        "operation_id": "op.read_context",
        "requested_categories": (ContextViewCategory.MEMORY,),
    }
    payload.update(overrides)
    return ContextViewRequest(**payload)


def test_valid_minimal_request() -> None:
    request = _minimal_request()
    assert request.scope.tenant_id == "tenant-a"
    assert request.requested_categories == (ContextViewCategory.MEMORY,)


def test_valid_work_item_scoped_view() -> None:
    scope = _scope(work_item_id="wi-42")
    view = ContextView(
        view_id="view-1",
        scope=scope,
        acting_principal_id="principal-1",
        entries=(),
    )
    assert view.scope.work_item_id == "wi-42"


def test_serialization_round_trip_request_and_view() -> None:
    request = _minimal_request(
        scope=_scope(work_item_id="wi-1"),
        requested_categories=(
            ContextViewCategory.MEMORY,
            ContextViewCategory.KNOWLEDGE,
        ),
    )
    restored_request = ContextViewRequest.model_validate_json(request.model_dump_json())
    assert restored_request == request

    entry = ContextViewEntry(
        entry_id="entry-1",
        source_ref=ContextViewMemorySourceRef(tenant_id="tenant-a", record_ref="mem-1"),
        visibility=ContextViewVisibilityClass.PRIVATE_TO_PRINCIPAL,
        entry_scope=_scope(),
    )
    view = ContextView(
        view_id="view-rt",
        scope=_scope(),
        acting_principal_id="principal-1",
        entries=(entry,),
    )
    restored_view = ContextView.model_validate_json(view.model_dump_json())
    assert restored_view == view


def test_immutability_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal_request(extra_field="nope")
    with pytest.raises(ValidationError):
        ContextViewScope(tenant_id="t", workspace_id="w", surprise=True)


def test_invalid_empty_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        _scope(tenant_id="  ")
    with pytest.raises(ValidationError):
        _minimal_request(acting_principal_id="")


def test_cross_tenant_authority_request_rejected() -> None:
    authority = EffectiveAuthorityRequest(
        tenant_id="tenant-b",
        workspace_id="ws-1",
        acting_principal_id="principal-1",
        requested_authority_scopes=("context.read",),
    )
    with pytest.raises(ValidationError):
        _minimal_request(authority_request=authority)


def test_bare_untyped_source_dict_rejected() -> None:
    with pytest.raises(ValidationError):
        ContextViewEntry(
            entry_id="e1",
            source_ref={"record_ref": "mem-1"},
            visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
            entry_scope=_scope(),
        )


def test_unknown_visibility_rejected() -> None:
    with pytest.raises(ValidationError):
        ContextViewEntry(
            entry_id="e1",
            source_ref=ContextViewMemorySourceRef(tenant_id="tenant-a", record_ref="mem-1"),
            visibility="not_a_visibility_class",
            entry_scope=_scope(),
        )


def test_unknown_source_schema_rejected() -> None:
    with pytest.raises(ValidationError):
        ContextViewEntry(
            entry_id="e1",
            source_ref={
                "schema_version": "context_view_unknown_ref.v1",
                "tenant_id": "tenant-a",
                "record_ref": "x",
            },
            visibility=ContextViewVisibilityClass.WORKSPACE_SHARED,
            entry_scope=_scope(),
        )


def test_entry_cross_tenant_source_rejected() -> None:
    entry = ContextViewEntry(
        entry_id="e1",
        source_ref=ContextViewKnowledgeSourceRef(tenant_id="tenant-b", knowledge_ref="k-1"),
        visibility=ContextViewVisibilityClass.WORK_ITEM,
        entry_scope=_scope(),
    )
    with pytest.raises(ValidationError):
        ContextView(
            view_id="view-x",
            scope=_scope(),
            acting_principal_id="principal-1",
            entries=(entry,),
        )


def test_validate_context_view_matches_request_scope() -> None:
    request = _minimal_request(scope=_scope(work_item_id="wi-9"))
    view = ContextView(
        view_id="v",
        scope=_scope(work_item_id="wi-9"),
        acting_principal_id="principal-1",
    )
    validate_context_view_matches_request(view=view, request=request)

    mismatched = ContextView(
        view_id="v",
        scope=_scope(work_item_id="wi-other"),
        acting_principal_id="principal-1",
    )
    with pytest.raises(ValueError, match="work_item_id"):
        validate_context_view_matches_request(view=mismatched, request=request)


def test_collaborative_source_requires_locator() -> None:
    with pytest.raises(ValidationError):
        ContextViewCollaborativeWorkSourceRef(tenant_id="tenant-a", workspace_id="ws-1")


def test_collaborative_source_reuses_work_artifact_version_ref() -> None:
    version_ref = WorkArtifactVersionRef(
        tenant_id="tenant-a",
        workspace_id="ws-1",
        work_item_id="wi-1",
        work_artifact_id="art-1",
        work_artifact_version_id="ver-1",
    )
    source = ContextViewCollaborativeWorkSourceRef(
        tenant_id="tenant-a",
        workspace_id="ws-1",
        work_artifact_version=version_ref,
    )
    assert source.work_artifact_version == version_ref


def test_ucl_and_memory_source_strictness() -> None:
    ucl = ContextViewUclSourceRef(tenant_id="tenant-a", ucl_artifact_ref="ucl-art-1")
    assert ucl.schema_version == "context_view_ucl_source_ref.v1"
    mem = ContextViewMemorySourceRef(tenant_id="tenant-a", record_ref="rec-1")
    assert mem.record_ref == "rec-1"
