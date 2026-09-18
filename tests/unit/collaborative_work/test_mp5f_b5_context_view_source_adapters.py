# © Artur Czarnecki. All rights reserved.

"""MP-5F-B5: ContextView source adapter mapping, outcomes, and pluginability."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.collaborative_work.context_view_composition import DefaultContextViewComposer
from intergrax.collaborative_work.context_view_source_adapters import (
    DefaultCollaborativeWorkContextSource,
    DefaultKnowledgeContextSource,
    DefaultMemoryContextSource,
    DefaultUclContextSource,
)
from intergrax.collaborative_work.context_view_source_mapping import (
    ContextViewSourceAdapterConfigurationError,
    format_context_view_memory_record_ref,
    map_collaborative_work_artifact_ref,
    map_collaborative_work_item_ref,
    map_collaborative_work_version_ref,
    map_knowledge_chunk_to_context_view_ref,
    map_ucl_ref_to_context_view_ref,
)
from intergrax.collaborative_work.context_view_source_wiring import wire_default_context_view_composer
from intergrax.collaborative_work.contracts.collaborative_work_reference_read import (
    CollaborativeWorkArtifactCanonicalRef,
    CollaborativeWorkArtifactVersionCanonicalRef,
    CollaborativeWorkItemCanonicalRef,
    CollaborativeWorkReferenceReadOutcome,
    CollaborativeWorkReferenceReadPort,
    CollaborativeWorkReferenceReadRequest,
    CollaborativeWorkReferenceReadResult,
)
from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.contracts.collaborative_work import WorkItemState
from intergrax.contracts.context_view import (
    ContextViewCollaborativeWorkSourceRef,
    ContextViewKnowledgeSourceRef,
    ContextViewMemorySourceRef,
    ContextViewOperationScope,
    ContextViewScope,
    ContextViewUclSourceRef,
    ContextViewVisibilityClass,
)
from intergrax.contracts.context_view_source_ports import (
    ContextViewCollaborativeWorkSourceRequest,
    ContextViewKnowledgeSourceRequest,
    ContextViewMemorySourceCandidatesResult,
    ContextViewMemorySourceRequest,
    ContextViewSourceOutcome,
    ContextViewUclSourceRequest,
    MemoryContextSourcePort,
)
from intergrax.knowledge.contracts.knowledge_reference_read import (
    KnowledgeChunkCanonicalRef,
    KnowledgeReferenceReadOutcome,
    KnowledgeReferenceReadPort,
    KnowledgeReferenceReadRequest,
    KnowledgeReferenceReadResult,
)
from intergrax.memory.contracts.memory_reference_read import (
    MemoryRecordCanonicalRef,
    MemoryReferenceReadOutcome,
    MemoryReferenceReadPort,
    MemoryReferenceReadRequest,
    MemoryReferenceReadResult,
)
from intergrax.ucl.contracts.ucl_reference_read import (
    UclOptimizationArtifactCanonicalRef,
    UclReferenceReadOutcome,
    UclReferenceReadPort,
    UclReferenceReadRequest,
    UclReferenceReadResult,
    format_ucl_artifact_locator,
)

pytestmark = pytest.mark.unit


class _ImmediateAsyncRunner:
    def run(self, coro):  # type: ignore[no-untyped-def]
        return asyncio.run(coro)


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


def test_memory_mapping_lossless() -> None:
    ref = MemoryRecordCanonicalRef(tenant_id="tenant-a", memory_id="mem-9", revision=3)
    mapped = ContextViewMemorySourceRef(
        tenant_id=ref.tenant_id,
        record_ref=format_context_view_memory_record_ref(ref),
    )
    assert mapped.record_ref == "memory-record/v1/mem-9@3"


def test_knowledge_mapping_preserves_knowledge_ref() -> None:
    ref = KnowledgeChunkCanonicalRef(
        tenant_id="tenant-a",
        knowledge_ref="vec-logical-1",
        document_id="doc-root",
    )
    out = map_knowledge_chunk_to_context_view_ref(ref)
    assert out.knowledge_ref == "vec-logical-1"
    assert out.tenant_id == "tenant-a"


def test_ucl_mapping_uses_canonical_locator() -> None:
    ref = UclOptimizationArtifactCanonicalRef(
        tenant_id="tenant-a",
        workspace_id="ws-1",
        context_scope_id="ctx-1",
        artifact_id="art-1",
        artifact_lookup_key_hash="key-hash",
        artifact_content_hash="content-hash",
        artifact_type="message_sequence",
        lifecycle_status="validated",
    )
    out = map_ucl_ref_to_context_view_ref(ref)
    assert out.ucl_artifact_ref == format_ucl_artifact_locator(ref)


def test_cw_work_item_mapping() -> None:
    ref = CollaborativeWorkItemCanonicalRef(
        tenant_id="tenant-a",
        workspace_id="ws-1",
        work_item_id="wi-1",
        state=WorkItemState.ACTIVE,
    )
    out = map_collaborative_work_item_ref(ref)
    assert out.work_item_id == "wi-1"
    assert out.work_artifact_version is None


def test_cw_artifact_current_version_mapping() -> None:
    ref = CollaborativeWorkArtifactCanonicalRef(
        tenant_id="tenant-a",
        workspace_id="ws-1",
        work_item_id="wi-1",
        work_artifact_id="wa-1",
        current_version_id="ver-9",
    )
    out = map_collaborative_work_artifact_ref(ref)
    version = out.work_artifact_version
    assert version is not None
    assert version.work_artifact_version_id == "ver-9"
    assert version.work_artifact_id == "wa-1"
    assert version.work_item_id == "wi-1"


def test_cw_version_mapping() -> None:
    ref = CollaborativeWorkArtifactVersionCanonicalRef(
        tenant_id="tenant-a",
        workspace_id="ws-1",
        work_item_id="wi-1",
        work_artifact_id="wa-1",
        work_artifact_version_id="ver-2",
    )
    out = map_collaborative_work_version_ref(ref)
    version = out.work_artifact_version
    assert version is not None
    assert version.work_artifact_version_id == "ver-2"


class _FakeMemoryReader:
    def __init__(self, result: MemoryReferenceReadResult) -> None:
        self._result = result

    async def read_references(
        self,
        identity: RequestIdentity,
        request: MemoryReferenceReadRequest,
    ) -> MemoryReferenceReadResult:
        return self._result


def test_memory_adapter_success_and_empty() -> None:
    ref = MemoryRecordCanonicalRef(tenant_id="tenant-a", memory_id="m1", revision=1)
    adapter = DefaultMemoryContextSource(
        reader=_FakeMemoryReader(
            MemoryReferenceReadResult(outcome=MemoryReferenceReadOutcome.OK, references=(ref,)),
        ),
        async_runner=_ImmediateAsyncRunner(),
    )
    ok = adapter.list_candidates(_memory_request())
    assert ok.outcome is ContextViewSourceOutcome.OK
    assert len(ok.candidates) == 1
    assert ok.candidates[0].source_ref.record_ref == format_context_view_memory_record_ref(ref)

    adapter_empty = DefaultMemoryContextSource(
        reader=_FakeMemoryReader(
            MemoryReferenceReadResult(outcome=MemoryReferenceReadOutcome.OK, references=()),
        ),
        async_runner=_ImmediateAsyncRunner(),
    )
    assert adapter_empty.list_candidates(_memory_request()).candidates == ()


def test_memory_cross_tenant_fail_closed() -> None:
    ref = MemoryRecordCanonicalRef(tenant_id="tenant-b", memory_id="m1", revision=1)
    adapter = DefaultMemoryContextSource(
        reader=_FakeMemoryReader(
            MemoryReferenceReadResult(outcome=MemoryReferenceReadOutcome.OK, references=(ref,)),
        ),
        async_runner=_ImmediateAsyncRunner(),
    )
    result = adapter.list_candidates(_memory_request())
    assert result.outcome is ContextViewSourceOutcome.SCOPE_REJECTED
    assert result.candidates == ()


def test_memory_unavailable_propagated() -> None:
    adapter = DefaultMemoryContextSource(
        reader=_FakeMemoryReader(
            MemoryReferenceReadResult(outcome=MemoryReferenceReadOutcome.UNAVAILABLE),
        ),
        async_runner=_ImmediateAsyncRunner(),
    )
    assert (
        adapter.list_candidates(_memory_request()).outcome
        is ContextViewSourceOutcome.SOURCE_UNAVAILABLE
    )


class _FakeKnowledgeReader:
    def __init__(self, result: KnowledgeReferenceReadResult) -> None:
        self._result = result

    def read_references(
        self,
        identity: RequestIdentity,
        request: KnowledgeReferenceReadRequest,
    ) -> KnowledgeReferenceReadResult:
        return self._result


def test_knowledge_scope_rejected_propagated() -> None:
    adapter = DefaultKnowledgeContextSource(
        reader=_FakeKnowledgeReader(
            KnowledgeReferenceReadResult(outcome=KnowledgeReferenceReadOutcome.SCOPE_REJECTED),
        ),
        reference_read_query_text="context-view-enumeration",
    )
    req = ContextViewKnowledgeSourceRequest(
        scope=_scope(),
        acting_principal_id="p1",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
    )
    assert adapter.list_candidates(req).outcome is ContextViewSourceOutcome.SCOPE_REJECTED


class _FakeUclReader:
    def __init__(self, result: UclReferenceReadResult) -> None:
        self._result = result

    async def read_references(
        self,
        identity: RequestIdentity,
        request: UclReferenceReadRequest,
    ) -> UclReferenceReadResult:
        return self._result


def test_ucl_invalid_without_context_scope() -> None:
    adapter = DefaultUclContextSource(
        reader=_FakeUclReader(UclReferenceReadResult(outcome=UclReferenceReadOutcome.OK)),
        async_runner=_ImmediateAsyncRunner(),
    )
    req = ContextViewUclSourceRequest(
        scope=_scope(),
        acting_principal_id="p1",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
    )
    assert adapter.list_candidates(req).outcome is ContextViewSourceOutcome.INVALID_REQUEST


def test_ucl_wrong_context_scope_in_result_fail_closed() -> None:
    ref = UclOptimizationArtifactCanonicalRef(
        tenant_id="tenant-a",
        workspace_id="ws-1",
        context_scope_id="other-ctx",
        artifact_id="a1",
        artifact_lookup_key_hash="kh",
        artifact_content_hash="ch",
        artifact_type="t",
        lifecycle_status="validated",
    )
    adapter = DefaultUclContextSource(
        reader=_FakeUclReader(
            UclReferenceReadResult(outcome=UclReferenceReadOutcome.OK, references=(ref,)),
        ),
        async_runner=_ImmediateAsyncRunner(),
    )
    req = ContextViewUclSourceRequest(
        scope=_scope(
            operation_scope=ContextViewOperationScope(
                operation_id="op-1",
                resource_scope="ctx-1",
            ),
        ),
        acting_principal_id="p1",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
    )
    assert adapter.list_candidates(req).outcome is ContextViewSourceOutcome.SCOPE_REJECTED


class _FakeCwReader:
    def __init__(self, result: CollaborativeWorkReferenceReadResult) -> None:
        self._result = result

    def read_references(
        self,
        identity: RequestIdentity,
        request: CollaborativeWorkReferenceReadRequest,
    ) -> CollaborativeWorkReferenceReadResult:
        return self._result


def test_cw_cross_workspace_fail_closed() -> None:
    ref = CollaborativeWorkItemCanonicalRef(
        tenant_id="tenant-a",
        workspace_id="ws-other",
        work_item_id="wi-1",
        state=WorkItemState.ACTIVE,
    )
    adapter = DefaultCollaborativeWorkContextSource(
        reader=_FakeCwReader(
            CollaborativeWorkReferenceReadResult(
                outcome=CollaborativeWorkReferenceReadOutcome.OK,
                references=(ref,),
            ),
        ),
    )
    req = ContextViewCollaborativeWorkSourceRequest(
        scope=_scope(),
        acting_principal_id="p1",
        eligible_visibility_classes=(ContextViewVisibilityClass.WORK_ITEM,),
    )
    assert adapter.list_candidates(req).outcome is ContextViewSourceOutcome.SCOPE_REJECTED


class _CustomMemoryPort(MemoryContextSourcePort):
    def list_candidates(
        self,
        request: ContextViewMemorySourceRequest,
    ):
        return ContextViewMemorySourceCandidatesResult(outcome=ContextViewSourceOutcome.OK)


def test_composer_accepts_custom_mp5d_port_without_default_adapter_modules() -> None:
    composer = DefaultContextViewComposer(memory_source=_CustomMemoryPort())
    assert composer._memory_source is not None


def test_default_wiring_injects_all_four_adapters() -> None:
    composer = wire_default_context_view_composer(
        memory_reader=_FakeMemoryReader(MemoryReferenceReadResult(outcome=MemoryReferenceReadOutcome.OK)),
        knowledge_reader=_FakeKnowledgeReader(
            KnowledgeReferenceReadResult(outcome=KnowledgeReferenceReadOutcome.OK),
        ),
        ucl_reader=_FakeUclReader(UclReferenceReadResult(outcome=UclReferenceReadOutcome.OK)),
        collaborative_work_reader=_FakeCwReader(
            CollaborativeWorkReferenceReadResult(
                outcome=CollaborativeWorkReferenceReadOutcome.OK,
            ),
        ),
        knowledge_reference_read_query_text="enumerate",
    )
    assert composer._memory_source is not None
    assert composer._knowledge_source is not None
    assert composer._ucl_source is not None
    assert composer._collaborative_work_source is not None


def test_adapter_requires_reader_at_construction() -> None:
    with pytest.raises(ContextViewSourceAdapterConfigurationError):
        DefaultMemoryContextSource(
            reader=None,  # type: ignore[arg-type]
            async_runner=_ImmediateAsyncRunner(),
        )
