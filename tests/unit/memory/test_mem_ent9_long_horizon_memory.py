# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-9 long-horizon memory contracts, store, compaction, recall, plugins."""

from __future__ import annotations

import inspect
from dataclasses import fields, replace
from datetime import datetime, timezone

import pytest

from intergrax.applications._shared.long_horizon_memory_wiring import (
    resolve_long_horizon_memory_capability,
    resolve_long_horizon_memory_store,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.memory.contracts.long_horizon_memory import (
    ChildSummaryRef,
    DefaultLongHorizonSummaryStrategy,
    LineageTraversalRequest,
    LongHorizonCompactionRequest,
    LongHorizonCompactionSource,
    LongHorizonMemoryScope,
    LongHorizonMemoryViolation,
    LongHorizonRecallQuery,
    LongHorizonSummaryRecord,
    MemorySourceRef,
    SummaryGenerationRequest,
    SummaryNodeKind,
    SummaryStatus,
    order_long_horizon_summaries_deterministic,
    validate_long_horizon_summary_record,
)
from intergrax.memory.long_horizon_memory_service import (
    LongHorizonMemoryService,
    build_default_long_horizon_strategies,
)
from intergrax.memory.resolver.discovery import (
    MemoryStorePluginCatalog,
    discover_classified_memory_store_plugins,
)
from intergrax.memory.resolver.errors import MemoryStorePluginResolutionError
from intergrax.memory.resolver.materialization import MemoryStoreMaterializationContext
from intergrax.memory.resolver.resolver import materialize_long_horizon_memory_store
from intergrax.memory.stores.in_memory_long_horizon_memory_plugin import (
    DEFAULT_IN_MEMORY_LONG_HORIZON_PLUGIN_ID,
    InMemoryLongHorizonMemoryStorePlugin,
)
from intergrax.memory.stores.in_memory_long_horizon_memory_store import (
    InMemoryLongHorizonMemoryStore,
)

pytestmark = pytest.mark.gate


def _scope(
    tenant: str,
    user: str = "user-a",
    workspace: str | None = None,
) -> LongHorizonMemoryScope:
    return LongHorizonMemoryScope(tenant_id=tenant, user_id=user, workspace_id=workspace)


def _leaf(
    summary_id: str,
    *,
    revision: int = 1,
    sources: tuple[MemorySourceRef, ...] | None = None,
    covered_from: str | None = "2025-01-01T00:00:00+00:00",
    covered_until: str | None = "2025-01-02T00:00:00+00:00",
) -> LongHorizonSummaryRecord:
    refs = sources or (MemorySourceRef(memory_id="mem-1", revision=1),)
    return LongHorizonSummaryRecord(
        summary_id=summary_id,
        summary_level=1,
        node_kind=SummaryNodeKind.LEAF,
        revision=revision,
        content="leaf summary content",
        source_memory_refs=refs,
        covered_from=covered_from,
        covered_until=covered_until,
        source_count=len(refs),
        created_at="2025-01-01T00:00:00+00:00",
    )


def _parent(
    summary_id: str,
    children: tuple[ChildSummaryRef, ...],
    *,
    revision: int = 1,
    level: int = 2,
) -> LongHorizonSummaryRecord:
    return LongHorizonSummaryRecord(
        summary_id=summary_id,
        summary_level=level,
        node_kind=SummaryNodeKind.AGGREGATE,
        revision=revision,
        content="aggregate summary content",
        child_summary_refs=children,
        covered_from="2025-01-01T00:00:00+00:00",
        covered_until="2025-01-03T00:00:00+00:00",
        source_count=1,
        created_at="2025-01-01T00:00:00+00:00",
    )


def _service(store: InMemoryLongHorizonMemoryStore | None = None) -> LongHorizonMemoryService:
    return LongHorizonMemoryService(
        _store=store or InMemoryLongHorizonMemoryStore(),
        _strategies=build_default_long_horizon_strategies(),
    )


def _sources(*pairs: tuple[str, int]) -> tuple[LongHorizonCompactionSource, ...]:
    return tuple(
        LongHorizonCompactionSource(
            memory_id=memory_id,
            revision=revision,
            content=f"content for {memory_id}",
            observed_at="2025-03-01T12:00:00+00:00",
        )
        for memory_id, revision in pairs
    )


class _StaticRevisionResolver:
    def __init__(self, revisions: dict[str, int | None]) -> None:
        self._revisions = revisions

    def resolve_source_revision(
        self,
        scope: LongHorizonMemoryScope,
        memory_id: str,
    ) -> int | None:
        return self._revisions.get(memory_id)


def test_record_rejects_empty_summary_id() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        _leaf("")


def test_record_rejects_empty_content() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        LongHorizonSummaryRecord(
            summary_id="s1",
            summary_level=1,
            node_kind=SummaryNodeKind.LEAF,
            revision=1,
            content="",
            source_memory_refs=(MemorySourceRef("m", 1),),
            source_count=1,
            created_at="2025-01-01T00:00:00+00:00",
        )


def test_record_rejects_missing_lineage() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        LongHorizonSummaryRecord(
            summary_id="s1",
            summary_level=1,
            node_kind=SummaryNodeKind.LEAF,
            revision=1,
            content="x",
            source_count=0,
            created_at="2025-01-01T00:00:00+00:00",
        )


def test_record_rejects_duplicate_source_refs() -> None:
    refs = (MemorySourceRef("m", 1), MemorySourceRef("m", 1))
    with pytest.raises(LongHorizonMemoryViolation):
        _leaf("s1", sources=refs)


def test_leaf_invariant_pass() -> None:
    record = _leaf("leaf-1")
    validate_long_horizon_summary_record(record)


def test_leaf_without_sources_rejected() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        LongHorizonSummaryRecord(
            summary_id="s1",
            summary_level=1,
            node_kind=SummaryNodeKind.LEAF,
            revision=1,
            content="x",
            child_summary_refs=(ChildSummaryRef("c", 1),),
            source_count=0,
            created_at="2025-01-01T00:00:00+00:00",
        )


def test_parent_invariant_pass() -> None:
    record = _parent("p1", (ChildSummaryRef("c1", 1),))
    validate_long_horizon_summary_record(record)


def test_parent_without_children_rejected() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        LongHorizonSummaryRecord(
            summary_id="p1",
            summary_level=2,
            node_kind=SummaryNodeKind.AGGREGATE,
            revision=1,
            content="x",
            source_memory_refs=(MemorySourceRef("m", 1),),
            source_count=1,
            created_at="2025-01-01T00:00:00+00:00",
        )


def test_self_child_reference_rejected() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        _parent("p1", (ChildSummaryRef("p1", 1),))


def test_temporal_coverage_mixed_awareness_rejected() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        _leaf(
            "s1",
            covered_from="2025-01-01T00:00:00+00:00",
            covered_until="2025-01-02T00:00:00",
        )


def test_temporal_coverage_from_after_until_rejected() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        _leaf(
            "s1",
            covered_from="2025-02-01T00:00:00+00:00",
            covered_until="2025-01-01T00:00:00+00:00",
        )


def test_scope_isolation_same_summary_id() -> None:
    store = InMemoryLongHorizonMemoryStore()
    record = _leaf("shared-id")
    store.upsert_summary(_scope("tenant-a"), record)
    store.upsert_summary(_scope("tenant-b"), record)
    assert store.get_summary(_scope("tenant-a"), "shared-id") is not None
    assert store.get_summary(_scope("tenant-b"), "shared-id") is not None


def test_store_same_revision_conflict() -> None:
    store = InMemoryLongHorizonMemoryStore()
    scope = _scope("T")
    base = _leaf("s1")
    store.upsert_summary(scope, base)
    conflict = replace(base, content="different payload")
    with pytest.raises(LongHorizonMemoryViolation):
        store.upsert_summary(scope, conflict)


def test_store_higher_revision_updates() -> None:
    store = InMemoryLongHorizonMemoryStore()
    scope = _scope("T")
    store.upsert_summary(scope, _leaf("s1", revision=4))
    updated = _leaf("s1", revision=5)
    result = store.upsert_summary(scope, updated)
    assert result.revision == 5


def test_store_stale_incoming_revision_rejected() -> None:
    store = InMemoryLongHorizonMemoryStore()
    scope = _scope("T")
    store.upsert_summary(scope, _leaf("s1", revision=5))
    with pytest.raises(LongHorizonMemoryViolation):
        store.upsert_summary(scope, _leaf("s1", revision=4))


def test_stale_detection_via_resolver() -> None:
    service = _service()
    scope = _scope("T")
    record = _leaf("s1", sources=(MemorySourceRef("mem-x", 4),))
    resolver = _StaticRevisionResolver({"mem-x": 5})
    status = service.validate_summary_sources(scope, record, resolver)
    assert status is SummaryStatus.STALE


def test_source_deletion_invalidates_leaf() -> None:
    service = _service()
    scope = _scope("T")
    record = _leaf("s1", sources=(MemorySourceRef("gone", 1),))
    service._store.upsert_summary(scope, record)
    invalidated = service.invalidate_summaries_for_deleted_source(scope, "gone")
    assert "s1" in invalidated
    stored = service._store.get_summary(scope, "s1")
    assert stored is not None
    assert stored.status is SummaryStatus.INVALIDATED


def test_deterministic_summary_ordering() -> None:
    a = _leaf("a", covered_until="2025-01-01T00:00:00+00:00")
    b = _leaf("b", covered_until="2025-02-01T00:00:00+00:00")
    ordered = order_long_horizon_summaries_deterministic((a, b))
    assert ordered[0].summary_id == "b"
    tie_a = _leaf("z-last", covered_until="2025-02-01T00:00:00+00:00")
    tie_b = _leaf("a-first", covered_until="2025-02-01T00:00:00+00:00")
    ordered_tie = order_long_horizon_summaries_deterministic((tie_a, tie_b))
    assert ordered_tie[0].summary_id == "a-first"


def test_bounded_recall_query() -> None:
    store = InMemoryLongHorizonMemoryStore()
    scope = _scope("T")
    for index in range(5):
        store.upsert_summary(scope, _leaf(f"s{index}"))
    result = store.query_summaries(scope, LongHorizonRecallQuery(limit=3))
    assert len(result) == 3


def test_compaction_idempotent_retry() -> None:
    service = _service()
    scope = _scope("T")
    request = LongHorizonCompactionRequest(
        scope=scope,
        target_level=1,
        sources=_sources(("mem-1", 1), ("mem-2", 1)),
        reference_time=datetime(2025, 3, 1, tzinfo=timezone.utc),
    )
    first = service.compact(request)
    second = service.compact(request)
    assert len(first.created) == 1
    assert not second.created
    assert first.created[0].summary_id in second.skipped_batch_keys


def test_compaction_parent_child_level_invariant() -> None:
    service = _service()
    scope = _scope("T")
    child = _leaf("child", revision=1)
    service._store.upsert_summary(scope, child)
    with pytest.raises(LongHorizonMemoryViolation):
        service.compact(
            LongHorizonCompactionRequest(
                scope=scope,
                target_level=1,
                child_summaries=(child,),
            )
        )


def test_lineage_traversal_to_canonical_sources() -> None:
    service = _service()
    scope = _scope("T")
    leaf = _leaf("leaf", sources=(MemorySourceRef("canonical-1", 3),))
    parent = _parent("parent", (ChildSummaryRef("leaf", 1),), level=2)
    service._store.upsert_summary(scope, leaf)
    service._store.upsert_summary(scope, parent)
    result = service.traverse_lineage(
        scope,
        LineageTraversalRequest(summary_id="parent", max_depth=4, max_nodes=16),
    )
    assert MemorySourceRef("canonical-1", 3) in result.canonical_source_refs


def test_traversal_respects_max_nodes() -> None:
    service = _service()
    scope = _scope("T")
    leaf = _leaf("leaf")
    parent = _parent("parent", (ChildSummaryRef("leaf", 1),))
    service._store.upsert_summary(scope, leaf)
    service._store.upsert_summary(scope, parent)
    result = service.traverse_lineage(
        scope,
        LineageTraversalRequest(summary_id="parent", max_depth=8, max_nodes=1),
    )
    assert result.truncated


def test_traversal_cycle_fail_closed() -> None:
    store = InMemoryLongHorizonMemoryStore()
    scope = _scope("T")
    a = _parent("a", (ChildSummaryRef("b", 1),), level=3)
    b = _parent("b", (ChildSummaryRef("a", 1),), level=2)
    store.upsert_summary(scope, a)
    store.upsert_summary(scope, b)
    service = _service(store)
    with pytest.raises(LongHorizonMemoryViolation):
        service.traverse_lineage(
            scope,
            LineageTraversalRequest(summary_id="a", max_depth=8, max_nodes=32),
        )


def test_strategy_purity_no_store_dependency() -> None:
    strategy = DefaultLongHorizonSummaryStrategy()
    signature = inspect.signature(strategy.generate)
    assert "store" not in signature.parameters


def test_summary_record_has_no_runtime_object_fields() -> None:
    forbidden = {"executor", "task", "coroutine", "runtime"}
    for field in fields(LongHorizonSummaryRecord):
        assert field.name not in forbidden


class _RecordingLongHorizonStore(InMemoryLongHorizonMemoryStore):
    pass


class _FakeLongHorizonMemoryStorePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.fake_long_horizon"

    @classmethod
    def create_long_horizon_memory_store(cls, **kwargs: object) -> _RecordingLongHorizonStore:
        return _RecordingLongHorizonStore()


def test_plugin_resolution_default_in_memory() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(enable_long_horizon_memory=True),
    )
    store = resolve_long_horizon_memory_store(env)
    assert store is not None
    assert isinstance(store, InMemoryLongHorizonMemoryStore)


def test_plugin_resolution_external_provider() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(
            enable_long_horizon_memory=True,
            long_horizon_memory_store_plugin_id=_FakeLongHorizonMemoryStorePlugin.plugin_id(),
        ),
    )
    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=(
            InMemoryLongHorizonMemoryStorePlugin,
            _FakeLongHorizonMemoryStorePlugin,
        ),
    )
    catalog = MemoryStorePluginCatalog.from_discovery(discovery)
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id=None,
        integration_profile=env.integration_profile,
    )
    store = materialize_long_horizon_memory_store(
        _FakeLongHorizonMemoryStorePlugin.plugin_id(),
        ctx,
        catalog=catalog,
    )
    assert isinstance(store, _RecordingLongHorizonStore)


def test_plugin_resolution_invalid_provider_fails() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(
            enable_long_horizon_memory=True,
            long_horizon_memory_store_plugin_id="plugin.does.not.exist",
        ),
    )
    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=(InMemoryLongHorizonMemoryStorePlugin,),
    )
    catalog = MemoryStorePluginCatalog.from_discovery(discovery)
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id=None,
        integration_profile=env.integration_profile,
    )
    with pytest.raises(MemoryStorePluginResolutionError):
        materialize_long_horizon_memory_store("plugin.does.not.exist", ctx, catalog=catalog)


def test_long_horizon_disabled_returns_none() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(enable_long_horizon_memory=False),
    )
    assert resolve_long_horizon_memory_store(env) is None
    assert resolve_long_horizon_memory_capability(env) is None


def test_contracts_no_vendor_imports() -> None:
    import intergrax.memory.contracts.long_horizon_memory as module

    text = open(inspect.getfile(module), encoding="utf-8").read()
    assert "openai" not in text
    assert "pinecone" not in text


def test_default_summary_strategy_generates_non_empty() -> None:
    strategy = DefaultLongHorizonSummaryStrategy()
    result = strategy.generate(
        SummaryGenerationRequest(
            scope=_scope("T"),
            target_level=1,
            node_kind=SummaryNodeKind.LEAF,
            sources=_sources(("m", 1)),
        )
    )
    assert result.content.strip()
