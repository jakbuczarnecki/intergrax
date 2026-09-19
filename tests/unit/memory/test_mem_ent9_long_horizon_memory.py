# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-9 long-horizon memory contracts, store, compaction, recall, plugins."""

from __future__ import annotations

import inspect
from dataclasses import fields, replace
from datetime import datetime, timezone

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.memory_security_governance_service import build_default_memory_security_governance_service

from intergrax.applications._shared.long_horizon_memory_wiring import (
    resolve_long_horizon_memory_capability,
    resolve_long_horizon_memory_store,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.memory import contracts
from intergrax.memory.contracts import long_horizon_memory as long_horizon_memory_contracts
from intergrax.memory.contracts.long_horizon_memory import (
    CanonicalMemorySourceSnapshot,
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
    long_horizon_batch_identity_key,
    long_horizon_summary_id_for_batch,
    order_long_horizon_summaries_deterministic,
    select_temporal_coverage,
    validate_canonical_source_snapshot,
    validate_long_horizon_summary_record,
)
from intergrax.memory import long_horizon_memory_service
from intergrax.memory.long_horizon_memory_service import (
    LongHorizonMemoryService,
    build_default_long_horizon_strategies,
)
from intergrax.memory.resolver.discovery import (
    MemoryStorePluginCatalog,
    discover_classified_memory_store_plugins,
)
from intergrax.memory.resolver.errors import MemoryStorePluginResolutionError
from intergrax.memory.contracts.memory_store_creation_context import (
    LongHorizonMemoryStoreCreationContext,
)
from intergrax.memory.resolver.materialization import MemoryStoreMaterializationContext
from intergrax.memory.resolver.resolver import materialize_long_horizon_memory_store
from intergrax.memory.stores.in_memory_long_horizon_memory_plugin import (
    DEFAULT_IN_MEMORY_LONG_HORIZON_PLUGIN_ID,
    InMemoryLongHorizonMemoryStorePlugin,
)
from intergrax.memory.stores.in_memory_long_horizon_memory_store import (
    InMemoryLongHorizonMemoryStore,
)
from tests.unit.memory.governance_source_fixtures import (
    PermissiveCanonicalGovernanceSourceAuthority,
)

pytestmark = pytest.mark.gate


def _identity(tenant: str = "T", user: str = "user-a") -> RequestIdentity:
    return RequestIdentity(tenant_id=tenant, user_id=user)



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


class _ScopedSourceAuthority:
    """Fake canonical source authority bound to one scope."""

    def __init__(
        self,
        scope: LongHorizonMemoryScope,
        *,
        snapshots: dict[tuple[str, int], CanonicalMemorySourceSnapshot] | None = None,
        default_observed_at: str = "2025-03-01T12:00:00+00:00",
    ) -> None:
        self._scope = scope
        self._snapshots = snapshots or {}
        self._default_observed_at = default_observed_at

    def resolve_canonical_source(
        self,
        scope: LongHorizonMemoryScope,
        memory_id: str,
        revision: int,
    ) -> CanonicalMemorySourceSnapshot:
        if (
            scope.tenant_id != self._scope.tenant_id
            or scope.user_id != self._scope.user_id
            or scope.workspace_id != self._scope.workspace_id
        ):
            raise LongHorizonMemoryViolation("canonical source scope mismatch")
        key = (memory_id, revision)
        if key in self._snapshots:
            return self._snapshots[key]
        return CanonicalMemorySourceSnapshot(
            memory_id=memory_id,
            revision=revision,
            content=f"content for {memory_id}",
            observed_at=self._default_observed_at,
        )


def _service(
    store: InMemoryLongHorizonMemoryStore | None = None,
    scope: LongHorizonMemoryScope | None = None,
    authority: _ScopedSourceAuthority | None = None,
) -> LongHorizonMemoryService:
    resolved_scope = scope or _scope("T")
    return LongHorizonMemoryService(
        _store=store or InMemoryLongHorizonMemoryStore(),
        _strategies=build_default_long_horizon_strategies(),
        _source_authority=authority or _ScopedSourceAuthority(resolved_scope),
        _governance_source_authority=PermissiveCanonicalGovernanceSourceAuthority(),
        _security_governance=build_default_memory_security_governance_service(),
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
    invalidated = service.invalidate_summaries_for_deleted_source(_identity(), scope, "gone")
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
        identity=_identity(scope.tenant_id, scope.user_id or 'user-a'),
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
        identity=_identity(scope.tenant_id, scope.user_id or 'user-a'),
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
        _identity(scope.tenant_id, scope.user_id or "user-a"),
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
        _identity(scope.tenant_id, scope.user_id or "user-a"),
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
            _identity(scope.tenant_id, scope.user_id or "user-a"),
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
    def create_long_horizon_memory_store(
        cls,
        context: LongHorizonMemoryStoreCreationContext,
    ) -> _RecordingLongHorizonStore:
        _ = context
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


def test_source_authority_rejects_cross_tenant() -> None:
    scope = _scope("tenant-a")
    service = _service(scope=scope)
    other_scope = _scope("tenant-b")
    request = LongHorizonCompactionRequest(
        identity=_identity(scope.tenant_id, scope.user_id or 'user-a'),
        scope=other_scope,
        target_level=1,
        sources=_sources(("mem-1", 1)),
    )
    result = service.compact(request)
    assert result.failures
    assert "scope mismatch" in result.failures[0].message


def test_source_authority_rejects_cross_user() -> None:
    scope = _scope("T", user="user-a")
    service = _service(scope=scope)
    request = LongHorizonCompactionRequest(
        identity=_identity(scope.tenant_id, scope.user_id or 'user-a'),
        scope=_scope("T", user="user-b"),
        target_level=1,
        sources=_sources(("mem-1", 1)),
    )
    result = service.compact(request)
    assert result.failures


def test_source_authority_rejects_cross_workspace() -> None:
    scope = _scope("T", workspace="ws-a")
    service = _service(scope=scope)
    request = LongHorizonCompactionRequest(
        identity=_identity(scope.tenant_id, scope.user_id or 'user-a'),
        scope=_scope("T", workspace="ws-b"),
        target_level=1,
        sources=_sources(("mem-1", 1)),
    )
    result = service.compact(request)
    assert result.failures


def test_source_authority_revision_mismatch_rejects() -> None:
    scope = _scope("T")

    class _ExactRevisionAuthority(_ScopedSourceAuthority):
        def resolve_canonical_source(
            self,
            scope: LongHorizonMemoryScope,
            memory_id: str,
            revision: int,
        ) -> CanonicalMemorySourceSnapshot:
            if revision != 2:
                raise LongHorizonMemoryViolation("canonical source revision mismatch")
            return super().resolve_canonical_source(scope, memory_id, revision)

    service = _service(scope=scope, authority=_ExactRevisionAuthority(scope))
    request = LongHorizonCompactionRequest(
        identity=_identity(scope.tenant_id, scope.user_id or 'user-a'),
        scope=scope,
        target_level=1,
        sources=_sources(("mem-1", 1)),
    )
    result = service.compact(request)
    assert result.failures


def test_source_authority_missing_source_rejects() -> None:
    scope = _scope("T")

    class _MissingAuthority(_ScopedSourceAuthority):
        def resolve_canonical_source(
            self,
            scope: LongHorizonMemoryScope,
            memory_id: str,
            revision: int,
        ) -> CanonicalMemorySourceSnapshot:
            raise LongHorizonMemoryViolation("canonical source not found")

    service = _service(scope=scope, authority=_MissingAuthority(scope))
    result = service.compact(
        LongHorizonCompactionRequest(
        identity=_identity(scope.tenant_id, scope.user_id or 'user-a'),
        scope=scope,
            target_level=1,
            sources=_sources(("mem-1", 1)),
        )
    )
    assert result.failures


def test_compaction_uses_authority_content_not_caller_spoof() -> None:
    scope = _scope("T")
    authority = _ScopedSourceAuthority(
        scope,
        snapshots={
            ("mem-1", 1): CanonicalMemorySourceSnapshot(
                memory_id="mem-1",
                revision=1,
                content="verified canonical body",
                observed_at="2025-03-01T12:00:00+00:00",
            )
        },
    )
    service = _service(scope=scope, authority=authority)
    spoofed = LongHorizonCompactionSource(
        memory_id="mem-1",
        revision=1,
        content="caller spoofed content",
        observed_at="2025-03-01T12:00:00+00:00",
    )
    result = service.compact(
        LongHorizonCompactionRequest(
            identity=_identity(scope.tenant_id, scope.user_id or "user-a"),
            scope=scope,
            target_level=1,
            sources=(spoofed,),
        )
    )
    assert result.created
    assert "verified canonical body" in result.created[0].content
    assert "caller spoofed" not in result.created[0].content


def test_authority_postcondition_rejects_wrong_memory_id() -> None:
    scope = _scope("T")

    class _WrongMemoryIdAuthority(_ScopedSourceAuthority):
        def resolve_canonical_source(
            self,
            scope: LongHorizonMemoryScope,
            memory_id: str,
            revision: int,
        ) -> CanonicalMemorySourceSnapshot:
            snapshot = super().resolve_canonical_source(scope, memory_id, revision)
            return CanonicalMemorySourceSnapshot(
                memory_id="M2",
                revision=snapshot.revision,
                content=snapshot.content,
                observed_at=snapshot.observed_at,
            )

    store = InMemoryLongHorizonMemoryStore()
    service = _service(store=store, scope=scope, authority=_WrongMemoryIdAuthority(scope))
    result = service.compact(
        LongHorizonCompactionRequest(
        identity=_identity(scope.tenant_id, scope.user_id or 'user-a'),
        scope=scope,
            target_level=1,
            sources=_sources(("M1", 4)),
        )
    )
    assert result.failures
    assert not result.created
    assert not result.updated
    assert "unexpected memory_id" in result.failures[0].message
    assert not store.query_summaries(scope, LongHorizonRecallQuery(limit=10))


def test_authority_postcondition_rejects_wrong_revision() -> None:
    scope = _scope("T")

    class _WrongRevisionAuthority(_ScopedSourceAuthority):
        def resolve_canonical_source(
            self,
            scope: LongHorizonMemoryScope,
            memory_id: str,
            revision: int,
        ) -> CanonicalMemorySourceSnapshot:
            snapshot = super().resolve_canonical_source(scope, memory_id, revision)
            return CanonicalMemorySourceSnapshot(
                memory_id=snapshot.memory_id,
                revision=revision + 1,
                content=snapshot.content,
                observed_at=snapshot.observed_at,
            )

    store = InMemoryLongHorizonMemoryStore()
    service = _service(store=store, scope=scope, authority=_WrongRevisionAuthority(scope))
    result = service.compact(
        LongHorizonCompactionRequest(
        identity=_identity(scope.tenant_id, scope.user_id or 'user-a'),
        scope=scope,
            target_level=1,
            sources=_sources(("M1", 4)),
        )
    )
    assert result.failures
    assert not result.created
    assert not result.updated
    assert "unexpected revision" in result.failures[0].message
    assert not store.query_summaries(scope, LongHorizonRecallQuery(limit=10))


def test_authority_postcondition_rejects_wrong_memory_id_and_revision() -> None:
    scope = _scope("T")

    class _WrongIdAndRevisionAuthority(_ScopedSourceAuthority):
        def resolve_canonical_source(
            self,
            scope: LongHorizonMemoryScope,
            memory_id: str,
            revision: int,
        ) -> CanonicalMemorySourceSnapshot:
            snapshot = super().resolve_canonical_source(scope, memory_id, revision)
            return CanonicalMemorySourceSnapshot(
                memory_id="M2",
                revision=revision + 1,
                content=snapshot.content,
                observed_at=snapshot.observed_at,
            )

    service = _service(scope=scope, authority=_WrongIdAndRevisionAuthority(scope))
    result = service.compact(
        LongHorizonCompactionRequest(
        identity=_identity(scope.tenant_id, scope.user_id or 'user-a'),
        scope=scope,
            target_level=1,
            sources=_sources(("M1", 4)),
        )
    )
    assert result.failures
    assert not result.created
    assert not result.updated


def test_authority_postcondition_matching_snapshot_compacts_with_exact_lineage() -> None:
    scope = _scope("T")
    authority = _ScopedSourceAuthority(
        scope,
        snapshots={
            ("M1", 4): CanonicalMemorySourceSnapshot(
                memory_id="M1",
                revision=4,
                content="canonical M1 body",
                observed_at="2025-03-01T12:00:00+00:00",
            )
        },
    )
    service = _service(scope=scope, authority=authority)
    result = service.compact(
        LongHorizonCompactionRequest(
        identity=_identity(scope.tenant_id, scope.user_id or 'user-a'),
        scope=scope,
            target_level=1,
            sources=_sources(("M1", 4)),
        )
    )
    assert result.created
    assert not result.failures
    assert result.created[0].source_memory_refs == (MemorySourceRef(memory_id="M1", revision=4),)


def test_batch_identity_collision_safe_source_refs() -> None:
    a = long_horizon_batch_identity_key(
        source_refs=(MemorySourceRef(memory_id="a@1|b", revision=2),)
    )
    b = long_horizon_batch_identity_key(
        source_refs=(
            MemorySourceRef(memory_id="a", revision=1),
            MemorySourceRef(memory_id="b", revision=2),
        )
    )
    assert a != b


def test_batch_identity_collision_safe_child_refs() -> None:
    a = long_horizon_batch_identity_key(
        child_refs=(ChildSummaryRef(summary_id="x@1|y", revision=3),)
    )
    b = long_horizon_batch_identity_key(
        child_refs=(
            ChildSummaryRef(summary_id="x", revision=1),
            ChildSummaryRef(summary_id="y", revision=3),
        )
    )
    assert a != b


def test_summary_id_collision_safe_scope_segments() -> None:
    scope_a = _scope("T", user="a:b", workspace="c")
    scope_b = _scope("T", user="a", workspace="b:c")
    batch = long_horizon_batch_identity_key(source_refs=(MemorySourceRef("m", 1),))
    assert long_horizon_summary_id_for_batch(scope_a, 1, batch) != long_horizon_summary_id_for_batch(
        scope_b, 1, batch
    )


def test_summary_id_none_workspace_vs_placeholder() -> None:
    batch = long_horizon_batch_identity_key(source_refs=(MemorySourceRef("m", 1),))
    none_ws = long_horizon_summary_id_for_batch(_scope("T", workspace=None), 1, batch)
    underscore_ws = long_horizon_summary_id_for_batch(_scope("T", workspace="_"), 1, batch)
    assert none_ws != underscore_ws


def test_temporal_coverage_aware_offsets() -> None:
    stamps = (
        "2025-06-01T10:00:00+02:00",
        "2025-06-01T09:30:00+00:00",
    )
    covered_from, covered_until = select_temporal_coverage(stamps)
    assert covered_from == "2025-06-01T10:00:00+02:00"
    assert covered_until == "2025-06-01T09:30:00+00:00"


def test_temporal_coverage_naive_year_boundary() -> None:
    covered_from, covered_until = select_temporal_coverage(
        ("2026-01-01T00:00:00", "2025-12-31T23:59:59")
    )
    assert covered_from == "2025-12-31T23:59:59"
    assert covered_until == "2026-01-01T00:00:00"


def test_temporal_coverage_naive_month_boundary() -> None:
    covered_from, covered_until = select_temporal_coverage(
        ("2025-02-01T00:00:00", "2025-01-31T23:59:59")
    )
    assert covered_from == "2025-01-31T23:59:59"
    assert covered_until == "2025-02-01T00:00:00"


def test_temporal_coverage_mixed_awareness_rejects() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        select_temporal_coverage(
            ("2025-01-01T00:00:00", "2025-01-02T00:00:00+00:00")
        )


def test_aggregate_child_coverage_uses_canonical_chronology() -> None:
    strategy = DefaultLongHorizonSummaryStrategy()
    child_a = _leaf(
        "c1",
        covered_from="2025-06-01T10:00:00+02:00",
        covered_until="2025-06-01T11:00:00+02:00",
    )
    child_b = _leaf(
        "c2",
        covered_from="2025-06-01T09:00:00+00:00",
        covered_until="2025-06-01T09:30:00+00:00",
    )
    result = strategy.generate(
        SummaryGenerationRequest(
            scope=_scope("T"),
            target_level=2,
            node_kind=SummaryNodeKind.AGGREGATE,
            child_summaries=(child_a, child_b),
        )
    )
    assert result.covered_from == "2025-06-01T10:00:00+02:00"
    assert result.covered_until == "2025-06-01T09:30:00+00:00"


def test_record_rejects_malformed_single_bound_covered_from() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        _leaf("s1", covered_from="bad", covered_until=None)


def test_recall_query_rejects_malformed_single_bound() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        LongHorizonRecallQuery(covered_from="not-a-timestamp")


def test_recall_query_rejects_reversed_range() -> None:
    with pytest.raises(LongHorizonMemoryViolation):
        LongHorizonRecallQuery(
            covered_from="2025-02-01T00:00:00+00:00",
            covered_until="2025-01-01T00:00:00+00:00",
        )


def test_long_horizon_service_does_not_import_private_canonical_validator() -> None:
    source = inspect.getsource(long_horizon_memory_service)
    assert "_validate_canonical_source_snapshot" not in source


def test_validate_canonical_source_snapshot_is_public_contract() -> None:
    assert hasattr(long_horizon_memory_contracts, "validate_canonical_source_snapshot")
    assert (
        "validate_canonical_source_snapshot"
        in long_horizon_memory_contracts.__all__
    )
    assert validate_canonical_source_snapshot is getattr(
        contracts.long_horizon_memory,
        "validate_canonical_source_snapshot",
    )
