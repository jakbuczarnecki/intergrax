# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from intergrax.contracts.bitemporal_knowledge import (
    KnowledgeOrderingScope,
    KnowledgeRevisionPosition,
    KnowledgeRevisionPositionLifecycle,
    RevisionAcceptanceConflictError,
    StaleRevisionFencingError,
    UnknownKnowledgeRevisionPositionError,
    UnresolvedPositionResolutionError,
    KnowledgeRevisionResolutionReason,
    KnowledgeRevisionResolutionSource,
    mint_knowledge_revision_id,
    mint_revision_acceptance_key,
)
from intergrax.runtime.observability.canonical_revision_ordering_provider import (
    CanonicalRevisionOrderingProvider,
)
from intergrax.runtime.observability.composition import open_revision_ordering_authority
from intergrax.runtime.observability.revision_ordering_store import RevisionOrderingStoreTestHooks
from intergrax.runtime.observability.unresolved_revision_recovery import UnresolvedRevisionRecovery


def _scope(tenant: str = "tenant-a") -> KnowledgeOrderingScope:
    return KnowledgeOrderingScope(tenant_id=tenant)


def _open_provider(
    tmp_path: Path,
    *,
    pause_after_allocate: bool = False,
    allow_late_physical_write: bool = False,
    suffix: str = "ordering.db",
) -> CanonicalRevisionOrderingProvider:
    provider = CanonicalRevisionOrderingProvider.open(
        str(tmp_path / suffix),
        test_hooks=RevisionOrderingStoreTestHooks(
            pause_after_allocate=pause_after_allocate,
            allow_late_physical_write=allow_late_physical_write,
        ),
    )
    assert isinstance(provider, CanonicalRevisionOrderingProvider)
    return provider


@pytest.mark.unit
@pytest.mark.gate
def test_idempotent_retry_returns_same_position(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path)
    scope = _scope()
    key = mint_revision_acceptance_key()
    revision = mint_knowledge_revision_id()
    first = provider.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)
    retry = provider.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)
    assert first.position == retry.position
    assert provider.position_lifecycle(first.position) is KnowledgeRevisionPositionLifecycle.ACCEPTED


@pytest.mark.unit
@pytest.mark.gate
def test_caller_timeout_after_commit_retry(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path)
    scope = _scope()
    key = mint_revision_acceptance_key()
    revision = mint_knowledge_revision_id()
    accepted = provider.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)
    provider.close()
    reopened = _open_provider(tmp_path)
    retry = reopened.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)
    assert retry.position == accepted.position
    assert reopened.watermark(scope).finalized_through_value == accepted.position.value


@pytest.mark.unit
@pytest.mark.gate
def test_concurrent_same_acceptance_single_k(tmp_path: Path) -> None:
    db_path = tmp_path / "shared.db"
    scope = _scope()
    key = mint_revision_acceptance_key()
    revision = mint_knowledge_revision_id()
    barrier = threading.Barrier(2)
    results: list = []
    errors: list[BaseException] = []

    def worker() -> None:
        provider = CanonicalRevisionOrderingProvider.open(str(db_path))
        try:
            barrier.wait(timeout=5)
            results.append(
                provider.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            provider.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker), pool.submit(worker)]
        for future in as_completed(futures):
            future.result()
    assert not errors
    assert len(results) == 2
    assert results[0].position == results[1].position


@pytest.mark.unit
@pytest.mark.gate
def test_concurrent_distinct_acceptance_distinct_positions(tmp_path: Path) -> None:
    db_path = tmp_path / "shared.db"
    scope = _scope()
    barrier = threading.Barrier(2)
    results = []

    def worker() -> None:
        provider = CanonicalRevisionOrderingProvider.open(str(db_path))
        try:
            barrier.wait(timeout=5)
            results.append(
                provider.accept_revision(
                    scope=scope,
                    revision_id=mint_knowledge_revision_id(),
                    acceptance_key=mint_revision_acceptance_key(),
                )
            )
        finally:
            provider.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker), pool.submit(worker)]
        for future in as_completed(futures):
            future.result()
    positions = sorted(result.position.value for result in results)
    assert positions == [1, 2]


@pytest.mark.unit
@pytest.mark.gate
def test_conflict_same_key_different_revision(tmp_path: Path) -> None:
    db_path = tmp_path / "shared.db"
    scope = _scope()
    key = mint_revision_acceptance_key()
    provider_one = CanonicalRevisionOrderingProvider.open(str(db_path))
    provider_two = CanonicalRevisionOrderingProvider.open(str(db_path))
    provider_one.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=key,
    )
    with pytest.raises(RevisionAcceptanceConflictError):
        provider_two.accept_revision(
            scope=scope,
            revision_id=mint_knowledge_revision_id(),
            acceptance_key=key,
        )
    assert provider_two.watermark(scope).finalized_through_value == 1
    provider_one.close()
    provider_two.close()


@pytest.mark.unit
@pytest.mark.gate
def test_restart_preserves_state(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path, pause_after_allocate=True)
    scope = _scope()
    key = mint_revision_acceptance_key()
    revision = mint_knowledge_revision_id()
    pending = provider.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)
    assert provider.position_lifecycle(pending.position) is KnowledgeRevisionPositionLifecycle.UNRESOLVED
    provider.close()

    reopened = _open_provider(tmp_path, pause_after_allocate=True)
    assert reopened.position_lifecycle(pending.position) is KnowledgeRevisionPositionLifecycle.UNRESOLVED
    assert reopened.unresolved_positions(scope) == (pending.position,)
    completed = reopened.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)
    assert completed.position == pending.position
    assert reopened.position_lifecycle(pending.position) is KnowledgeRevisionPositionLifecycle.ACCEPTED


@pytest.mark.unit
@pytest.mark.gate
def test_watermark_cases(tmp_path: Path) -> None:
    simple_db = tmp_path / "simple.db"
    scope = _scope("tenant-watermark")
    provider = CanonicalRevisionOrderingProvider.open(str(simple_db))
    provider.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    provider.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    assert provider.watermark(scope).finalized_through_value == 2
    provider.close()

    gap_db = tmp_path / "gap.db"
    gap_scope = _scope("tenant-gap")
    normal = CanonicalRevisionOrderingProvider.open(str(gap_db))
    normal.accept_revision(
        scope=gap_scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    normal.close()

    paused = CanonicalRevisionOrderingProvider.open(
        str(gap_db),
        test_hooks=RevisionOrderingStoreTestHooks(pause_after_allocate=True),
    )
    unresolved = paused.accept_revision(
        scope=gap_scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    paused.close()

    normal = CanonicalRevisionOrderingProvider.open(str(gap_db))
    normal.accept_revision(
        scope=gap_scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    assert normal.watermark(gap_scope).finalized_through_value == 1
    normal.close()

    resolver = CanonicalRevisionOrderingProvider.open(str(gap_db))
    authority = resolver.acquire_resolution_authority(gap_scope)
    resolver.resolve_unresolved_position(
        position=unresolved.position,
        authority=authority,
        reason=KnowledgeRevisionResolutionReason.NO_CANONICAL_DURABLE_ACCEPTANCE,
        source=KnowledgeRevisionResolutionSource.RECOVERY,
    )
    assert resolver.watermark(gap_scope).finalized_through_value == 3
    resolver.close()


@pytest.mark.unit
@pytest.mark.gate
def test_unknown_position_raises(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path)
    scope = _scope()
    with pytest.raises(UnknownKnowledgeRevisionPositionError):
        provider.position_lifecycle(KnowledgeRevisionPosition(scope=scope, value=99))


@pytest.mark.unit
@pytest.mark.gate
def test_resolution_race_acceptance_first(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path, pause_after_allocate=True)
    scope = _scope()
    key = mint_revision_acceptance_key()
    revision = mint_knowledge_revision_id()
    pending = provider.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)
    assert provider.position_lifecycle(pending.position) is KnowledgeRevisionPositionLifecycle.UNRESOLVED
    completed = provider.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)
    assert completed.position == pending.position
    assert provider.position_lifecycle(pending.position) is KnowledgeRevisionPositionLifecycle.ACCEPTED
    authority = provider.acquire_resolution_authority(scope)
    with pytest.raises(UnresolvedPositionResolutionError):
        provider.resolve_unresolved_position(
            position=pending.position,
            authority=authority,
            reason=KnowledgeRevisionResolutionReason.NO_CANONICAL_DURABLE_ACCEPTANCE,
            source=KnowledgeRevisionResolutionSource.RECOVERY,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_resolution_race_terminalization_first(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path, pause_after_allocate=True)
    scope = _scope()
    key = mint_revision_acceptance_key()
    revision = mint_knowledge_revision_id()
    pending = provider.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)
    authority = provider.acquire_resolution_authority(scope)
    provider.resolve_unresolved_position(
        position=pending.position,
        authority=authority,
        reason=KnowledgeRevisionResolutionReason.NO_CANONICAL_DURABLE_ACCEPTANCE,
        source=KnowledgeRevisionResolutionSource.RECOVERY,
    )
    assert provider.position_lifecycle(pending.position) is (
        KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED
    )
    with pytest.raises(StaleRevisionFencingError):
        provider.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)


@pytest.mark.unit
@pytest.mark.gate
def test_orphan_late_physical_write(tmp_path: Path) -> None:
    provider = _open_provider(
        tmp_path,
        pause_after_allocate=True,
        allow_late_physical_write=True,
    )
    scope = _scope()
    key = mint_revision_acceptance_key()
    revision = mint_knowledge_revision_id()
    pending = provider.accept_revision(scope=scope, revision_id=revision, acceptance_key=key)
    authority = provider.acquire_resolution_authority(scope)
    provider.resolve_unresolved_position(
        position=pending.position,
        authority=authority,
        reason=KnowledgeRevisionResolutionReason.STALE_FENCE_SUPERSEDED,
        source=KnowledgeRevisionResolutionSource.RECOVERY,
    )
    watermark_before = provider.watermark(scope)
    orphan = provider.inject_late_physical_write(
        position=pending.position,
        revision_id=revision,
        stale_fencing_generation=authority,
    )
    assert orphan is not None
    assert provider.position_lifecycle(pending.position) is (
        KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED
    )
    assert provider.watermark(scope) == watermark_before
    assert provider.list_orphan_records(scope)


@pytest.mark.unit
@pytest.mark.gate
def test_historical_immutability_after_orphan(tmp_path: Path) -> None:
    provider = _open_provider(
        tmp_path,
        pause_after_allocate=True,
        allow_late_physical_write=True,
    )
    scope = _scope()
    for _ in range(16):
        provider.accept_revision(
            scope=scope,
            revision_id=mint_knowledge_revision_id(),
            acceptance_key=mint_revision_acceptance_key(),
        )
    pending = provider.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    authority = provider.acquire_resolution_authority(scope)
    provider.resolve_unresolved_position(
        position=pending.position,
        authority=authority,
        reason=KnowledgeRevisionResolutionReason.NO_CANONICAL_DURABLE_ACCEPTANCE,
        source=KnowledgeRevisionResolutionSource.RECOVERY,
    )
    provider.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    provider.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    watermark = provider.watermark(scope)
    historical_before = provider.records_through(watermark)
    provider.inject_late_physical_write(
        position=pending.position,
        revision_id=mint_knowledge_revision_id(),
        stale_fencing_generation=authority,
    )
    historical_after = provider.records_through(watermark)
    assert historical_before == historical_after


@pytest.mark.unit
@pytest.mark.gate
def test_tenant_isolation(tmp_path: Path) -> None:
    db_path = tmp_path / "tenant.db"
    scope_a = _scope("tenant-a")
    scope_b = _scope("tenant-b")

    normal = CanonicalRevisionOrderingProvider.open(str(db_path))
    normal.accept_revision(
        scope=scope_a,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    normal.accept_revision(
        scope=scope_b,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    normal.close()

    paused = CanonicalRevisionOrderingProvider.open(
        str(db_path),
        test_hooks=RevisionOrderingStoreTestHooks(pause_after_allocate=True),
    )
    unresolved_a = paused.accept_revision(
        scope=scope_a,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    assert paused.watermark(scope_a).finalized_through_value == 1
    assert paused.watermark(scope_b).finalized_through_value == 1
    assert paused.unresolved_positions(scope_a) == (unresolved_a.position,)
    assert paused.unresolved_positions(scope_b) == ()
    paused.close()


@pytest.mark.unit
@pytest.mark.gate
def test_recovery_component(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path, pause_after_allocate=True)
    scope = _scope()
    pending = provider.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    recovery = UnresolvedRevisionRecovery(provider)
    result = recovery.recover_scope(scope)
    assert len(result.resolved) == 1
    assert result.remaining_unresolved == ()
    assert provider.position_lifecycle(pending.position) is (
        KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED
    )


@pytest.mark.unit
@pytest.mark.gate
def test_composition_opens_authority(tmp_path: Path) -> None:
    authority = open_revision_ordering_authority(db_path=tmp_path / "composed.db")
    scope = _scope()
    accepted = authority.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    assert accepted.position.value == 1
    if isinstance(authority, CanonicalRevisionOrderingProvider):
        authority.close()
