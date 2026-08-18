# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.bitemporal_knowledge import (
    KnowledgeOrderingScope,
    KnowledgeRevisionId,
    KnowledgeRevisionPosition,
    KnowledgeRevisionPositionLifecycle,
    KnowledgeRevisionPositionRecord,
    KnowledgeRevisionResolutionReason,
    KnowledgeRevisionResolutionSource,
    KnowledgeRevisionWatermark,
    RevisionFencingGeneration,
    RevisionOrderingAuthority,
    mint_knowledge_revision_id,
    mint_revision_acceptance_key,
)
from intergrax.runtime.observability.canonical_revision_ordering_provider import (
    CanonicalRevisionOrderingProvider,
)
from intergrax.runtime.observability.knowledge_reconstruction import (
    BlockingKnowledgeRevisionPositionError,
    IncompleteKnowledgeRevisionPrefixError,
    reconstruct_knowledge_at_watermark,
)
from intergrax.runtime.observability.revision_ordering_store import RevisionOrderingStoreTestHooks
from tests.unit.contracts.test_bitemporal_revision_ordering import (
    _InMemoryRevisionOrderingAuthority,
)


def _scope(tenant: str = "tenant-a") -> KnowledgeOrderingScope:
    return KnowledgeOrderingScope(tenant_id=tenant)


@dataclass(frozen=True, slots=True)
class _Revision:
    revision_id: KnowledgeRevisionId
    payload: str


class _DictRevisionReader:
    def __init__(self, revisions: dict[KnowledgeRevisionId, _Revision]) -> None:
        self._revisions = revisions

    def load_revision(self, revision_id: KnowledgeRevisionId) -> _Revision:
        return self._revisions[revision_id]


def _reduce_payloads(state: tuple[str, ...], revision: _Revision) -> tuple[str, ...]:
    return state + (revision.payload,)


def _open_provider(
    tmp_path: Path,
    *,
    pause_after_allocate: bool = False,
    allow_late_physical_write: bool = False,
    suffix: str = "ordering.db",
) -> CanonicalRevisionOrderingProvider:
    return CanonicalRevisionOrderingProvider.open(
        str(tmp_path / suffix),
        test_hooks=RevisionOrderingStoreTestHooks(
            pause_after_allocate=pause_after_allocate,
            allow_late_physical_write=allow_late_physical_write,
        ),
    )


def _accept_sequence(
    authority: RevisionOrderingAuthority,
    scope: KnowledgeOrderingScope,
    count: int,
) -> list[tuple[KnowledgeRevisionId, KnowledgeRevisionPosition]]:
    accepted: list[tuple[KnowledgeRevisionId, KnowledgeRevisionPosition]] = []
    for _ in range(count):
        revision = mint_knowledge_revision_id()
        result = authority.accept_revision(
            scope=scope,
            revision_id=revision,
            acceptance_key=mint_revision_acceptance_key(),
        )
        accepted.append((revision, result.position))
    return accepted


@pytest.mark.unit
@pytest.mark.gate
def test_records_through_exposes_accepted_revision_identity(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path)
    scope = _scope()
    revision = mint_knowledge_revision_id()
    accepted = provider.accept_revision(
        scope=scope,
        revision_id=revision,
        acceptance_key=mint_revision_acceptance_key(),
    )
    watermark = provider.watermark(scope)
    records = provider.records_through(watermark)
    assert len(records) == 1
    assert records[0].lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED
    assert records[0].accepted_revision_id == revision
    assert records[0].position == accepted.position


@pytest.mark.unit
@pytest.mark.gate
def test_records_through_terminal_has_no_accepted_revision(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path, pause_after_allocate=True, allow_late_physical_write=True)
    scope = _scope()
    revision = mint_knowledge_revision_id()
    pending = provider.accept_revision(
        scope=scope,
        revision_id=revision,
        acceptance_key=mint_revision_acceptance_key(),
    )
    authority = provider.acquire_resolution_authority(scope)
    provider.resolve_unresolved_position(
        position=pending.position,
        authority=authority,
        reason=KnowledgeRevisionResolutionReason.NO_CANONICAL_DURABLE_ACCEPTANCE,
        source=KnowledgeRevisionResolutionSource.RECOVERY,
    )
    watermark = provider.watermark(scope)
    records = provider.records_through(watermark)
    assert records[0].lifecycle is KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED
    assert records[0].accepted_revision_id is None


@pytest.mark.unit
@pytest.mark.gate
def test_orphaned_physical_write_not_in_accepted_records(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path, pause_after_allocate=True, allow_late_physical_write=True)
    scope = _scope()
    revision = mint_knowledge_revision_id()
    pending = provider.accept_revision(
        scope=scope,
        revision_id=revision,
        acceptance_key=mint_revision_acceptance_key(),
    )
    stale_writer_generation = RevisionFencingGeneration(scope=scope, value=0)
    authority = provider.acquire_resolution_authority(scope)
    provider.resolve_unresolved_position(
        position=pending.position,
        authority=authority,
        reason=KnowledgeRevisionResolutionReason.NO_CANONICAL_DURABLE_ACCEPTANCE,
        source=KnowledgeRevisionResolutionSource.RECOVERY,
    )
    provider.inject_late_physical_write(
        position=pending.position,
        revision_id=revision,
        stale_fencing_generation=stale_writer_generation,
    )
    watermark = provider.watermark(scope)
    records = provider.records_through(watermark)
    assert records[0].accepted_revision_id is None


@pytest.mark.unit
@pytest.mark.gate
def test_reconstruction_reduces_accepted_revisions_in_k_order(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path)
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    payloads: list[str] = []
    for index in range(3):
        revision_id = mint_knowledge_revision_id()
        payload = f"r{index + 1}"
        revisions[revision_id] = _Revision(revision_id=revision_id, payload=payload)
        payloads.append(payload)
        provider.accept_revision(
            scope=scope,
            revision_id=revision_id,
            acceptance_key=mint_revision_acceptance_key(),
        )
    watermark = provider.watermark(scope)
    projection = reconstruct_knowledge_at_watermark(
        provider,
        watermark,
        revision_reader=_DictRevisionReader(revisions),
        reducer=_reduce_payloads,
        initial_state=(),
    )
    assert projection.state == tuple(payloads)
    assert [ref.position.value for ref in projection.accepted_revisions] == [1, 2, 3]


@pytest.mark.unit
@pytest.mark.gate
def test_reconstruction_skips_terminal_non_committed(tmp_path: Path) -> None:
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}

    normal = _open_provider(tmp_path, suffix="terminal.db")
    first_id = mint_knowledge_revision_id()
    revisions[first_id] = _Revision(revision_id=first_id, payload="r1")
    normal.accept_revision(
        scope=scope,
        revision_id=first_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    normal.close()

    paused = CanonicalRevisionOrderingProvider.open(
        str(tmp_path / "terminal.db"),
        test_hooks=RevisionOrderingStoreTestHooks(
            pause_after_allocate=True,
            allow_late_physical_write=True,
        ),
    )
    pending = paused.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    authority = paused.acquire_resolution_authority(scope)
    paused.resolve_unresolved_position(
        position=pending.position,
        authority=authority,
        reason=KnowledgeRevisionResolutionReason.NO_CANONICAL_DURABLE_ACCEPTANCE,
        source=KnowledgeRevisionResolutionSource.RECOVERY,
    )
    paused.close()

    provider = _open_provider(tmp_path, suffix="terminal.db")
    third_id = mint_knowledge_revision_id()
    revisions[third_id] = _Revision(revision_id=third_id, payload="r3")
    provider.accept_revision(
        scope=scope,
        revision_id=third_id,
        acceptance_key=mint_revision_acceptance_key(),
    )

    watermark = provider.watermark(scope)
    projection = reconstruct_knowledge_at_watermark(
        provider,
        watermark,
        revision_reader=_DictRevisionReader(revisions),
        reducer=_reduce_payloads,
        initial_state=(),
    )
    assert projection.state == ("r1", "r3")
    assert [ref.position.value for ref in projection.accepted_revisions] == [1, 3]


@pytest.mark.unit
@pytest.mark.gate
def test_same_watermark_same_reconstruction(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path)
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    accepted = _accept_sequence(provider, scope, 2)
    for revision_id, _ in accepted:
        revisions[revision_id] = _Revision(revision_id=revision_id, payload=revision_id.value)
    watermark = KnowledgeRevisionWatermark(scope=scope, finalized_through_value=2)
    reader = _DictRevisionReader(revisions)
    first = reconstruct_knowledge_at_watermark(
        provider,
        watermark,
        revision_reader=reader,
        reducer=_reduce_payloads,
        initial_state=(),
    )
    second = reconstruct_knowledge_at_watermark(
        provider,
        watermark,
        revision_reader=reader,
        reducer=_reduce_payloads,
        initial_state=(),
    )
    assert first == second


@pytest.mark.unit
@pytest.mark.gate
def test_later_append_does_not_change_earlier_reconstruction(tmp_path: Path) -> None:
    provider = _open_provider(tmp_path)
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    accepted = _accept_sequence(provider, scope, 2)
    for revision_id, _ in accepted:
        revisions[revision_id] = _Revision(revision_id=revision_id, payload=revision_id.value)
    early_watermark = KnowledgeRevisionWatermark(scope=scope, finalized_through_value=2)
    reader = _DictRevisionReader(revisions)
    early = reconstruct_knowledge_at_watermark(
        provider,
        early_watermark,
        revision_reader=reader,
        reducer=_reduce_payloads,
        initial_state=(),
    )

    late_revision = mint_knowledge_revision_id()
    revisions[late_revision] = _Revision(revision_id=late_revision, payload="late")
    provider.accept_revision(
        scope=scope,
        revision_id=late_revision,
        acceptance_key=mint_revision_acceptance_key(),
    )
    replayed = reconstruct_knowledge_at_watermark(
        provider,
        early_watermark,
        revision_reader=reader,
        reducer=_reduce_payloads,
        initial_state=(),
    )
    assert replayed == early


@pytest.mark.unit
@pytest.mark.gate
def test_orphan_detection_after_reconstruction_is_immutable(tmp_path: Path) -> None:
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}

    normal = _open_provider(tmp_path, suffix="orphan.db")
    first_id = mint_knowledge_revision_id()
    revisions[first_id] = _Revision(revision_id=first_id, payload="r1")
    normal.accept_revision(
        scope=scope,
        revision_id=first_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    normal.close()

    paused = CanonicalRevisionOrderingProvider.open(
        str(tmp_path / "orphan.db"),
        test_hooks=RevisionOrderingStoreTestHooks(
            pause_after_allocate=True,
            allow_late_physical_write=True,
        ),
    )
    pending = paused.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    stale_writer_generation = RevisionFencingGeneration(scope=scope, value=0)
    authority = paused.acquire_resolution_authority(scope)
    paused.resolve_unresolved_position(
        position=pending.position,
        authority=authority,
        reason=KnowledgeRevisionResolutionReason.NO_CANONICAL_DURABLE_ACCEPTANCE,
        source=KnowledgeRevisionResolutionSource.RECOVERY,
    )
    paused.close()

    provider = _open_provider(tmp_path, suffix="orphan.db")
    third_id = mint_knowledge_revision_id()
    revisions[third_id] = _Revision(revision_id=third_id, payload="r3")
    provider.accept_revision(
        scope=scope,
        revision_id=third_id,
        acceptance_key=mint_revision_acceptance_key(),
    )

    watermark = provider.watermark(scope)
    reader = _DictRevisionReader(revisions)
    before = reconstruct_knowledge_at_watermark(
        provider,
        watermark,
        revision_reader=reader,
        reducer=_reduce_payloads,
        initial_state=(),
    )
    orphan_provider = _open_provider(
        tmp_path,
        suffix="orphan.db",
        allow_late_physical_write=True,
    )
    orphan_provider.inject_late_physical_write(
        position=pending.position,
        revision_id=mint_knowledge_revision_id(),
        stale_fencing_generation=stale_writer_generation,
    )
    after = reconstruct_knowledge_at_watermark(
        orphan_provider,
        watermark,
        revision_reader=reader,
        reducer=_reduce_payloads,
        initial_state=(),
    )
    assert before == after


@pytest.mark.unit
@pytest.mark.gate
def test_incomplete_prefix_fails_closed() -> None:
    class _GapAuthority(RevisionOrderingAuthority):
        def accept_revision(self, *, scope, revision_id, acceptance_key):
            raise NotImplementedError

        def position_lifecycle(self, position):
            raise NotImplementedError

        def watermark(self, scope):
            raise NotImplementedError

        def records_through(self, watermark):
            revision = mint_knowledge_revision_id()
            return (
                KnowledgeRevisionPositionRecord(
                    position=KnowledgeRevisionPosition(scope=watermark.scope, value=1),
                    lifecycle=KnowledgeRevisionPositionLifecycle.ACCEPTED,
                    accepted_revision_id=revision,
                ),
                KnowledgeRevisionPositionRecord(
                    position=KnowledgeRevisionPosition(scope=watermark.scope, value=3),
                    lifecycle=KnowledgeRevisionPositionLifecycle.ACCEPTED,
                    accepted_revision_id=mint_knowledge_revision_id(),
                ),
            )

        def unresolved_positions(self, scope):
            return ()

        def acquire_resolution_authority(self, scope):
            raise NotImplementedError

        def resolve_unresolved_position(self, **kwargs):
            raise NotImplementedError

    authority = _GapAuthority()
    watermark = KnowledgeRevisionWatermark(scope=_scope(), finalized_through_value=3)
    with pytest.raises(IncompleteKnowledgeRevisionPrefixError):
        reconstruct_knowledge_at_watermark(
            authority,
            watermark,
            revision_reader=_DictRevisionReader({}),
            reducer=_reduce_payloads,
            initial_state=(),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_unresolved_prefix_fails_closed() -> None:
    class _UnresolvedAuthority(RevisionOrderingAuthority):
        def accept_revision(self, *, scope, revision_id, acceptance_key):
            raise NotImplementedError

        def position_lifecycle(self, position):
            raise NotImplementedError

        def watermark(self, scope):
            raise NotImplementedError

        def records_through(self, watermark):
            revision = mint_knowledge_revision_id()
            return (
                KnowledgeRevisionPositionRecord(
                    position=KnowledgeRevisionPosition(scope=watermark.scope, value=1),
                    lifecycle=KnowledgeRevisionPositionLifecycle.ACCEPTED,
                    accepted_revision_id=revision,
                ),
                KnowledgeRevisionPositionRecord(
                    position=KnowledgeRevisionPosition(scope=watermark.scope, value=2),
                    lifecycle=KnowledgeRevisionPositionLifecycle.UNRESOLVED,
                    accepted_revision_id=None,
                ),
            )

        def unresolved_positions(self, scope):
            return ()

        def acquire_resolution_authority(self, scope):
            raise NotImplementedError

        def resolve_unresolved_position(self, **kwargs):
            raise NotImplementedError

    authority = _UnresolvedAuthority()
    watermark = KnowledgeRevisionWatermark(scope=_scope(), finalized_through_value=2)
    with pytest.raises(BlockingKnowledgeRevisionPositionError):
        reconstruct_knowledge_at_watermark(
            authority,
            watermark,
            revision_reader=_DictRevisionReader({}),
            reducer=_reduce_payloads,
            initial_state=(),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_reconstruction_is_provider_independent() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    for index in range(2):
        revision_id = mint_knowledge_revision_id()
        revisions[revision_id] = _Revision(revision_id=revision_id, payload=f"p{index}")
        authority.accept_revision(
            scope=scope,
            revision_id=revision_id,
            acceptance_key=mint_revision_acceptance_key(),
        )
    watermark = authority.watermark(scope)
    projection = reconstruct_knowledge_at_watermark(
        authority,
        watermark,
        revision_reader=_DictRevisionReader(revisions),
        reducer=_reduce_payloads,
        initial_state=(),
    )
    assert projection.state == ("p0", "p1")
    assert len(projection.accepted_revisions) == 2


@pytest.mark.unit
@pytest.mark.gate
def test_reconstruction_survives_provider_reopen(tmp_path: Path) -> None:
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    provider = _open_provider(tmp_path)
    accepted = _accept_sequence(provider, scope, 2)
    for revision_id, _ in accepted:
        revisions[revision_id] = _Revision(revision_id=revision_id, payload=revision_id.value)
    watermark = provider.watermark(scope)
    before = reconstruct_knowledge_at_watermark(
        provider,
        watermark,
        revision_reader=_DictRevisionReader(revisions),
        reducer=_reduce_payloads,
        initial_state=(),
    )
    provider.close()

    reopened = _open_provider(tmp_path)
    after = reconstruct_knowledge_at_watermark(
        reopened,
        watermark,
        revision_reader=_DictRevisionReader(revisions),
        reducer=_reduce_payloads,
        initial_state=(),
    )
    assert after == before
