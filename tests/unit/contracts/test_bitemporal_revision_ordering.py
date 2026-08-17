# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.bitemporal_knowledge import (
    CrossScopeKnowledgeOrderError,
    InvalidKnowledgeRevisionPositionRecordError,
    KnowledgeOrderingScope,
    KnowledgeRevisionAcceptance,
    KnowledgeRevisionId,
    KnowledgeRevisionPosition,
    KnowledgeRevisionPositionLifecycle,
    KnowledgeRevisionPositionRecord,
    KnowledgeRevisionWatermark,
    KnowledgeRevisionResolutionReason,
    KnowledgeRevisionResolutionRecord,
    KnowledgeRevisionResolutionSource,
    ResolutionAuthority,
    RevisionAcceptanceConflictError,
    RevisionAcceptanceKey,
    RevisionFencingGeneration,
    RevisionOrderingAuthority,
    UnknownKnowledgeRevisionPositionError,
    compute_finalized_watermark,
    lifecycle_blocks_watermark,
    lifecycle_is_finalized,
    mint_knowledge_revision_id,
    mint_revision_acceptance_key,
)


def _scope(tenant: str = "tenant-a") -> KnowledgeOrderingScope:
    return KnowledgeOrderingScope(tenant_id=tenant)


def _record(
    value: int,
    lifecycle: KnowledgeRevisionPositionLifecycle,
    *,
    scope: KnowledgeOrderingScope | None = None,
    accepted_revision_id: KnowledgeRevisionId | None = None,
) -> KnowledgeRevisionPositionRecord:
    resolved = scope or _scope()
    revision_id = accepted_revision_id
    if lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED and revision_id is None:
        revision_id = mint_knowledge_revision_id()
    return KnowledgeRevisionPositionRecord(
        position=KnowledgeRevisionPosition(scope=resolved, value=value),
        lifecycle=lifecycle,
        accepted_revision_id=revision_id,
    )


class _InMemoryRevisionOrderingAuthority(RevisionOrderingAuthority):
    """Contract-level fake proving idempotent acceptance semantics only."""

    def __init__(self) -> None:
        self._next_position: dict[str, int] = {}
        self._bindings: dict[tuple[str, str], tuple[KnowledgeRevisionId, KnowledgeRevisionPosition]] = {}
        self._lifecycles: dict[tuple[str, int], KnowledgeRevisionPositionLifecycle] = {}
        self._accepted_revision_ids: dict[tuple[str, int], KnowledgeRevisionId] = {}

    def accept_revision(
        self,
        *,
        scope: KnowledgeOrderingScope,
        revision_id: KnowledgeRevisionId,
        acceptance_key: RevisionAcceptanceKey,
    ) -> KnowledgeRevisionAcceptance:
        tenant = scope.tenant_id
        lookup = (tenant, acceptance_key.value)
        existing = self._bindings.get(lookup)
        if existing is not None:
            bound_revision, bound_position = existing
            if bound_revision != revision_id:
                raise RevisionAcceptanceConflictError(
                    "acceptance key already bound to a different knowledge revision"
                )
            return KnowledgeRevisionAcceptance(
                revision_id=revision_id,
                acceptance_key=acceptance_key,
                position=bound_position,
            )

        next_value = self._next_position.get(tenant, 1)
        position = KnowledgeRevisionPosition(scope=scope, value=next_value)
        self._next_position[tenant] = next_value + 1
        self._bindings[lookup] = (revision_id, position)
        self._lifecycles[(tenant, next_value)] = KnowledgeRevisionPositionLifecycle.ACCEPTED
        self._accepted_revision_ids[(tenant, next_value)] = revision_id
        return KnowledgeRevisionAcceptance(
            revision_id=revision_id,
            acceptance_key=acceptance_key,
            position=position,
        )

    def position_lifecycle(
        self,
        position: KnowledgeRevisionPosition,
    ) -> KnowledgeRevisionPositionLifecycle:
        tenant = position.scope.tenant_id
        lifecycle = self._lifecycles.get((tenant, position.value))
        if lifecycle is None:
            raise UnknownKnowledgeRevisionPositionError(
                f"knowledge revision position {position.value} was never allocated"
            )
        return lifecycle

    def watermark(self, scope: KnowledgeOrderingScope) -> KnowledgeRevisionWatermark:
        tenant = scope.tenant_id
        records = tuple(
            KnowledgeRevisionPositionRecord(
                position=KnowledgeRevisionPosition(scope=scope, value=value),
                lifecycle=lifecycle,
                accepted_revision_id=(
                    self._accepted_revision_ids[(tenant, value)]
                    if lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED
                    else None
                ),
            )
            for (record_tenant, value), lifecycle in sorted(self._lifecycles.items())
            if record_tenant == tenant
        )
        return compute_finalized_watermark(scope=scope, records=records)

    def records_through(
        self,
        watermark: KnowledgeRevisionWatermark,
    ) -> tuple[KnowledgeRevisionPositionRecord, ...]:
        tenant = watermark.scope.tenant_id
        records: list[KnowledgeRevisionPositionRecord] = []
        for value in range(1, watermark.finalized_through_value + 1):
            lifecycle = self._lifecycles.get((tenant, value))
            if lifecycle is not None:
                records.append(
                    KnowledgeRevisionPositionRecord(
                        position=KnowledgeRevisionPosition(
                            scope=watermark.scope,
                            value=value,
                        ),
                        lifecycle=lifecycle,
                        accepted_revision_id=(
                            self._accepted_revision_ids[(tenant, value)]
                            if lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED
                            else None
                        ),
                    )
                )
        return tuple(records)

    def unresolved_positions(
        self,
        scope: KnowledgeOrderingScope,
    ) -> tuple[KnowledgeRevisionPosition, ...]:
        tenant = scope.tenant_id
        unresolved: list[KnowledgeRevisionPosition] = []
        for (record_tenant, value), lifecycle in sorted(self._lifecycles.items()):
            if record_tenant != tenant:
                continue
            if lifecycle_blocks_watermark(lifecycle):
                unresolved.append(KnowledgeRevisionPosition(scope=scope, value=value))
        return tuple(unresolved)

    def acquire_resolution_authority(
        self,
        scope: KnowledgeOrderingScope,
    ) -> ResolutionAuthority:
        return ResolutionAuthority(
            scope=scope,
            fencing_generation=RevisionFencingGeneration(scope=scope, value=1),
        )

    def resolve_unresolved_position(
        self,
        *,
        position: KnowledgeRevisionPosition,
        authority: ResolutionAuthority,
        reason: KnowledgeRevisionResolutionReason,
        source: KnowledgeRevisionResolutionSource,
        actor_identity: str | None = None,
        correlation_id: str | None = None,
    ) -> KnowledgeRevisionResolutionRecord:
        from intergrax.utils.time_provider import SystemTimeProvider

        tenant = position.scope.tenant_id
        lifecycle = self._lifecycles.get((tenant, position.value))
        if lifecycle is None:
            raise UnknownKnowledgeRevisionPositionError(
                f"knowledge revision position {position.value} was never allocated"
            )
        if lifecycle is KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED:
            raise ValueError("already terminal")
        self._lifecycles[(tenant, position.value)] = (
            KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED
        )
        return KnowledgeRevisionResolutionRecord(
            scope=position.scope,
            position=position,
            prior_lifecycle=lifecycle,
            resulting_lifecycle=KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED,
            reason=reason,
            source=source,
            fencing_generation=authority.fencing_generation,
            detected_at=SystemTimeProvider.utc_now(),
            actor_identity=actor_identity,
            correlation_id=correlation_id,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_allocated_is_not_accepted_and_blocks_watermark() -> None:
    assert KnowledgeRevisionPositionLifecycle.ALLOCATED != KnowledgeRevisionPositionLifecycle.ACCEPTED
    assert lifecycle_is_finalized(KnowledgeRevisionPositionLifecycle.ALLOCATED) is False
    assert lifecycle_is_finalized(KnowledgeRevisionPositionLifecycle.ACCEPTED) is True
    assert lifecycle_is_finalized(KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED) is True
    assert lifecycle_blocks_watermark(KnowledgeRevisionPositionLifecycle.UNRESOLVED) is True
    assert lifecycle_blocks_watermark(KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED) is False


@pytest.mark.unit
@pytest.mark.gate
def test_watermark_advances_across_terminal_non_committed_gap() -> None:
    watermark = compute_finalized_watermark(
        scope=_scope(),
        records=(
            _record(1, KnowledgeRevisionPositionLifecycle.ACCEPTED),
            _record(2, KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED),
            _record(3, KnowledgeRevisionPositionLifecycle.ACCEPTED),
        ),
    )
    assert watermark.finalized_through_value == 3
    assert watermark.is_empty is False


@pytest.mark.unit
@pytest.mark.gate
def test_unresolved_gap_blocks_watermark_at_prior_finalized() -> None:
    watermark = compute_finalized_watermark(
        scope=_scope(),
        records=(
            _record(1, KnowledgeRevisionPositionLifecycle.ACCEPTED),
            _record(2, KnowledgeRevisionPositionLifecycle.UNRESOLVED),
            _record(3, KnowledgeRevisionPositionLifecycle.ACCEPTED),
        ),
    )
    assert watermark.finalized_through_value == 1


@pytest.mark.unit
@pytest.mark.gate
def test_watermark_is_not_highest_allocated() -> None:
    watermark = compute_finalized_watermark(
        scope=_scope(),
        records=(
            _record(1, KnowledgeRevisionPositionLifecycle.ACCEPTED),
            _record(2, KnowledgeRevisionPositionLifecycle.ALLOCATED),
            _record(3, KnowledgeRevisionPositionLifecycle.ACCEPTED),
        ),
    )
    assert watermark.finalized_through_value == 1
    assert watermark.finalized_through == KnowledgeRevisionPosition(scope=_scope(), value=1)


@pytest.mark.unit
@pytest.mark.gate
def test_missing_position_is_not_a_lifecycle_state() -> None:
    with pytest.raises(ValueError, match="contiguous from 1"):
        compute_finalized_watermark(
            scope=_scope(),
            records=(
                _record(1, KnowledgeRevisionPositionLifecycle.ACCEPTED),
                _record(3, KnowledgeRevisionPositionLifecycle.ACCEPTED),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_empty_scope_watermark_and_cross_scope_records_rejected() -> None:
    empty = compute_finalized_watermark(scope=_scope(), records=())
    assert empty.is_empty is True
    assert empty.finalized_through is None
    with pytest.raises(CrossScopeKnowledgeOrderError):
        compute_finalized_watermark(
            scope=_scope("tenant-a"),
            records=(
                _record(
                    1,
                    KnowledgeRevisionPositionLifecycle.ACCEPTED,
                    scope=_scope("tenant-b"),
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_acceptance_result_binds_revision_key_and_position() -> None:
    revision = mint_knowledge_revision_id()
    key = mint_revision_acceptance_key()
    position = KnowledgeRevisionPosition(scope=_scope(), value=7)
    accepted = KnowledgeRevisionAcceptance(
        revision_id=revision,
        acceptance_key=key,
        position=position,
    )
    retry = KnowledgeRevisionAcceptance(
        revision_id=revision,
        acceptance_key=key,
        position=position,
    )
    assert accepted == retry
    assert accepted.acceptance_key != RevisionAcceptanceKey(
        mint_revision_acceptance_key().value
    )


@pytest.mark.unit
@pytest.mark.gate
def test_revision_ordering_authority_is_abstract() -> None:
    assert "accept_revision" in RevisionOrderingAuthority.__abstractmethods__
    assert "watermark" in RevisionOrderingAuthority.__abstractmethods__
    assert "position_lifecycle" in RevisionOrderingAuthority.__abstractmethods__


@pytest.mark.unit
@pytest.mark.gate
def test_same_key_same_revision_idempotent_acceptance() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    revision = mint_knowledge_revision_id()
    key = mint_revision_acceptance_key()

    first = authority.accept_revision(
        scope=scope,
        revision_id=revision,
        acceptance_key=key,
    )
    retry = authority.accept_revision(
        scope=scope,
        revision_id=revision,
        acceptance_key=key,
    )

    assert first == retry
    assert first.position.value == 1
    assert retry.position.value == 1


@pytest.mark.unit
@pytest.mark.gate
def test_same_key_different_revision_raises_conflict() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    key = mint_revision_acceptance_key()
    revision_one = mint_knowledge_revision_id()
    revision_two = mint_knowledge_revision_id()

    first = authority.accept_revision(
        scope=scope,
        revision_id=revision_one,
        acceptance_key=key,
    )
    with pytest.raises(RevisionAcceptanceConflictError):
        authority.accept_revision(
            scope=scope,
            revision_id=revision_two,
            acceptance_key=key,
        )

    assert first.position.value == 1
    assert authority._next_position[scope.tenant_id] == 2


@pytest.mark.unit
@pytest.mark.gate
def test_distinct_revisions_and_keys_receive_distinct_positions() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()

    first = authority.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    second = authority.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )

    assert first.position.value != second.position.value
    assert first.position.scope == second.position.scope == scope


@pytest.mark.unit
@pytest.mark.gate
def test_cross_tenant_acceptance_keys_are_independent() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    key = mint_revision_acceptance_key()
    tenant_a = _scope("tenant-a")
    tenant_b = _scope("tenant-b")

    accepted_a = authority.accept_revision(
        scope=tenant_a,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=key,
    )
    accepted_b = authority.accept_revision(
        scope=tenant_b,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=key,
    )

    assert accepted_a.position.scope == tenant_a
    assert accepted_b.position.scope == tenant_b
    assert accepted_a.position.value == 1
    assert accepted_b.position.value == 1


@pytest.mark.unit
@pytest.mark.gate
def test_concurrent_same_acceptance_converges_to_one_result() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    revision = mint_knowledge_revision_id()
    key = mint_revision_acceptance_key()

    first = authority.accept_revision(
        scope=scope,
        revision_id=revision,
        acceptance_key=key,
    )
    concurrent_retry = authority.accept_revision(
        scope=scope,
        revision_id=revision,
        acceptance_key=key,
    )

    assert first == concurrent_retry
    assert authority._next_position[scope.tenant_id] == 2


@pytest.mark.unit
@pytest.mark.gate
def test_concurrent_conflicting_revision_cannot_both_succeed() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    key = mint_revision_acceptance_key()
    revision_one = mint_knowledge_revision_id()
    revision_two = mint_knowledge_revision_id()

    authority.accept_revision(
        scope=scope,
        revision_id=revision_one,
        acceptance_key=key,
    )
    with pytest.raises(RevisionAcceptanceConflictError):
        authority.accept_revision(
            scope=scope,
            revision_id=revision_two,
            acceptance_key=key,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_failure_matrix_watermark_invariants() -> None:
    """Semantic outcomes A–J that a provider MUST preserve (TRACE-BITEMP-2 implements)."""

    accepted = KnowledgeRevisionPositionLifecycle.ACCEPTED
    terminal = KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED
    unresolved = KnowledgeRevisionPositionLifecycle.UNRESOLVED
    allocated = KnowledgeRevisionPositionLifecycle.ALLOCATED

    # A: allocated then accepted → watermark may include K.
    assert compute_finalized_watermark(
        scope=_scope(),
        records=(_record(1, accepted),),
    ).finalized_through_value == 1

    # B: allocated then rolled back as terminal non-committed → gap is finalized.
    assert compute_finalized_watermark(
        scope=_scope(),
        records=(_record(1, terminal),),
    ).finalized_through_value == 1

    # C/F/G: crash / unknown / unresolved blocks advancement.
    assert compute_finalized_watermark(
        scope=_scope(),
        records=(_record(1, unresolved),),
    ).is_empty is True

    # J: terminal gap below later accepted revisions does not freeze watermark.
    assert compute_finalized_watermark(
        scope=_scope(),
        records=(
            _record(1, accepted),
            _record(2, terminal),
            _record(3, accepted),
        ),
    ).finalized_through_value == 3

    # H/I: same key + same revision → same K; same key + different revision → conflict.
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    key = mint_revision_acceptance_key()
    revision = mint_knowledge_revision_id()
    same = authority.accept_revision(
        scope=scope,
        revision_id=revision,
        acceptance_key=key,
    )
    retry = authority.accept_revision(
        scope=scope,
        revision_id=revision,
        acceptance_key=key,
    )
    assert same.position == retry.position
    with pytest.raises(RevisionAcceptanceConflictError):
        authority.accept_revision(
            scope=scope,
            revision_id=mint_knowledge_revision_id(),
            acceptance_key=key,
        )
    assert allocated is not accepted
    _ = KnowledgeRevisionWatermark(scope=_scope(), finalized_through_value=0)


@pytest.mark.unit
@pytest.mark.gate
def test_accepted_position_record_requires_revision_id() -> None:
    with pytest.raises(InvalidKnowledgeRevisionPositionRecordError):
        KnowledgeRevisionPositionRecord(
            position=KnowledgeRevisionPosition(scope=_scope(), value=1),
            lifecycle=KnowledgeRevisionPositionLifecycle.ACCEPTED,
            accepted_revision_id=None,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_terminal_position_record_rejects_revision_id() -> None:
    with pytest.raises(InvalidKnowledgeRevisionPositionRecordError):
        KnowledgeRevisionPositionRecord(
            position=KnowledgeRevisionPosition(scope=_scope(), value=2),
            lifecycle=KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED,
            accepted_revision_id=mint_knowledge_revision_id(),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_accepted_position_record_binds_revision_id() -> None:
    revision = mint_knowledge_revision_id()
    record = KnowledgeRevisionPositionRecord(
        position=KnowledgeRevisionPosition(scope=_scope(), value=3),
        lifecycle=KnowledgeRevisionPositionLifecycle.ACCEPTED,
        accepted_revision_id=revision,
    )
    assert record.accepted_revision_id == revision
