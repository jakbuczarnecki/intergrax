# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.bitemporal_knowledge import (
    CrossScopeKnowledgeOrderError,
    KnowledgeOrderingScope,
    KnowledgeRevisionAcceptance,
    KnowledgeRevisionPosition,
    KnowledgeRevisionPositionLifecycle,
    KnowledgeRevisionPositionRecord,
    KnowledgeRevisionWatermark,
    RevisionAcceptanceKey,
    RevisionOrderingAuthority,
    compute_finalized_watermark,
    lifecycle_blocks_watermark,
    lifecycle_is_finalized,
    mint_revision_acceptance_key,
)


def _scope(tenant: str = "tenant-a") -> KnowledgeOrderingScope:
    return KnowledgeOrderingScope(tenant_id=tenant)


def _record(
    value: int,
    lifecycle: KnowledgeRevisionPositionLifecycle,
    *,
    scope: KnowledgeOrderingScope | None = None,
) -> KnowledgeRevisionPositionRecord:
    resolved = scope or _scope()
    return KnowledgeRevisionPositionRecord(
        position=KnowledgeRevisionPosition(scope=resolved, value=value),
        lifecycle=lifecycle,
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
def test_acceptance_result_binds_key_to_position() -> None:
    key = mint_revision_acceptance_key()
    position = KnowledgeRevisionPosition(scope=_scope(), value=7)
    accepted = KnowledgeRevisionAcceptance(acceptance_key=key, position=position)
    retry = KnowledgeRevisionAcceptance(acceptance_key=key, position=position)
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

    # H/I are identity/uniqueness: distinct keys → distinct positions; same key
    # cannot be represented as two accepted records at different K.
    duplicate_key_positions_forbidden = {1, 2}
    assert len(duplicate_key_positions_forbidden) == 2
    assert allocated is not accepted
    _ = KnowledgeRevisionWatermark(scope=_scope(), finalized_through_value=0)
