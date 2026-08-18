# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-independent historical knowledge reconstruction at watermark K (TRACE-BITEMP-3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from intergrax.contracts.bitemporal_knowledge import (
    CrossScopeKnowledgeOrderError,
    KnowledgeRevisionId,
    KnowledgeRevisionPosition,
    KnowledgeRevisionPositionLifecycle,
    KnowledgeRevisionPositionRecord,
    KnowledgeRevisionWatermark,
    RevisionOrderingAuthority,
    lifecycle_blocks_watermark,
)

RevisionT_co = TypeVar("RevisionT_co", covariant=True)
StateT = TypeVar("StateT")


class KnowledgeReconstructionError(Exception):
    """Base error for historical knowledge reconstruction."""


class IncompleteKnowledgeRevisionPrefixError(KnowledgeReconstructionError):
    """Finalized watermark prefix is missing or non-contiguous."""


class BlockingKnowledgeRevisionPositionError(KnowledgeReconstructionError):
    """Prefix contains a lifecycle that blocks finalized watermark semantics."""


class KnowledgeRevisionReader(Protocol[RevisionT_co]):
    """Typed revision source — reconstruction does not depend on storage."""

    def load_revision(self, revision_id: KnowledgeRevisionId) -> RevisionT_co: ...


@dataclass(frozen=True, slots=True)
class HistoricalKnowledgeRevisionReference:
    """Provenance for one accepted revision included in reconstruction."""

    position: KnowledgeRevisionPosition
    revision_id: KnowledgeRevisionId

    def __post_init__(self) -> None:
        if type(self.position) is not KnowledgeRevisionPosition:
            raise TypeError("position must be KnowledgeRevisionPosition")
        if type(self.revision_id) is not KnowledgeRevisionId:
            raise TypeError("revision_id must be KnowledgeRevisionId")


@dataclass(frozen=True, slots=True)
class HistoricalKnowledgeProjection(Generic[StateT]):
    """Immutable historical knowledge state reconstructed at watermark K."""

    watermark: KnowledgeRevisionWatermark
    state: StateT
    accepted_revisions: tuple[HistoricalKnowledgeRevisionReference, ...]

    def __post_init__(self) -> None:
        if type(self.watermark) is not KnowledgeRevisionWatermark:
            raise TypeError("watermark must be KnowledgeRevisionWatermark")
        if type(self.accepted_revisions) is not tuple:
            raise TypeError("accepted_revisions must be a tuple")


def reconstruct_knowledge_at_watermark(
    authority: RevisionOrderingAuthority,
    watermark: KnowledgeRevisionWatermark,
    *,
    revision_reader: KnowledgeRevisionReader[RevisionT_co],
    reducer: Callable[[StateT, RevisionT_co], StateT],
    initial_state: StateT,
) -> HistoricalKnowledgeProjection[StateT]:
    """Reconstruct canonical knowledge state at finalized watermark K.

  Ordering authority supplies finalized prefix semantics only; revision content
  is loaded through ``revision_reader`` and folded by a pure deterministic
  ``reducer``.
    """

    records = authority.records_through(watermark)
    _validate_finalized_prefix(watermark=watermark, records=records)
    accepted_references = _accepted_revision_references(records)

    state: StateT = initial_state
    for reference in accepted_references:
        revision = revision_reader.load_revision(reference.revision_id)
        state = reducer(state, revision)

    return HistoricalKnowledgeProjection(
        watermark=watermark,
        state=state,
        accepted_revisions=accepted_references,
    )


def _validate_finalized_prefix(
    *,
    watermark: KnowledgeRevisionWatermark,
    records: tuple[KnowledgeRevisionPositionRecord, ...],
) -> None:
    expected_count = watermark.finalized_through_value
    if len(records) != expected_count:
        raise IncompleteKnowledgeRevisionPrefixError(
            "finalized prefix must contain exactly one record per position through watermark"
        )
    for index, record in enumerate(records, start=1):
        if record.position.scope != watermark.scope:
            raise CrossScopeKnowledgeOrderError(
                "records_through returned a position outside the watermark scope"
            )
        if record.position.value != index:
            raise IncompleteKnowledgeRevisionPrefixError(
                f"finalized prefix missing or out-of-order position {index}"
            )
        if lifecycle_blocks_watermark(record.lifecycle):
            raise BlockingKnowledgeRevisionPositionError(
                f"position {index} lifecycle {record.lifecycle.value} blocks finalized prefix"
            )


def _accepted_revision_references(
    records: tuple[KnowledgeRevisionPositionRecord, ...],
) -> tuple[HistoricalKnowledgeRevisionReference, ...]:
    references: list[HistoricalKnowledgeRevisionReference] = []
    for record in records:
        if record.lifecycle is not KnowledgeRevisionPositionLifecycle.ACCEPTED:
            continue
        revision_id = record.accepted_revision_id
        if revision_id is None:
            raise KnowledgeReconstructionError(
                "ACCEPTED position record missing accepted_revision_id"
            )
        references.append(
            HistoricalKnowledgeRevisionReference(
                position=record.position,
                revision_id=revision_id,
            )
        )
    return tuple(references)
