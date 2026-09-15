# © Artur Czarnecki. All rights reserved.

"""OBS-BITEMP-REBASE — E/K/V/S temporal composition qualification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from intergrax.contracts.bitemporal_knowledge import (
    BitemporalKnowledgeBasis,
    KnowledgeOrderingScope,
    KnowledgeRevisionId,
    KnowledgeRevisionPosition,
    KnowledgeRevisionPositionLifecycle,
    KnowledgeRevisionPositionRecord,
    KnowledgeRevisionWatermark,
    RevisionOrderingAuthority,
    SystemTimeBasis,
    ValidTimeBasis,
    mint_knowledge_revision_id,
    mint_revision_acceptance_key,
)
from intergrax.contracts.execution_identity import RunId, mint_run_id
from intergrax.contracts.historical_reconstruction import (
    ExecutionHistoricalReconstructionRequest,
    revision_admissible_at_bitemporal_query,
)
from intergrax.runtime.events.execution_position import AsOfBoundary, ExecutionEventPosition
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.historical_reconstruction import HistoricalReconstructionService
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from tests.unit.contracts.test_bitemporal_revision_ordering import (
    _InMemoryRevisionOrderingAuthority,
)
from tests.unit.runtime.architecture.test_npsc5f_r4_reconstruction_asof_bitemporal import (
    _append_events,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-bitemp"
_T0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
_T5 = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
_T10 = datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc)
_T11 = datetime(2026, 1, 11, 0, 0, tzinfo=timezone.utc)
_S4 = datetime(2026, 1, 4, 0, 0, tzinfo=timezone.utc)
_S5 = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
_S6 = datetime(2026, 1, 6, 0, 0, tzinfo=timezone.utc)


def _scope() -> KnowledgeOrderingScope:
    return KnowledgeOrderingScope(tenant_id=_TENANT)


@dataclass(frozen=True, slots=True)
class _Revision:
    revision_id: KnowledgeRevisionId
    payload: str
    basis: BitemporalKnowledgeBasis


class _RevisionReader:
    def __init__(self, revisions: dict[KnowledgeRevisionId, _Revision]) -> None:
        self._revisions = revisions

    def load_revision(self, revision_id: KnowledgeRevisionId) -> _Revision:
        return self._revisions[revision_id]


def _reduce(state: tuple[str, ...], revision: _Revision) -> tuple[str, ...]:
    return state + (revision.payload,)


class _CustomRevisionOrderingAuthority(RevisionOrderingAuthority):
    """Minimal alternate provider — proves core does not depend on in-memory store."""

    def __init__(self) -> None:
        self._records: list[KnowledgeRevisionPositionRecord] = []

    def accept_revision(self, *, scope, revision_id, acceptance_key):
        from intergrax.contracts.bitemporal_knowledge import KnowledgeRevisionAcceptance

        value = len(self._records) + 1
        position = KnowledgeRevisionPosition(scope=scope, value=value)
        self._records.append(
            KnowledgeRevisionPositionRecord(
                position=position,
                lifecycle=KnowledgeRevisionPositionLifecycle.ACCEPTED,
                accepted_revision_id=revision_id,
            )
        )
        return KnowledgeRevisionAcceptance(
            revision_id=revision_id,
            acceptance_key=acceptance_key,
            position=position,
        )

    def position_lifecycle(self, position: KnowledgeRevisionPosition):
        for record in self._records:
            if record.position == position:
                return record.lifecycle
        from intergrax.contracts.bitemporal_knowledge import UnknownKnowledgeRevisionPositionError

        raise UnknownKnowledgeRevisionPositionError("unknown")

    def watermark(self, scope: KnowledgeOrderingScope) -> KnowledgeRevisionWatermark:
        return KnowledgeRevisionWatermark(scope=scope, finalized_through_value=len(self._records))

    def records_through(self, watermark: KnowledgeRevisionWatermark):
        return tuple(self._records[: watermark.finalized_through_value])

    def unresolved_positions(self, scope: KnowledgeOrderingScope):
        return ()

    def acquire_resolution_authority(self, scope: KnowledgeOrderingScope):
        from intergrax.contracts.bitemporal_knowledge import (
            ResolutionAuthority,
            RevisionFencingGeneration,
        )

        return ResolutionAuthority(
            scope=scope,
            fencing_generation=RevisionFencingGeneration(scope=scope, value=0),
        )

    def resolve_unresolved_position(self, **kwargs):
        raise NotImplementedError


def _service(
    store: InMemoryRuntimeEventStore,
    authority: RevisionOrderingAuthority,
) -> HistoricalReconstructionService:
    return HistoricalReconstructionService(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
        revision_ordering=authority,
    )


def _request(
    *,
    run_id: RunId,
    e_position: int,
    watermark: KnowledgeRevisionWatermark,
    query: BitemporalKnowledgeBasis,
) -> ExecutionHistoricalReconstructionRequest:
    return ExecutionHistoricalReconstructionRequest(
        tenant_id=_TENANT,
        run_id=run_id,
        execution_as_of=AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(e_position)),
        knowledge_watermark=watermark,
        bitemporal_query=query,
    )


def _open_query(*, valid_at: datetime, system_at: datetime) -> BitemporalKnowledgeBasis:
    return BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(valid_at),
        system_time=SystemTimeBasis.instant(system_at),
    )


def test_obs_bitemp_k_watermark_excludes_later_revision() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    basis = _open_query(valid_at=_T5, system_at=_S6)
    for label in ("k1", "k2"):
        revision_id = mint_knowledge_revision_id()
        revisions[revision_id] = _Revision(revision_id=revision_id, payload=label, basis=basis)
        authority.accept_revision(
            scope=scope,
            revision_id=revision_id,
            acceptance_key=mint_revision_acceptance_key(),
        )
    watermark_k1 = KnowledgeRevisionWatermark(scope=scope, finalized_through_value=1)
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=1, tenant_id=_TENANT)
    service = _service(store, authority)
    request = _request(run_id=run_id, e_position=1, watermark=watermark_k1, query=basis)
    result = service.reconstruct(
        request,
        revision_reader=_RevisionReader(revisions),
        revision_bitemporal_basis=lambda r: r.basis,
        reducer=_reduce,
        initial_state=(),
    )
    assert result.knowledge_view.state == ("k1",)


def test_obs_bitemp_valid_time_interval_filter() -> None:
    revision_basis = BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.interval(_T0, _T10),
        system_time=SystemTimeBasis.instant(_S6),
    )
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    revision_id = mint_knowledge_revision_id()
    revisions[revision_id] = _Revision(
        revision_id=revision_id,
        payload="interval-fact",
        basis=revision_basis,
    )
    authority.accept_revision(
        scope=scope,
        revision_id=revision_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    watermark = authority.watermark(scope)
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=1, tenant_id=_TENANT)
    service = _service(store, authority)

    visible = service.reconstruct(
        _request(
            run_id=run_id,
            e_position=1,
            watermark=watermark,
            query=_open_query(valid_at=_T5, system_at=_S6),
        ),
        revision_reader=_RevisionReader(revisions),
        revision_bitemporal_basis=lambda r: r.basis,
        reducer=_reduce,
        initial_state=(),
    )
    hidden = service.reconstruct(
        _request(
            run_id=run_id,
            e_position=1,
            watermark=watermark,
            query=_open_query(valid_at=_T11, system_at=_S6),
        ),
        revision_reader=_RevisionReader(revisions),
        revision_bitemporal_basis=lambda r: r.basis,
        reducer=_reduce,
        initial_state=(),
    )
    assert visible.knowledge_view.state == ("interval-fact",)
    assert hidden.knowledge_view.state == ()
    assert hidden.limitations.knowledge_bitemporal_filtered is True


def test_obs_bitemp_system_time_visibility() -> None:
    revision_basis = BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T0),
        system_time=SystemTimeBasis.instant(_S5),
    )
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    revision_id = mint_knowledge_revision_id()
    revisions[revision_id] = _Revision(
        revision_id=revision_id,
        payload="stored-jan-5",
        basis=revision_basis,
    )
    authority.accept_revision(
        scope=scope,
        revision_id=revision_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    watermark = authority.watermark(scope)
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=1, tenant_id=_TENANT)
    service = _service(store, authority)

    before = service.reconstruct(
        _request(
            run_id=run_id,
            e_position=1,
            watermark=watermark,
            query=_open_query(valid_at=_T0, system_at=_S4),
        ),
        revision_reader=_RevisionReader(revisions),
        revision_bitemporal_basis=lambda r: r.basis,
        reducer=_reduce,
        initial_state=(),
    )
    after = service.reconstruct(
        _request(
            run_id=run_id,
            e_position=1,
            watermark=watermark,
            query=_open_query(valid_at=_T0, system_at=_S6),
        ),
        revision_reader=_RevisionReader(revisions),
        revision_bitemporal_basis=lambda r: r.basis,
        reducer=_reduce,
        initial_state=(),
    )
    assert before.knowledge_view.state == ()
    assert after.knowledge_view.state == ("stored-jan-5",)


def test_obs_bitemp_e_k_independence() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    basis = _open_query(valid_at=_T5, system_at=_S6)
    for label in ("a", "b"):
        revision_id = mint_knowledge_revision_id()
        revisions[revision_id] = _Revision(revision_id=revision_id, payload=label, basis=basis)
        authority.accept_revision(
            scope=scope,
            revision_id=revision_id,
            acceptance_key=mint_revision_acceptance_key(),
        )
    watermark = authority.watermark(scope)
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=10, tenant_id=_TENANT)
    service = _service(store, authority)
    kwargs = {
        "revision_reader": _RevisionReader(revisions),
        "revision_bitemporal_basis": lambda r: r.basis,
        "reducer": _reduce,
        "initial_state": (),
    }

    high_e_low_k = service.reconstruct(
        _request(run_id=run_id, e_position=10, watermark=KnowledgeRevisionWatermark(scope=scope, finalized_through_value=1), query=basis),
        **kwargs,
    )
    low_e_high_k = service.reconstruct(
        _request(run_id=run_id, e_position=1, watermark=watermark, query=basis),
        **kwargs,
    )
    assert high_e_low_k.execution_projection.last_included_position == ExecutionEventPosition(10)
    assert high_e_low_k.knowledge_view.state == ("a",)
    assert low_e_high_k.execution_projection.last_included_position == ExecutionEventPosition(1)
    assert low_e_high_k.knowledge_view.state == ("a", "b")


def test_obs_bitemp_full_coordinate_append_immunity() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    basis = _open_query(valid_at=_T5, system_at=_S6)
    revision_id = mint_knowledge_revision_id()
    revisions[revision_id] = _Revision(revision_id=revision_id, payload="only", basis=basis)
    authority.accept_revision(
        scope=scope,
        revision_id=revision_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    watermark = KnowledgeRevisionWatermark(scope=scope, finalized_through_value=1)
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=2, tenant_id=_TENANT)
    service = _service(store, authority)
    request = _request(run_id=run_id, e_position=2, watermark=watermark, query=basis)
    kwargs = {
        "revision_reader": _RevisionReader(revisions),
        "revision_bitemporal_basis": lambda r: r.basis,
        "reducer": _reduce,
        "initial_state": (),
    }
    first = service.reconstruct(request, **kwargs)
    _append_events(store, run_id=run_id, count=5, tenant_id=_TENANT)
    authority.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    second = service.reconstruct(request, **kwargs)
    assert first == second
    assert first.basis == request.to_basis()


def test_obs_bitemp_retroactive_knowledge_at_k1() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    k1_id = mint_knowledge_revision_id()
    revisions[k1_id] = _Revision(
        revision_id=k1_id,
        payload="A",
        basis=BitemporalKnowledgeBasis(
            valid_time=ValidTimeBasis.instant(_T0),
            system_time=SystemTimeBasis.instant(_S5),
        ),
    )
    authority.accept_revision(
        scope=scope,
        revision_id=k1_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    k2_id = mint_knowledge_revision_id()
    revisions[k2_id] = _Revision(
        revision_id=k2_id,
        payload="B",
        basis=BitemporalKnowledgeBasis(
            valid_time=ValidTimeBasis.instant(_T0),
            system_time=SystemTimeBasis.instant(_S6),
        ),
    )
    authority.accept_revision(
        scope=scope,
        revision_id=k2_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    watermark_k1 = KnowledgeRevisionWatermark(scope=scope, finalized_through_value=1)
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=1, tenant_id=_TENANT)
    service = _service(store, authority)
    query = _open_query(valid_at=_T0, system_at=_S6)
    at_k1 = service.reconstruct(
        _request(run_id=run_id, e_position=1, watermark=watermark_k1, query=query),
        revision_reader=_RevisionReader(revisions),
        revision_bitemporal_basis=lambda r: r.basis,
        reducer=_reduce,
        initial_state=(),
    )
    assert at_k1.knowledge_view.state == ("A",)


def test_obs_bitemp_v_s_independence_contract() -> None:
    revision = BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T0),
        system_time=SystemTimeBasis.instant(_S6),
    )
    valid_miss = BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T11),
        system_time=SystemTimeBasis.instant(_S6),
    )
    system_miss = BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T0),
        system_time=SystemTimeBasis.instant(_S4),
    )
    assert not revision_admissible_at_bitemporal_query(query=valid_miss, revision=revision)
    assert not revision_admissible_at_bitemporal_query(query=system_miss, revision=revision)
    assert revision_admissible_at_bitemporal_query(
        query=_open_query(valid_at=_T0, system_at=_S6),
        revision=revision,
    )


def test_obs_bitemp_custom_revision_ordering_authority() -> None:
    authority = _CustomRevisionOrderingAuthority()
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    basis = _open_query(valid_at=_T5, system_at=_S6)
    revision_id = mint_knowledge_revision_id()
    revisions[revision_id] = _Revision(revision_id=revision_id, payload="custom", basis=basis)
    authority.accept_revision(
        scope=scope,
        revision_id=revision_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    watermark = authority.watermark(scope)
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=1, tenant_id=_TENANT)
    result = _service(store, authority).reconstruct(
        _request(run_id=run_id, e_position=1, watermark=watermark, query=basis),
        revision_reader=_RevisionReader(revisions),
        revision_bitemporal_basis=lambda r: r.basis,
        reducer=_reduce,
        initial_state=(),
    )
    assert result.knowledge_view.state == ("custom",)


def test_obs_bitemp_k_append_immunity_same_watermark() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    basis = _open_query(valid_at=_T5, system_at=_S6)
    revision_id = mint_knowledge_revision_id()
    revisions[revision_id] = _Revision(revision_id=revision_id, payload="k-only", basis=basis)
    authority.accept_revision(
        scope=scope,
        revision_id=revision_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    watermark = KnowledgeRevisionWatermark(scope=scope, finalized_through_value=1)
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=1, tenant_id=_TENANT)
    service = _service(store, authority)
    request = _request(run_id=run_id, e_position=1, watermark=watermark, query=basis)
    kwargs = {
        "revision_reader": _RevisionReader(revisions),
        "revision_bitemporal_basis": lambda r: r.basis,
        "reducer": _reduce,
        "initial_state": (),
    }
    first = service.reconstruct(request, **kwargs)
    authority.accept_revision(
        scope=scope,
        revision_id=mint_knowledge_revision_id(),
        acceptance_key=mint_revision_acceptance_key(),
    )
    second = service.reconstruct(request, **kwargs)
    assert first.knowledge_view == second.knowledge_view
