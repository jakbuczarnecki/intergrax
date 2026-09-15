# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-6: recall ranking, conflict strategies, and control plane integration."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordLineage,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.contracts.memory_control import (
    MemoryControlBackendError,
    MemoryControlPartialLifecycleError,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryLifecycleDisposition,
    UserProfileMemoryReconciliationContext,
)
from intergrax.memory.contracts.memory_models import MemoryKind, UserProfileMemoryEntry
from intergrax.memory.default_memory_control_plane import (
    DefaultMemoryControlPlane,
    UserProfileManagerMemoryCapability,
)
from intergrax.memory.recall.pipeline import run_recall_decision_pipeline
from intergrax.memory.recall_strategy_bundle import MemoryRecallStrategySet, build_default_memory_recall_strategies
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.strategies.defaults.conservative_conflict import (
    ConservativeMemoryConflictDetectionStrategy,
    FailSafeMemoryConflictResolutionStrategy,
)
from intergrax.memory.strategies.defaults.enterprise_ranking import EnterpriseMemoryRankingStrategy
from intergrax.memory.strategies.errors import MemoryStrategyContractError
from intergrax.memory.strategies.recall_models import (
    MemoryConflictKind,
    MemoryConflictResolutionAction,
    MemoryRankingRequest,
    MemoryRankingResult,
    MemoryRankingScore,
    MemoryRankedCandidate,
    MemoryRecallCandidate,
    MemoryRetrievalSource,
    MemorySupersessionIntent,
)
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.memory.strategies.recall_validation import validate_ranking_result
from intergrax.memory.user_profile_manager import UserProfileManager

pytestmark = pytest.mark.gate

_TENANT = "tenant-mem6"
_USER = "user-mem6"
_FIXED_AS_OF = "2026-06-15T12:00:00+00:00"
_FIXED_CREATED = "2026-01-01T00:00:00+00:00"


def _entry(
    entry_id: str,
    content: str,
    *,
    title: str | None = None,
    revision: int = 1,
    trust_class: MemoryTrustClass = MemoryTrustClass.UNKNOWN,
    confidence: float | None = None,
    superseded_by: str | None = None,
    valid_until: str | None = None,
    valid_from: str | None = None,
    created_at: str = _FIXED_CREATED,
    updated_at: str | None = None,
) -> UserProfileMemoryEntry:
    return UserProfileMemoryEntry(
        entry_id=entry_id,
        revision=revision,
        content=content,
        kind=MemoryKind.USER_FACT,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        trust=MemoryRecordTrust(trust_class=trust_class, confidence=confidence),
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
        lineage=MemoryRecordLineage(superseded_by_memory_id=superseded_by),
        valid_until=valid_until,
        valid_from=valid_from,
    )


def _rank_request(
    entries: tuple[MemoryRecallCandidate, ...],
    *,
    top_k: int = 10,
    as_of_iso: str | None = _FIXED_AS_OF,
) -> MemoryRankingRequest:
    return MemoryRankingRequest(
        candidates=entries,
        query="q",
        top_k=top_k,
        as_of_iso=as_of_iso,
    )


def _candidate(entry: UserProfileMemoryEntry, score: float | None = 0.8) -> MemoryRecallCandidate:
    return MemoryRecallCandidate(
        record=entry,
        retrieval_source=MemoryRetrievalSource.SEMANTIC,
        retrieval_score=score,
    )


def test_deterministic_ranking_is_stable() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    entries = (
        _candidate(_entry("a", "one"), 0.9),
        _candidate(_entry("b", "two"), 0.7),
        _candidate(_entry("c", "three"), 0.7),
    )
    request = _rank_request(entries, top_k=3)
    scores = [ranker.rank(request).ranked for _ in range(5)]
    first_ids = [r.candidate.record.entry_id for r in scores[0]]
    first_totals = [r.score.total for r in scores[0]]
    for run in scores[1:]:
        assert [r.candidate.record.entry_id for r in run] == first_ids
        assert [r.score.total for r in run] == first_totals


def test_ranking_tie_break_uses_memory_id() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    shared_score = 0.75
    shared_ts = "2026-03-01T10:00:00+00:00"
    entries = (
        _candidate(
            _entry("zzzz", "x", title="t", created_at=shared_ts, updated_at=shared_ts),
            shared_score,
        ),
        _candidate(
            _entry("aaaa", "y", title="t", created_at=shared_ts, updated_at=shared_ts),
            shared_score,
        ),
    )
    result = ranker.rank(_rank_request(entries, top_k=2))
    assert [r.candidate.record.entry_id for r in result.ranked] == ["aaaa", "zzzz"]


def test_ranking_tie_break_prefers_newer_timestamp() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    shared_score = 0.75
    older = _candidate(
        _entry("older", "x", created_at="2026-01-01T00:00:00+00:00"),
        shared_score,
    )
    newer = _candidate(
        _entry("newer", "y", created_at="2026-06-01T00:00:00+00:00"),
        shared_score,
    )
    result = ranker.rank(_rank_request((older, newer), top_k=2))
    assert [r.candidate.record.entry_id for r in result.ranked] == ["newer", "older"]


_NAIVE_TIE_CREATED = "2020-06-15T12:00:00"


def _naive_tie_candidates(
    older_updated: str,
    newer_updated: str,
    *,
    older_id: str = "older",
    newer_id: str = "newer",
    shared_score: float = 0.75,
) -> tuple[MemoryRecallCandidate, MemoryRecallCandidate]:
    older = _candidate(
        _entry(
            older_id,
            "x",
            created_at=_NAIVE_TIE_CREATED,
            updated_at=older_updated,
        ),
        shared_score,
    )
    newer = _candidate(
        _entry(
            newer_id,
            "y",
            created_at=_NAIVE_TIE_CREATED,
            updated_at=newer_updated,
        ),
        shared_score,
    )
    return older, newer


def test_ranking_naive_tie_break_year_boundary() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    older, newer = _naive_tie_candidates(
        "2025-12-31T23:59:59",
        "2026-01-01T00:00:00",
    )
    result = ranker.rank(_rank_request((older, newer), top_k=2))
    assert [r.candidate.record.entry_id for r in result.ranked] == ["newer", "older"]


def test_ranking_naive_tie_break_month_boundary() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    older, newer = _naive_tie_candidates(
        "2026-01-31T23:59:59",
        "2026-02-01T00:00:00",
    )
    result = ranker.rank(_rank_request((older, newer), top_k=2))
    assert [r.candidate.record.entry_id for r in result.ranked] == ["newer", "older"]


def test_ranking_naive_tie_break_leap_day_boundary() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    older, newer = _naive_tie_candidates(
        "2024-02-29T23:59:59",
        "2024-03-01T00:00:00",
    )
    result = ranker.rank(_rank_request((older, newer), top_k=2))
    assert [r.candidate.record.entry_id for r in result.ranked] == ["newer", "older"]


def test_ranking_naive_tie_break_microseconds() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    older, newer = _naive_tie_candidates(
        "2026-06-01T10:00:00.000000",
        "2026-06-01T10:00:00.000001",
    )
    result = ranker.rank(_rank_request((older, newer), top_k=2))
    assert [r.candidate.record.entry_id for r in result.ranked] == ["newer", "older"]


def test_ranking_aware_tie_break_uses_instant_not_lexical() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    shared_score = 0.75
    shared_created = "2020-06-15T12:00:00+00:00"
    earlier_instant = _candidate(
        _entry(
            "earlier",
            "x",
            created_at=shared_created,
            updated_at="2026-06-01T10:00:00+02:00",
        ),
        shared_score,
    )
    later_instant = _candidate(
        _entry(
            "later",
            "y",
            created_at=shared_created,
            updated_at="2026-06-01T09:30:00+00:00",
        ),
        shared_score,
    )
    result = ranker.rank(_rank_request((earlier_instant, later_instant), top_k=2))
    assert [r.candidate.record.entry_id for r in result.ranked] == ["later", "earlier"]


def test_freshness_depends_on_explicit_as_of() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    entry = _candidate(
        _entry("e1", "fact", created_at="2020-01-01T00:00:00+00:00"),
        0.5,
    )
    recent_reference = ranker.rank(
        _rank_request((entry,), as_of_iso="2020-01-02T00:00:00+00:00")
    ).ranked[0].score.freshness
    stale_reference = ranker.rank(
        _rank_request((entry,), as_of_iso="2021-06-01T00:00:00+00:00")
    ).ranked[0].score.freshness
    assert recent_reference > stale_reference


def test_enterprise_ranking_has_no_hidden_wall_clock() -> None:
    import inspect

    from intergrax.memory.strategies.defaults import enterprise_ranking

    source = inspect.getsource(enterprise_ranking)
    assert "datetime.now" not in source


def test_mixed_naive_and_aware_as_of_yields_neutral_freshness() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    aware = _candidate(
        _entry("a", "x", created_at="2026-01-01T00:00:00+00:00"),
        0.8,
    )
    result = ranker.rank(
        _rank_request((aware,), as_of_iso="2026-06-01T00:00:00")
    )
    assert result.ranked[0].score.freshness == 0.5


def test_trust_contributes_without_absolute_override() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    low_trust = _candidate(
        _entry("low", "fact", trust_class=MemoryTrustClass.UNKNOWN),
        0.95,
    )
    high_trust = _candidate(
        _entry("high", "fact", trust_class=MemoryTrustClass.USER_EXPLICIT, confidence=0.95),
        0.55,
    )
    result = ranker.rank(_rank_request((low_trust, high_trust), top_k=2))
    assert result.ranked[0].candidate.record.entry_id == "low"
    assert result.ranked[1].score.total < result.ranked[0].score.total


def test_superseded_record_excluded_from_default_ranking() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    active = _candidate(_entry("active", "current"))
    superseded = _candidate(_entry("old", "stale", superseded_by="active"))
    result = ranker.rank(_rank_request((active, superseded), top_k=5))
    assert [r.candidate.record.entry_id for r in result.ranked] == ["active"]


def test_non_finite_score_raises_contract_error() -> None:
    class BadRanker:
        strategy_id = "bad"

        def rank(self, request: MemoryRankingRequest) -> MemoryRankingResult:
            ranked = MemoryRankedCandidate(
                candidate=request.candidates[0],
                score=MemoryRankingScore(total=float("nan")),
            )
            result = MemoryRankingResult(ranked=(ranked,))
            validate_ranking_result(request, result)
            return result

    ranker = BadRanker()
    request = MemoryRankingRequest(candidates=(_candidate(_entry("a", "x")),), query="", top_k=1)
    with pytest.raises(MemoryStrategyContractError):
        ranker.rank(request)


def test_foreign_ranked_id_raises_contract_error() -> None:
    class ForeignRanker:
        strategy_id = "foreign"

        def rank(self, request: MemoryRankingRequest) -> MemoryRankingResult:
            foreign = _candidate(_entry("foreign", "x"))
            ranked = MemoryRankedCandidate(
                candidate=foreign,
                score=MemoryRankingScore(total=0.5),
            )
            result = MemoryRankingResult(ranked=(ranked,))
            validate_ranking_result(request, result)
            return result

    request = MemoryRankingRequest(candidates=(_candidate(_entry("a", "x")),), query="", top_k=1)
    with pytest.raises(MemoryStrategyContractError):
        ForeignRanker().rank(request)


def test_duplicate_ranked_id_raises_contract_error() -> None:
    class DupRanker:
        strategy_id = "dup"

        def rank(self, request: MemoryRankingRequest) -> MemoryRankingResult:
            item = MemoryRankedCandidate(
                candidate=request.candidates[0],
                score=MemoryRankingScore(total=0.5),
            )
            result = MemoryRankingResult(ranked=(item, item))
            validate_ranking_result(request, result)
            return result

    request = MemoryRankingRequest(candidates=(_candidate(_entry("a", "x")),), query="", top_k=2)
    with pytest.raises(MemoryStrategyContractError):
        DupRanker().rank(request)


def test_conflict_detection_for_same_subject_different_content() -> None:
    detector = ConservativeMemoryConflictDetectionStrategy()
    ranked = (
        MemoryRankedCandidate(
            candidate=_candidate(_entry("e1", "likes tea", title="drink")),
            score=MemoryRankingScore(total=0.8),
        ),
        MemoryRankedCandidate(
            candidate=_candidate(_entry("e2", "likes coffee", title="drink")),
            score=MemoryRankingScore(total=0.7),
        ),
    )
    from intergrax.memory.strategies.recall_models import MemoryConflictDetectionRequest

    result = detector.detect(MemoryConflictDetectionRequest(ranked=ranked))
    assert len(result.conflicts) == 1
    assert result.conflicts[0].kind is MemoryConflictKind.CONTRADICTION


def test_unrelated_records_do_not_conflict() -> None:
    detector = ConservativeMemoryConflictDetectionStrategy()
    ranked = (
        MemoryRankedCandidate(
            candidate=_candidate(_entry("e1", "alpha", title="a")),
            score=MemoryRankingScore(total=0.8),
        ),
        MemoryRankedCandidate(
            candidate=_candidate(_entry("e2", "beta", title="b")),
            score=MemoryRankingScore(total=0.7),
        ),
    )
    from intergrax.memory.strategies.recall_models import MemoryConflictDetectionRequest

    result = detector.detect(MemoryConflictDetectionRequest(ranked=ranked))
    assert result.conflicts == ()


def test_resolver_keep_both_when_ambiguous() -> None:
    from intergrax.memory.strategies.recall_models import (
        MemoryConflict,
        MemoryConflictKind,
        MemoryConflictResolutionRequest,
    )

    resolver = FailSafeMemoryConflictResolutionStrategy()
    ranked = (
        MemoryRankedCandidate(
            candidate=_candidate(_entry("e1", "v1", title="k", revision=1)),
            score=MemoryRankingScore(total=0.6),
        ),
        MemoryRankedCandidate(
            candidate=_candidate(_entry("e2", "v2", title="k", revision=1)),
            score=MemoryRankingScore(total=0.6),
        ),
    )
    conflicts = (
        MemoryConflict(
            conflict_id="e1:e2",
            records=(ranked[0].candidate.record, ranked[1].candidate.record),
            kind=MemoryConflictKind.CONTRADICTION,
            reason="test",
        ),
    )
    result = resolver.resolve(
        MemoryConflictResolutionRequest(conflicts=conflicts, ranked=ranked)
    )
    assert result.decisions[0].action is MemoryConflictResolutionAction.KEEP_BOTH


def test_resolver_ignores_cross_record_revision_for_recency() -> None:
    from intergrax.memory.strategies.recall_models import (
        MemoryConflict,
        MemoryConflictResolutionRequest,
    )

    resolver = FailSafeMemoryConflictResolutionStrategy()
    high_revision = _entry(
        "high-rev",
        "old fact",
        title="fact",
        revision=10,
        valid_from="2020-01-01T00:00:00+00:00",
    )
    low_revision = _entry(
        "low-rev",
        "new fact",
        title="fact",
        revision=1,
        valid_from="2026-01-01T00:00:00+00:00",
    )
    ranked = (
        MemoryRankedCandidate(
            candidate=_candidate(high_revision, 0.9),
            score=MemoryRankingScore(total=0.9),
        ),
        MemoryRankedCandidate(
            candidate=_candidate(low_revision, 0.5),
            score=MemoryRankingScore(total=0.5),
        ),
    )
    conflicts = (
        MemoryConflict(
            conflict_id="high-rev:low-rev",
            records=(high_revision, low_revision),
            kind=MemoryConflictKind.POTENTIAL_SUPERSESSION,
            reason="test",
        ),
    )
    result = resolver.resolve(
        MemoryConflictResolutionRequest(conflicts=conflicts, ranked=ranked)
    )
    intent = result.decisions[0].supersession_intent
    assert intent is not None
    assert intent.superseding_memory_id == "low-rev"
    assert intent.superseded_memory_id == "high-rev"


def test_resolver_supersede_without_mutation() -> None:
    from intergrax.memory.strategies.recall_models import (
        MemoryConflict,
        MemoryConflictKind,
        MemoryConflictResolutionRequest,
    )

    resolver = FailSafeMemoryConflictResolutionStrategy()
    left = _entry(
        "older",
        "old value",
        title="fact",
        revision=1,
        valid_from="2024-01-01T00:00:00+00:00",
    )
    right = _entry(
        "newer",
        "new value",
        title="fact",
        revision=1,
        valid_from="2026-01-01T00:00:00+00:00",
    )
    before = (copy.deepcopy(left), copy.deepcopy(right))
    ranked = (
        MemoryRankedCandidate(
            candidate=_candidate(left, 0.5),
            score=MemoryRankingScore(total=0.5),
        ),
        MemoryRankedCandidate(
            candidate=_candidate(right, 0.9),
            score=MemoryRankingScore(total=0.9),
        ),
    )
    conflicts = (
        MemoryConflict(
            conflict_id="older:newer",
            records=(left, right),
            kind=MemoryConflictKind.POTENTIAL_SUPERSESSION,
            reason="test",
        ),
    )
    result = resolver.resolve(
        MemoryConflictResolutionRequest(conflicts=conflicts, ranked=ranked)
    )
    assert result.decisions[0].action is MemoryConflictResolutionAction.SUPERSEDE_EXISTING
    assert result.decisions[0].supersession_intent is not None
    assert (left.lineage, right.lineage) == (before[0].lineage, before[1].lineage)


@dataclass
class ReverseRankingStrategy:
    strategy_id: str = "reverse_test"

    def rank(self, request: MemoryRankingRequest) -> MemoryRankingResult:
        ranked = [
            MemoryRankedCandidate(
                candidate=candidate,
                score=MemoryRankingScore(total=float(index)),
            )
            for index, candidate in enumerate(reversed(request.candidates))
        ]
        result = MemoryRankingResult(ranked=tuple(ranked))
        validate_ranking_result(request, result)
        return result


@pytest.mark.asyncio
async def test_control_plane_respects_custom_ranker() -> None:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store, tenant_id=_TENANT)
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
        recall_strategies=MemoryRecallStrategySet(
            ranking=ReverseRankingStrategy(),
            conflict_detection=ConservativeMemoryConflictDetectionStrategy(),
            conflict_resolution=FailSafeMemoryConflictResolutionStrategy(),
        ),
    )
    identity = RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )
    scope = user_memory_scope(identity)
    first = await plane.remember(
        identity, scope, MemoryControlRememberRequest(content="first entry")
    )
    second = await plane.remember(
        identity, scope, MemoryControlRememberRequest(content="second entry")
    )
    recall = await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=10))
    ids = [item.entry_id for item in recall.items]
    assert ids.index(second.entry_id or "") < ids.index(first.entry_id or "")


@pytest.mark.asyncio
async def test_recall_does_not_mutate_lineage_or_revision() -> None:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store, tenant_id=_TENANT)
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
        recall_strategies=build_default_memory_recall_strategies(),
    )
    identity = RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )
    scope = user_memory_scope(identity)
    await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="prefers tea", title="drink"),
    )
    await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="prefers coffee", title="drink"),
    )
    profile_before = copy.deepcopy(await manager.get_profile(_USER))
    await plane.recall(identity, scope, MemoryControlRecallRequest(query="prefers", top_k=10))
    profile_after = await manager.get_profile(_USER)
    assert profile_before.memory_entries == profile_after.memory_entries


@pytest.mark.asyncio
async def test_apply_supersession_updates_lineage_and_revision() -> None:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store, tenant_id=_TENANT)
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
    )
    identity = RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )
    scope = user_memory_scope(identity)
    older = await plane.remember(
        identity, scope, MemoryControlRememberRequest(content="old", title="fact")
    )
    newer = await plane.remember(
        identity, scope, MemoryControlRememberRequest(content="new", title="fact")
    )
    assert older.entry_id and newer.entry_id
    await plane.apply_memory_supersession(
        identity,
        scope,
        MemorySupersessionIntent(
            superseded_memory_id=older.entry_id,
            superseding_memory_id=newer.entry_id,
            reason="test",
        ),
    )
    profile = await manager.get_profile(_USER)
    by_id = {entry.entry_id: entry for entry in profile.memory_entries}
    assert by_id[older.entry_id].lineage.superseded_by_memory_id == newer.entry_id
    assert by_id[newer.entry_id].lineage.supersedes_memory_id == older.entry_id
    assert by_id[older.entry_id].revision >= 2
    assert by_id[newer.entry_id].revision >= 2


@pytest.mark.asyncio
async def test_custom_ranker_failure_surfaces_as_backend_error() -> None:
    class FailingRanker:
        strategy_id = "fail"

        def rank(self, request: MemoryRankingRequest) -> MemoryRankingResult:
            raise MemoryStrategyContractError("boom")

    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store, tenant_id=_TENANT)
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
        recall_strategies=MemoryRecallStrategySet(
            ranking=FailingRanker(),
            conflict_detection=ConservativeMemoryConflictDetectionStrategy(),
            conflict_resolution=FailSafeMemoryConflictResolutionStrategy(),
        ),
    )
    identity = RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )
    scope = user_memory_scope(identity)
    await plane.remember(identity, scope, MemoryControlRememberRequest(content="x"))
    with pytest.raises(MemoryControlBackendError):
        await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=5))


@dataclass
class SelectiveFailProjection:
    fail_entry_ids: frozenset[str] = frozenset()
    projection_id: str = "selective_fail"
    always_fail: bool = False
    upsert_calls: list[str] = field(default_factory=list)

    async def upsert_memory_entry(
        self,
        user_id: str,
        entry: UserProfileMemoryEntry,
    ) -> None:
        self.upsert_calls.append(entry.entry_id)
        if self.always_fail or entry.entry_id in self.fail_entry_ids:
            raise TimeoutError("projection failed")

    async def delete_memory_entries(self, entry_ids: Sequence[str]) -> None:
        return None

    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ):
        from intergrax.memory.contracts.memory_lifecycle import (
            MemoryProjectionReconciliationDisposition,
            MemoryProjectionReconciliationResult,
        )

        return MemoryProjectionReconciliationResult(
            projection_id=self.projection_id,
            disposition=MemoryProjectionReconciliationDisposition.CONSISTENT,
        )


@pytest.mark.asyncio
async def test_supersession_projection_partial_attempts_both_entries() -> None:
    store = InMemoryUserProfileStore()
    projection = SelectiveFailProjection(fail_entry_ids=frozenset())
    manager = UserProfileManager(
        store,
        tenant_id=_TENANT,
        memory_projections=(projection,),
    )
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
    )
    identity = RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )
    scope = user_memory_scope(identity)
    older = await plane.remember(
        identity, scope, MemoryControlRememberRequest(content="old", title="fact")
    )
    newer = await plane.remember(
        identity, scope, MemoryControlRememberRequest(content="new", title="fact")
    )
    assert older.entry_id and newer.entry_id
    projection.fail_entry_ids = frozenset({older.entry_id})
    with pytest.raises(MemoryControlPartialLifecycleError) as exc_info:
        await plane.apply_memory_supersession(
            identity,
            scope,
            MemorySupersessionIntent(
                superseded_memory_id=older.entry_id,
                superseding_memory_id=newer.entry_id,
                reason="test",
            ),
        )
    lifecycle = exc_info.value.lifecycle
    assert lifecycle.disposition is MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE
    assert older.entry_id in lifecycle.memory_entity_ids
    assert newer.entry_id in lifecycle.memory_entity_ids
    assert older.entry_id in projection.upsert_calls
    assert newer.entry_id in projection.upsert_calls


@pytest.mark.asyncio
async def test_supersession_projection_both_fail_aggregates_evidence() -> None:
    store = InMemoryUserProfileStore()
    projection = SelectiveFailProjection()
    manager = UserProfileManager(
        store,
        tenant_id=_TENANT,
        memory_projections=(projection,),
    )
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
    )
    identity = RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )
    scope = user_memory_scope(identity)
    older = await plane.remember(
        identity, scope, MemoryControlRememberRequest(content="old", title="fact")
    )
    newer = await plane.remember(
        identity, scope, MemoryControlRememberRequest(content="new", title="fact")
    )
    assert older.entry_id and newer.entry_id
    projection.fail_entry_ids = frozenset({older.entry_id, newer.entry_id})
    with pytest.raises(MemoryControlPartialLifecycleError) as exc_info:
        await plane.apply_memory_supersession(
            identity,
            scope,
            MemorySupersessionIntent(
                superseded_memory_id=older.entry_id,
                superseding_memory_id=newer.entry_id,
                reason="test",
            ),
        )
    assert len(exc_info.value.lifecycle.projection_evidence) >= 2


async def _supersession_with_projections(
    *projections: SelectiveFailProjection,
) -> tuple[DefaultMemoryControlPlane, RequestIdentity, object, object]:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(
        store,
        tenant_id=_TENANT,
        memory_projections=projections,
    )
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
    )
    identity = RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )
    scope = user_memory_scope(identity)
    older = await plane.remember(
        identity, scope, MemoryControlRememberRequest(content="old", title="fact")
    )
    newer = await plane.remember(
        identity, scope, MemoryControlRememberRequest(content="new", title="fact")
    )
    return plane, identity, scope, older, newer


@pytest.mark.asyncio
async def test_supersession_dual_projection_success_success_consistent() -> None:
    projection_a = SelectiveFailProjection(projection_id="projection_a")
    projection_b = SelectiveFailProjection(projection_id="projection_b")
    plane, identity, scope, older, newer = await _supersession_with_projections(
        projection_a,
        projection_b,
    )
    assert older.entry_id and newer.entry_id
    result = await plane.apply_memory_supersession(
        identity,
        scope,
        MemorySupersessionIntent(
            superseded_memory_id=older.entry_id,
            superseding_memory_id=newer.entry_id,
            reason="test",
        ),
    )
    assert result.lifecycle.disposition is MemoryLifecycleDisposition.CONSISTENT
    assert older.entry_id in projection_a.upsert_calls
    assert newer.entry_id in projection_a.upsert_calls
    assert older.entry_id in projection_b.upsert_calls
    assert newer.entry_id in projection_b.upsert_calls


@pytest.mark.asyncio
async def test_supersession_dual_projection_partial_success() -> None:
    projection_a = SelectiveFailProjection(projection_id="projection_a")
    projection_b = SelectiveFailProjection(projection_id="projection_b")
    plane, identity, scope, older, newer = await _supersession_with_projections(
        projection_a,
        projection_b,
    )
    assert older.entry_id and newer.entry_id
    projection_a.always_fail = True
    with pytest.raises(MemoryControlPartialLifecycleError) as exc_info:
        await plane.apply_memory_supersession(
            identity,
            scope,
            MemorySupersessionIntent(
                superseded_memory_id=older.entry_id,
                superseding_memory_id=newer.entry_id,
                reason="test",
            ),
        )
    lifecycle = exc_info.value.lifecycle
    assert lifecycle.disposition is MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE
    assert older.entry_id in lifecycle.memory_entity_ids
    assert newer.entry_id in lifecycle.memory_entity_ids
    failed_a = [
        ev
        for ev in lifecycle.projection_evidence
        if ev.projection_id == "projection_a" and not ev.succeeded
    ]
    assert failed_a


@pytest.mark.asyncio
async def test_supersession_dual_projection_success_partial() -> None:
    projection_a = SelectiveFailProjection(projection_id="projection_a")
    projection_b = SelectiveFailProjection(projection_id="projection_b")
    plane, identity, scope, older, newer = await _supersession_with_projections(
        projection_a,
        projection_b,
    )
    assert older.entry_id and newer.entry_id
    projection_b.fail_entry_ids = frozenset({older.entry_id})
    with pytest.raises(MemoryControlPartialLifecycleError) as exc_info:
        await plane.apply_memory_supersession(
            identity,
            scope,
            MemorySupersessionIntent(
                superseded_memory_id=older.entry_id,
                superseding_memory_id=newer.entry_id,
                reason="test",
            ),
        )
    lifecycle = exc_info.value.lifecycle
    assert lifecycle.disposition is MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE
    assert older.entry_id in lifecycle.memory_entity_ids
    assert newer.entry_id in lifecycle.memory_entity_ids
    assert older.entry_id in projection_a.upsert_calls
    assert newer.entry_id in projection_a.upsert_calls
    assert older.entry_id in projection_b.upsert_calls
    assert newer.entry_id in projection_b.upsert_calls
    failed_b = [
        ev
        for ev in lifecycle.projection_evidence
        if ev.projection_id == "projection_b" and not ev.succeeded
    ]
    assert failed_b


@pytest.mark.asyncio
async def test_supersession_dual_projection_partial_partial() -> None:
    projection_a = SelectiveFailProjection(projection_id="projection_a")
    projection_b = SelectiveFailProjection(projection_id="projection_b")
    plane, identity, scope, older, newer = await _supersession_with_projections(
        projection_a,
        projection_b,
    )
    assert older.entry_id and newer.entry_id
    projection_a.always_fail = True
    projection_b.always_fail = True
    with pytest.raises(MemoryControlPartialLifecycleError) as exc_info:
        await plane.apply_memory_supersession(
            identity,
            scope,
            MemorySupersessionIntent(
                superseded_memory_id=older.entry_id,
                superseding_memory_id=newer.entry_id,
                reason="test",
            ),
        )
    lifecycle = exc_info.value.lifecycle
    assert lifecycle.disposition is MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE
    assert len(lifecycle.projection_evidence) >= 4
