# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-6: recall ranking, conflict strategies, and control plane integration."""

from __future__ import annotations

import copy
from dataclasses import dataclass

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
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
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
    MemoryConflictResolutionAction,
    MemoryRankingRequest,
    MemoryRankingResult,
    MemoryRankingScore,
    MemoryRankedCandidate,
    MemoryRecallCandidate,
    MemoryRetrievalSource,
    MemorySupersessionIntent,
)
from intergrax.memory.strategies.recall_validation import validate_ranking_result
from intergrax.memory.user_profile_manager import UserProfileManager

pytestmark = pytest.mark.gate

_TENANT = "tenant-mem6"
_USER = "user-mem6"


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
) -> UserProfileMemoryEntry:
    return UserProfileMemoryEntry(
        entry_id=entry_id,
        revision=revision,
        content=content,
        kind=MemoryKind.USER_FACT,
        title=title,
        trust=MemoryRecordTrust(trust_class=trust_class, confidence=confidence),
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
        lineage=MemoryRecordLineage(superseded_by_memory_id=superseded_by),
        valid_until=valid_until,
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
    request = MemoryRankingRequest(candidates=entries, query="q", top_k=3)
    first = ranker.rank(request)
    second = ranker.rank(request)
    assert [r.candidate.record.entry_id for r in first.ranked] == [
        r.candidate.record.entry_id for r in second.ranked
    ]
    assert [r.score.total for r in first.ranked] == [r.score.total for r in second.ranked]


def test_ranking_tie_break_uses_memory_id() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    shared_score = 0.75
    entries = (
        _candidate(_entry("zzzz", "x", title="t"), shared_score),
        _candidate(_entry("aaaa", "y", title="t"), shared_score),
    )
    result = ranker.rank(MemoryRankingRequest(candidates=entries, query="q", top_k=2))
    assert [r.candidate.record.entry_id for r in result.ranked] == ["aaaa", "zzzz"]


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
    result = ranker.rank(
        MemoryRankingRequest(candidates=(low_trust, high_trust), query="q", top_k=2)
    )
    assert result.ranked[0].candidate.record.entry_id == "low"
    assert result.ranked[1].score.total < result.ranked[0].score.total


def test_superseded_record_excluded_from_default_ranking() -> None:
    ranker = EnterpriseMemoryRankingStrategy()
    active = _candidate(_entry("active", "current"))
    superseded = _candidate(_entry("old", "stale", superseded_by="active"))
    result = ranker.rank(
        MemoryRankingRequest(candidates=(active, superseded), query="q", top_k=5)
    )
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
    assert result.conflicts[0].kind.value in {"contradiction", "potential_supersession"}


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


def test_resolver_supersede_without_mutation() -> None:
    from intergrax.memory.strategies.recall_models import (
        MemoryConflict,
        MemoryConflictKind,
        MemoryConflictResolutionRequest,
    )

    resolver = FailSafeMemoryConflictResolutionStrategy()
    left = _entry("older", "old value", title="fact", revision=1)
    right = _entry("newer", "new value", title="fact", revision=3)
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
