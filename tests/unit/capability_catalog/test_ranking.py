# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 4 ranking tests."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    CapabilityCatalogEntry,
    CapabilityDiscoveryCandidate,
    CapabilityRankingError,
    RankedCapabilityCandidate,
    StableIdentityRanker,
    rank_capability_candidates,
)
from intergrax.capability_catalog.adapters import (
    AgentStableIdentityCapabilityRanker,
    KeywordOverlapToolCapabilityRanker,
)
from intergrax.capability_catalog.adapters.tool_ranking import _tool_keyword_search_document
from intergrax.capability_catalog.ranking import STABLE_IDENTITY_RANKER_ID
from intergrax.tools.search.keyword_ranking import (
    ToolKeywordSearchDocument,
    score_tool_keyword_document,
    tokenize_tool_search_query,
)
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilityRankingContext,
    CapabilityRankingEvidence,
    CapabilityRankingSignal,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)

pytestmark = pytest.mark.unit


def _source(
    source_id: str = "official.catalog",
    kind: CapabilitySourceKind = CapabilitySourceKind.OFFICIAL,
) -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(source_id=source_id, source_kind=kind)


def _entry(
    *,
    kind: CapabilityKind = CapabilityKind.TOOL,
    logical_id: str = "tools.echo.ping",
    display_label: str | None = None,
    use_logical_id_fallback: bool = True,
) -> CapabilityCatalogEntry:
    source = _source()
    resolved_label = display_label
    if resolved_label is None and use_logical_id_fallback:
        resolved_label = logical_id
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(source=source, version_label="1.0.0"),
        display_label=resolved_label,
    )


def _candidate(
    entry: CapabilityCatalogEntry,
    availability: AvailabilityDisposition = AvailabilityDisposition.CATALOG_AVAILABLE,
) -> CapabilityDiscoveryCandidate:
    return CapabilityDiscoveryCandidate(catalog_entry=entry, availability=availability)


def test_empty_candidates_returns_empty_ranked_output() -> None:
    ranker = StableIdentityRanker()
    assert rank_capability_candidates((), ranker) == ()


def test_one_candidate_rank_position_is_one() -> None:
    entry = _entry()
    candidate = _candidate(entry)
    ranked = rank_capability_candidates((candidate,), StableIdentityRanker())
    assert len(ranked) == 1
    assert ranked[0].candidate is candidate
    assert ranked[0].evidence.rank_position == 1
    assert ranked[0].evidence.ranker_id == STABLE_IDENTITY_RANKER_ID


def test_multiple_kinds_deterministic_identity_order() -> None:
    agent = _candidate(_entry(kind=CapabilityKind.AGENT, logical_id="agents.alpha"))
    skill = _candidate(_entry(kind=CapabilityKind.SKILL, logical_id="skills.beta"))
    tool = _candidate(_entry(kind=CapabilityKind.TOOL, logical_id="tools.gamma"))
    shuffled = (tool, agent, skill)
    first = rank_capability_candidates(shuffled, StableIdentityRanker())
    second = rank_capability_candidates(shuffled, StableIdentityRanker())
    assert first == second
    assert [item.identity.logical.logical_id for item in first] == [
        "agents.alpha",
        "skills.beta",
        "tools.gamma",
    ]
    assert [item.evidence.rank_position for item in first] == [1, 2, 3]


def test_preserves_identity_provenance_availability() -> None:
    entry = _entry(kind=CapabilityKind.TOOL, logical_id="tools.blocked")
    blocked = _candidate(entry, AvailabilityDisposition.BLOCKED)
    ranked = rank_capability_candidates((blocked,), StableIdentityRanker())[0]
    assert ranked.identity == blocked.identity
    assert ranked.provenance == blocked.provenance
    assert ranked.availability is AvailabilityDisposition.BLOCKED


def test_keyword_tool_ranker_orders_by_overlap_then_identity() -> None:
    echo = _candidate(
        _entry(
            kind=CapabilityKind.TOOL,
            logical_id="tools.echo.ping",
            display_label="echo ping utility",
        ),
    )
    browser = _candidate(
        _entry(
            kind=CapabilityKind.TOOL,
            logical_id="tools.browser.navigate",
            display_label="browser navigation",
        ),
    )
    ranker = KeywordOverlapToolCapabilityRanker()
    ranked = rank_capability_candidates(
        (browser, echo),
        ranker,
        context=CapabilityRankingContext(semantic_need="echo ping"),
    )
    assert ranked[0].identity.logical.logical_id == "tools.echo.ping"
    assert ranked[0].evidence.signal is CapabilityRankingSignal.KEYWORD_OVERLAP
    assert ranked[0].evidence.score == 2.0


def test_keyword_tool_ranker_equal_scores_use_identity_tie_break() -> None:
    left = _candidate(_entry(kind=CapabilityKind.TOOL, logical_id="tools.alpha"))
    right = _candidate(_entry(kind=CapabilityKind.TOOL, logical_id="tools.beta"))
    ranked = rank_capability_candidates(
        (right, left),
        KeywordOverlapToolCapabilityRanker(),
        context=CapabilityRankingContext(semantic_need="zzz"),
    )
    assert [item.identity.logical.logical_id for item in ranked] == [
        "tools.alpha",
        "tools.beta",
    ]


def test_keyword_tool_ranker_display_label_none_scores_from_logical_id() -> None:
    candidate = _candidate(
        _entry(
            kind=CapabilityKind.TOOL,
            logical_id="tools.echo.ping",
            display_label=None,
            use_logical_id_fallback=False,
        ),
    )
    ranked = rank_capability_candidates(
        (candidate,),
        KeywordOverlapToolCapabilityRanker(),
        context=CapabilityRankingContext(semantic_need="echo"),
    )
    assert ranked[0].evidence.score == 1.0


def test_keyword_tool_ranker_display_label_none_unrelated_query_scores_zero() -> None:
    candidate = _candidate(
        _entry(
            kind=CapabilityKind.TOOL,
            logical_id="tools.echo.ping",
            display_label=None,
            use_logical_id_fallback=False,
        ),
    )
    ranked = rank_capability_candidates(
        (candidate,),
        KeywordOverlapToolCapabilityRanker(),
        context=CapabilityRankingContext(semantic_need="browser navigation"),
    )
    assert ranked[0].evidence.score == 0.0


def test_keyword_tool_ranker_unicode_display_label() -> None:
    candidate = _candidate(
        _entry(
            kind=CapabilityKind.TOOL,
            logical_id="tools.locale.greeting",
            display_label="Witaj świecie",
            use_logical_id_fallback=False,
        ),
    )
    ranked = rank_capability_candidates(
        (candidate,),
        KeywordOverlapToolCapabilityRanker(),
        context=CapabilityRankingContext(semantic_need="witaj"),
    )
    assert ranked[0].evidence.score == 1.0


def test_keyword_tool_ranker_logical_id_only_projection() -> None:
    candidate = _candidate(
        _entry(
            kind=CapabilityKind.TOOL,
            logical_id="tools.search.catalog",
            display_label=None,
            use_logical_id_fallback=False,
        ),
    )
    document = _tool_keyword_search_document(candidate)
    tokens = tokenize_tool_search_query("catalog search")
    assert score_tool_keyword_document(document, tokens) == 2


def test_display_label_empty_string_rejected_by_stage2_contract() -> None:
    with pytest.raises(ValueError, match="display_label must be non-empty"):
        _entry(display_label="   ", use_logical_id_fallback=False)


def test_stage4_and_shared_primitive_parity_on_minimal_document() -> None:
    candidate = _candidate(
        _entry(
            kind=CapabilityKind.TOOL,
            logical_id="tools.search.catalog",
            display_label="Catalog Search",
            use_logical_id_fallback=False,
        ),
    )
    tokens = tokenize_tool_search_query("catalog search")
    stage4_score = score_tool_keyword_document(_tool_keyword_search_document(candidate), tokens)
    shared_score = score_tool_keyword_document(
        ToolKeywordSearchDocument(
            tool_id="tools.search.catalog",
            text_parts=("Catalog Search",),
        ),
        tokens,
    )
    assert stage4_score == shared_score == 2


def test_agent_and_baseline_rankers_are_interchangeable_plugins() -> None:
    candidates = (
        _candidate(_entry(kind=CapabilityKind.AGENT, logical_id="agents.z")),
        _candidate(_entry(kind=CapabilityKind.AGENT, logical_id="agents.a")),
    )
    baseline = rank_capability_candidates(candidates, StableIdentityRanker())
    agent_ranker = rank_capability_candidates(candidates, AgentStableIdentityCapabilityRanker())
    assert [item.identity.logical.logical_id for item in baseline] == [
        "agents.a",
        "agents.z",
    ]
    assert [item.identity.logical.logical_id for item in agent_ranker] == [
        "agents.a",
        "agents.z",
    ]
    assert agent_ranker[0].evidence.ranker_id == "agent.stable_identity"


class _DropCandidateRanker:
    @property
    def ranker_id(self) -> str:
        return "broken.drop"

    def rank(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        context: CapabilityRankingContext,
    ) -> tuple[RankedCapabilityCandidate, ...]:
        if not candidates:
            return ()
        return (
            RankedCapabilityCandidate(
                candidate=candidates[0],
                evidence=CapabilityRankingEvidence(
                    ranker_id=self.ranker_id,
                    rank_position=1,
                    signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                ),
            ),
        )


class _DuplicateCandidateRanker:
    @property
    def ranker_id(self) -> str:
        return "broken.duplicate"

    def rank(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        context: CapabilityRankingContext,
    ) -> tuple[RankedCapabilityCandidate, ...]:
        first = candidates[0]
        return (
            RankedCapabilityCandidate(
                candidate=first,
                evidence=CapabilityRankingEvidence(
                    ranker_id=self.ranker_id,
                    rank_position=1,
                    signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                ),
            ),
            RankedCapabilityCandidate(
                candidate=first,
                evidence=CapabilityRankingEvidence(
                    ranker_id=self.ranker_id,
                    rank_position=2,
                    signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                ),
            ),
        )


class _MutatedAvailabilityRanker:
    @property
    def ranker_id(self) -> str:
        return "broken.mutate"

    def rank(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        context: CapabilityRankingContext,
    ) -> tuple[RankedCapabilityCandidate, ...]:
        first, second = candidates[0], candidates[1]
        mutated = CapabilityDiscoveryCandidate(
            catalog_entry=first.catalog_entry,
            availability=AvailabilityDisposition.HOST_AVAILABLE,
        )
        return (
            RankedCapabilityCandidate(
                candidate=mutated,
                evidence=CapabilityRankingEvidence(
                    ranker_id=self.ranker_id,
                    rank_position=1,
                    signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                ),
            ),
            RankedCapabilityCandidate(
                candidate=second,
                evidence=CapabilityRankingEvidence(
                    ranker_id=self.ranker_id,
                    rank_position=2,
                    signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                ),
            ),
        )


class _NonContiguousRanker:
    @property
    def ranker_id(self) -> str:
        return "broken.ranks"

    def rank(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        context: CapabilityRankingContext,
    ) -> tuple[RankedCapabilityCandidate, ...]:
        return tuple(
            RankedCapabilityCandidate(
                candidate=candidate,
                evidence=CapabilityRankingEvidence(
                    ranker_id=self.ranker_id,
                    rank_position=index + 2,
                    signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                ),
            )
            for index, candidate in enumerate(candidates)
        )


@pytest.mark.parametrize(
    ("ranker", "message"),
    (
        (_DropCandidateRanker(), "same number"),
        (_DuplicateCandidateRanker(), "duplicate"),
        (_MutatedAvailabilityRanker(), "mutate"),
        (_NonContiguousRanker(), "contiguous"),
    ),
)
def test_broken_rankers_fail_closed(ranker: object, message: str) -> None:
    candidates = (
        _candidate(_entry(logical_id="tools.one")),
        _candidate(_entry(logical_id="tools.two")),
    )
    with pytest.raises(CapabilityRankingError, match=message):
        rank_capability_candidates(candidates, ranker)  # type: ignore[arg-type]
