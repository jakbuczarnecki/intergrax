# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 5 governance tests."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityCatalogEntry,
    CapabilityDiscoveryCandidate,
    CapabilityGovernanceError,
    GovernedCapabilityCandidate,
    GovernedDiscoveryResult,
    RankedCapabilityCandidate,
    StableIdentityRanker,
    govern_capability_candidates,
    rank_capability_candidates,
)
from intergrax.capability_catalog.governance import (
    CapabilityGovernanceDecision,
    _evaluate_candidate,
)
from intergrax.capability_catalog.governance_validation import validate_governed_output
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryIdentity,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityGovernanceReasonCode,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilityRankingEvidence,
    CapabilityRankingSignal,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    GovernanceDecisionEvidence,
    GovernanceDisposition,
)

pytestmark = pytest.mark.unit

_BASELINE = AvailabilityPreservingGovernanceEvaluator()


def _source() -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id="official.catalog",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )


def _entry(
    *,
    kind: CapabilityKind = CapabilityKind.TOOL,
    logical_id: str = "tools.echo.ping",
) -> CapabilityCatalogEntry:
    source = _source()
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(source=source, version_label="1.0.0"),
        display_label=logical_id,
    )


def _ranked(
    entry: CapabilityCatalogEntry,
    *,
    availability: AvailabilityDisposition = AvailabilityDisposition.CATALOG_AVAILABLE,
    position: int = 1,
) -> RankedCapabilityCandidate:
    candidate = CapabilityDiscoveryCandidate(
        catalog_entry=entry,
        availability=availability,
    )
    return RankedCapabilityCandidate(
        candidate=candidate,
        evidence=CapabilityRankingEvidence(
            ranker_id="stable.identity",
            rank_position=position,
            signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
            original_stage3_position=position,
        ),
    )


def _rank_many(
    *items: tuple[CapabilityCatalogEntry, AvailabilityDisposition],
) -> tuple[RankedCapabilityCandidate, ...]:
    candidates = tuple(
        CapabilityDiscoveryCandidate(catalog_entry=entry, availability=availability)
        for entry, availability in items
    )
    return rank_capability_candidates(candidates, StableIdentityRanker())


def test_empty_input_returns_empty_partitions() -> None:
    result = govern_capability_candidates((), evaluators=(_BASELINE,))
    assert result == GovernedDiscoveryResult(allowed=(), blocked=())


def test_all_allowed_preserves_order() -> None:
    ranked = _rank_many(
        (_entry(logical_id="tools.alpha"), AvailabilityDisposition.CATALOG_AVAILABLE),
        (_entry(logical_id="tools.beta"), AvailabilityDisposition.HOST_AVAILABLE),
    )
    result = govern_capability_candidates(ranked, evaluators=(_BASELINE,))
    assert len(result.allowed) == 2
    assert not result.blocked
    assert [item.identity.logical.logical_id for item in result.allowed] == [
        "tools.alpha",
        "tools.beta",
    ]


def test_one_blocked_preserves_relative_order() -> None:
    ranked = _rank_many(
        (_entry(logical_id="tools.alpha"), AvailabilityDisposition.CATALOG_AVAILABLE),
        (_entry(logical_id="tools.beta"), AvailabilityDisposition.BLOCKED),
        (_entry(logical_id="tools.gamma"), AvailabilityDisposition.CATALOG_AVAILABLE),
    )
    result = govern_capability_candidates(ranked, evaluators=(_BASELINE,))
    assert [item.identity.logical.logical_id for item in result.allowed] == [
        "tools.alpha",
        "tools.gamma",
    ]
    assert [item.identity.logical.logical_id for item in result.blocked] == ["tools.beta"]


def test_all_blocked() -> None:
    ranked = _rank_many(
        (_entry(logical_id="tools.alpha"), AvailabilityDisposition.UNAVAILABLE),
        (_entry(logical_id="tools.beta"), AvailabilityDisposition.SCOPE_UNAVAILABLE),
    )
    result = govern_capability_candidates(ranked, evaluators=(_BASELINE,))
    assert not result.allowed
    assert len(result.blocked) == 2


def test_blocked_carries_typed_reason() -> None:
    ranked = (_ranked(_entry(), availability=AvailabilityDisposition.BLOCKED),)
    result = govern_capability_candidates(ranked, evaluators=(_BASELINE,))
    blocked = result.blocked[0]
    assert any(
        item.reason_code is CapabilityGovernanceReasonCode.AVAILABILITY_BLOCKED
        for item in blocked.evidence
    )


def test_exactly_once_membership() -> None:
    ranked = _rank_many(
        (_entry(logical_id="tools.alpha"), AvailabilityDisposition.CATALOG_AVAILABLE),
        (_entry(logical_id="tools.beta"), AvailabilityDisposition.CATALOG_AVAILABLE),
    )
    result = govern_capability_candidates(ranked, evaluators=(_BASELINE,))
    keys = {
        item.ranked.identity.sort_key
        for item in (*result.allowed, *result.blocked)
    }
    assert len(keys) == 2


class _DropCandidateEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "malicious.drop"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        if candidate.identity.logical.logical_id == "tools.beta":
            raise RuntimeError("skip candidate")
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.ALLOWED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id=self.evaluator_id,
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        )


def test_evaluator_exception_strict_fail_closed() -> None:
    ranked = _rank_many(
        (_entry(logical_id="tools.alpha"), AvailabilityDisposition.CATALOG_AVAILABLE),
        (_entry(logical_id="tools.beta"), AvailabilityDisposition.CATALOG_AVAILABLE),
    )
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT)
    result = govern_capability_candidates(
        ranked,
        evaluators=(_BASELINE, _DropCandidateEvaluator()),
        context=context,
    )
    assert len(result.allowed) == 1
    assert len(result.blocked) == 1
    assert result.blocked[0].identity.logical.logical_id == "tools.beta"
    assert any(
        item.reason_code is CapabilityGovernanceReasonCode.EVALUATOR_FAILURE
        for item in result.blocked[0].evidence
    )


class _MutateAvailabilityEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "malicious.mutate"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        del candidate, context
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.ALLOWED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id=self.evaluator_id,
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        )


def test_validator_rejects_elevated_blocked_candidate() -> None:
    ranked = (_ranked(_entry(), availability=AvailabilityDisposition.BLOCKED),)
    blocked_item = govern_capability_candidates(ranked, evaluators=(_BASELINE,)).blocked[0]
    allowed_evidence = (
        GovernanceDecisionEvidence(
            evaluator_id="test.forge",
            disposition=GovernanceDisposition.ALLOWED,
            reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
        ),
    )
    forged = GovernedDiscoveryResult(
        allowed=(
            GovernedCapabilityCandidate(
                ranked=blocked_item.ranked,
                evidence=allowed_evidence,
            ),
        ),
        blocked=(),
    )
    with pytest.raises(CapabilityGovernanceError, match="elevate"):
        validate_governed_output(input_ranked=ranked, result=forged)


class _InvalidReasonEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "malicious.reason"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.ALLOWED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id="wrong.evaluator",
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        )


def test_invalid_evaluator_output_fail_closed() -> None:
    ranked = (_ranked(_entry()),)
    with pytest.raises(CapabilityGovernanceError):
        govern_capability_candidates(
            ranked,
            evaluators=(_BASELINE, _InvalidReasonEvaluator()),
        )


class _AllowBlockedEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "malicious.allow_blocked"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.ALLOWED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id=self.evaluator_id,
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        )


def test_baseline_blocks_even_when_later_evaluator_allows() -> None:
    ranked = (_ranked(_entry(), availability=AvailabilityDisposition.BLOCKED),)
    result = govern_capability_candidates(
        ranked,
        evaluators=(_BASELINE, _AllowBlockedEvaluator()),
    )
    assert len(result.blocked) == 1
    assert not result.allowed


def test_preserves_identity_provenance_ranking_evidence() -> None:
    ranked = (_ranked(_entry()),)
    governed = govern_capability_candidates(ranked, evaluators=(_BASELINE,)).allowed[0]
    assert governed.ranked == ranked[0]
    assert governed.ranking_evidence == ranked[0].evidence


def test_strict_empty_pipeline_evaluates_candidate_as_allowed_without_evidence() -> None:
    """Regression anchor: empty STRICT pipeline used to reach ALLOWED disposition."""
    ranked = (_ranked(_entry(), availability=AvailabilityDisposition.HOST_AVAILABLE),)
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT)
    disposition, evidence = _evaluate_candidate(ranked[0], (), context)
    assert disposition is GovernanceDisposition.ALLOWED
    assert evidence == ()


@pytest.mark.parametrize(
    "availability",
    [
        AvailabilityDisposition.HOST_AVAILABLE,
        AvailabilityDisposition.CATALOG_AVAILABLE,
    ],
)
def test_strict_empty_evaluator_pipeline_raises_configuration_error(
    availability: AvailabilityDisposition,
) -> None:
    ranked = (_ranked(_entry(), availability=availability),)
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT)
    with pytest.raises(
        CapabilityGovernanceError,
        match="STRICT capability governance requires at least one evaluator",
    ):
        govern_capability_candidates(ranked, evaluators=(), context=context)


def test_strict_empty_evaluator_pipeline_with_empty_candidates_raises() -> None:
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT)
    with pytest.raises(
        CapabilityGovernanceError,
        match="STRICT capability governance requires at least one evaluator",
    ):
        govern_capability_candidates((), evaluators=(), context=context)


def test_non_strict_empty_evaluator_pipeline_with_empty_candidates() -> None:
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.NON_STRICT)
    result = govern_capability_candidates((), evaluators=(), context=context)
    assert result == GovernedDiscoveryResult(allowed=(), blocked=())


def test_strict_with_valid_evaluator_pipeline_works() -> None:
    ranked = (_ranked(_entry(), availability=AvailabilityDisposition.HOST_AVAILABLE),)
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT)
    result = govern_capability_candidates(
        ranked,
        evaluators=(_BASELINE,),
        context=context,
    )
    assert len(result.allowed) == 1
    assert not result.blocked


class _RuntimeFailureEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "test.runtime_failure"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        del candidate, context
        raise RuntimeError("evaluator exploded")


def test_strict_evaluator_runtime_failure_blocks_candidate() -> None:
    ranked = (_ranked(_entry()),)
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT)
    result = govern_capability_candidates(
        ranked,
        evaluators=(_RuntimeFailureEvaluator(),),
        context=context,
    )
    assert not result.allowed
    assert len(result.blocked) == 1
    assert any(
        item.reason_code is CapabilityGovernanceReasonCode.EVALUATOR_FAILURE
        for item in result.blocked[0].evidence
    )


def test_strict_evaluator_contract_violation_raises_operation_error() -> None:
    ranked = (_ranked(_entry()),)
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT)
    with pytest.raises(CapabilityGovernanceError):
        govern_capability_candidates(
            ranked,
            evaluators=(_BASELINE, _InvalidReasonEvaluator()),
            context=context,
        )


class _DuplicateIdEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "policy"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        del candidate, context
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.ALLOWED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id=self.evaluator_id,
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        )


@pytest.mark.parametrize(
    "posture",
    [
        CapabilityGovernancePosture.STRICT,
        CapabilityGovernancePosture.NON_STRICT,
    ],
)
def test_duplicate_evaluator_ids_raise_configuration_error(
    posture: CapabilityGovernancePosture,
) -> None:
    ranked = (_ranked(_entry()),)
    context = CapabilityGovernanceContext(posture=posture)
    with pytest.raises(
        CapabilityGovernanceError,
        match="evaluator_id values must be unique",
    ):
        govern_capability_candidates(
            ranked,
            evaluators=(_DuplicateIdEvaluator(), _DuplicateIdEvaluator()),
            context=context,
        )


class _BlankEvaluatorId:
    def __init__(self, evaluator_id: str) -> None:
        self._evaluator_id = evaluator_id

    @property
    def evaluator_id(self) -> str:
        return self._evaluator_id

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        del candidate, context
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.ALLOWED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id="valid.id",
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        )


@pytest.mark.parametrize("evaluator_id", ["", "   "])
def test_blank_evaluator_id_raises_configuration_error(evaluator_id: str) -> None:
    ranked = (_ranked(_entry()),)
    with pytest.raises(CapabilityGovernanceError):
        govern_capability_candidates(
            ranked,
            evaluators=(_BlankEvaluatorId(evaluator_id),),
        )
