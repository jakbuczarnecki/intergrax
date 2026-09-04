# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryCandidate,
    AgentDiscoveryRequest,
    AgentDiscoveryStrategyId,
    StaticAgentDiscoveryStrategy,
    project_package_contract_capabilities,
    project_to_capability_candidate,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
)
from intergrax.agent_distribution.agent_selection import (
    AgentSelectionContractError,
    AgentSelectionDecision,
    AgentSelectionIdentityConflict,
    AgentSelectionNoEligibleCandidate,
    AgentSelectionRequest,
    AgentSelectionStrategy,
    AgentSelectionStrategyId,
    DeterministicIdentitySelectionStrategy,
    SelectionDecisionBasis,
    SelectionOutcome,
    build_agent_selection_request,
    require_selected_identity,
)
from intergrax.agent_distribution.capability_matching import (
    CapabilityMatcher,
    CapabilityMatchResult,
    build_agent_capability_requirement,
)
from intergrax.agent_distribution.catalog import (
    AgentDiscoveryCandidateIdentity,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.identity import AgentPackageCandidate


def _source(source_id: str, kind: CatalogProviderKind) -> CatalogSourceIdentity:
    return CatalogSourceIdentity(
        catalog_source_id=source_id,
        provider_kind=kind,
    )


def _package(
    package_id: str,
    *,
    version: str = "1.0.0",
    digest: str | None = None,
) -> AgentPackageCandidate:
    return AgentPackageCandidate(
        distribution_package_id=package_id,
        package_version=version,
        package_digest=digest,
    )


def _identity(
    source_id: str,
    package_id: str,
    *,
    kind: CatalogProviderKind = CatalogProviderKind.LOCAL_DEVELOPER,
    version: str = "1.0.0",
    digest: str | None = None,
) -> AgentDiscoveryCandidateIdentity:
    return AgentDiscoveryCandidateIdentity(
        source=_source(source_id, kind),
        package=_package(package_id, version=version, digest=digest),
    )


def _eligible_match(identity: AgentDiscoveryCandidateIdentity) -> CapabilityMatchResult:
    return CapabilityMatchResult(
        identity=identity,
        eligible=True,
        matched_required=(),
        missing_required=(),
        matched_optional=(),
    )


def _ineligible_match(
    identity: AgentDiscoveryCandidateIdentity,
) -> CapabilityMatchResult:
    return CapabilityMatchResult(
        identity=identity,
        eligible=False,
        matched_required=(),
        missing_required=(),
        matched_optional=(),
    )


def _request(
    *,
    matches: tuple[CapabilityMatchResult, ...],
) -> AgentSelectionRequest:
    return build_agent_selection_request(
        requirement=build_agent_capability_requirement(required=("document.search",)),
        eligible_matches=matches,
    )


def _discovery_candidate(
    source_id: str,
    package_id: str,
    *,
    capability_ids: tuple[str, ...] = (),
    kind: CatalogProviderKind = CatalogProviderKind.LOCAL_DEVELOPER,
) -> AgentDiscoveryCandidate:
    return AgentDiscoveryCandidate(
        identity=_identity(source_id, package_id, kind=kind),
        capabilities=project_package_contract_capabilities(
            AgentPackageContractDeclaration(
                contract_id="contract.v1",
                contract_version="1",
                capabilities=capability_ids,
            ),
        ),
        catalog_entry_id=f"{source_id}:{package_id}",
    )


class _FakeSelectionStrategy:
    """Structural Protocol implementation without inheritance."""

    def __init__(self, strategy_id: str) -> None:
        self._strategy_id = AgentSelectionStrategyId(value=strategy_id)

    @property
    def strategy_id(self) -> AgentSelectionStrategyId:
        return self._strategy_id

    def select(self, request: AgentSelectionRequest) -> AgentSelectionDecision:
        considered = tuple(match.identity for match in request.eligible_matches)
        if not considered:
            return AgentSelectionDecision(
                strategy_id=self._strategy_id,
                outcome=SelectionOutcome.NO_ELIGIBLE_CANDIDATE,
                considered_candidates=(),
            )
        selected = considered[-1]
        return AgentSelectionDecision(
            strategy_id=self._strategy_id,
            outcome=SelectionOutcome.SELECTED,
            selected_identity=selected,
            considered_candidates=considered,
            decision_basis=SelectionDecisionBasis.STABLE_IDENTITY_ORDER,
        )


class _LastIdentitySelectionStrategy:
    """Test-only alternate strategy — selects last considered identity."""

    @property
    def strategy_id(self) -> AgentSelectionStrategyId:
        return AgentSelectionStrategyId(value="test.last_identity")

    def select(self, request: AgentSelectionRequest) -> AgentSelectionDecision:
        considered = tuple(
            match.identity
            for match in sorted(
                request.eligible_matches,
                key=lambda item: item.identity.sort_key,
            )
        )
        return AgentSelectionDecision(
            strategy_id=self.strategy_id,
            outcome=SelectionOutcome.SELECTED,
            selected_identity=considered[-1],
            considered_candidates=considered,
            decision_basis=SelectionDecisionBasis.STABLE_IDENTITY_ORDER,
        )


def test_structural_protocol_without_inheritance() -> None:
    strategy: AgentSelectionStrategy = _FakeSelectionStrategy("test.structural")
    assert strategy.strategy_id.value == "test.structural"


def test_one_eligible_candidate_selected() -> None:
    identity = _identity("source-a", "agent-a")
    strategy = DeterministicIdentitySelectionStrategy()
    decision = strategy.select(_request(matches=(_eligible_match(identity),)))

    assert decision.outcome is SelectionOutcome.SELECTED
    assert decision.selected_identity == identity
    assert decision.considered_candidates == (identity,)
    assert decision.decision_basis is SelectionDecisionBasis.STABLE_IDENTITY_ORDER
    assert decision.strategy_id.value == "deterministic.identity"


def test_two_eligible_candidates_stable_identity_ordering() -> None:
    identity_a = _identity("source-a", "agent-a")
    identity_b = _identity("source-b", "agent-b")
    strategy = DeterministicIdentitySelectionStrategy()
    decision = strategy.select(
        _request(matches=(_eligible_match(identity_b), _eligible_match(identity_a))),
    )

    assert decision.selected_identity == identity_a
    assert decision.considered_candidates == (identity_a, identity_b)


def test_input_order_reversed_same_selected() -> None:
    identity_a = _identity("source-a", "agent-a")
    identity_b = _identity("source-b", "agent-b")
    strategy = DeterministicIdentitySelectionStrategy()

    forward = strategy.select(
        _request(matches=(_eligible_match(identity_a), _eligible_match(identity_b))),
    )
    reversed_order = strategy.select(
        _request(matches=(_eligible_match(identity_b), _eligible_match(identity_a))),
    )

    assert forward.selected_identity == reversed_order.selected_identity == identity_a


def test_same_package_id_different_sources_unambiguous() -> None:
    identity_local = _identity(
        "source-local",
        "shared-agent",
        kind=CatalogProviderKind.LOCAL_DEVELOPER,
    )
    identity_remote = _identity(
        "source-remote",
        "shared-agent",
        kind=CatalogProviderKind.OFFICIAL_CATALOG,
    )
    strategy = DeterministicIdentitySelectionStrategy()
    decision = strategy.select(
        _request(
            matches=(
                _eligible_match(identity_remote),
                _eligible_match(identity_local),
            ),
        ),
    )

    assert decision.selected_identity == identity_local
    assert decision.considered_candidates == (identity_local, identity_remote)


def test_selected_identity_equals_matcher_identity() -> None:
    matcher = CapabilityMatcher()
    requirement = build_agent_capability_requirement(required=("document.search",))
    candidate = project_to_capability_candidate(
        _discovery_candidate(
            "source-a", "agent-a", capability_ids=("document.search",)
        ),
    )
    match = matcher.match(requirement=requirement, candidate=candidate)
    strategy = DeterministicIdentitySelectionStrategy()
    decision = strategy.select(
        build_agent_selection_request(
            requirement=requirement,
            eligible_matches=(match,),
        ),
    )

    assert decision.selected_identity == match.identity == candidate.identity


def test_zero_eligible_fails_closed() -> None:
    strategy = DeterministicIdentitySelectionStrategy()
    decision = strategy.select(
        build_agent_selection_request(
            requirement=build_agent_capability_requirement(
                required=("document.search",)
            ),
            eligible_matches=(),
        ),
    )

    assert decision.outcome is SelectionOutcome.NO_ELIGIBLE_CANDIDATE
    assert decision.selected_identity is None
    with pytest.raises(AgentSelectionNoEligibleCandidate):
        require_selected_identity(decision)


def test_ineligible_match_rejected_in_request() -> None:
    with pytest.raises(AgentSelectionContractError, match="only eligible"):
        _request(matches=(_ineligible_match(_identity("source-a", "agent-a")),))


def test_duplicate_identity_fails_closed() -> None:
    identity = _identity("source-a", "agent-a")
    with pytest.raises(AgentSelectionIdentityConflict, match="duplicate"):
        _request(matches=(_eligible_match(identity), _eligible_match(identity)))


def test_discovery_matching_selection_pipeline() -> None:
    requirement = build_agent_capability_requirement(required=("document.search",))
    discovery = StaticAgentDiscoveryStrategy(
        strategy_id=AgentDiscoveryStrategyId(value="test.static"),
        candidates=(
            _discovery_candidate(
                "source-a", "agent-a", capability_ids=("document.search",)
            ),
            _discovery_candidate("source-b", "agent-b", capability_ids=()),
            _discovery_candidate(
                "source-c", "agent-c", capability_ids=("document.search",)
            ),
        ),
    )
    discovery_result = discovery.discover(
        AgentDiscoveryRequest(requirement=requirement),
    )
    matcher = CapabilityMatcher()
    eligible = matcher.find_eligible(
        requirement=requirement,
        candidates=tuple(
            project_to_capability_candidate(candidate)
            for candidate in discovery_result.candidates
        ),
    )
    assert len(eligible) == 2
    assert all(result.eligible for result in eligible)
    assert not any(
        result.identity.sort_key == _identity("source-b", "agent-b").sort_key
        for result in eligible
    )

    strategy = DeterministicIdentitySelectionStrategy()
    decision = strategy.select(
        build_agent_selection_request(
            requirement=requirement,
            eligible_matches=eligible,
        ),
    )

    assert decision.outcome is SelectionOutcome.SELECTED
    assert decision.selected_identity == _identity("source-a", "agent-a")
    assert decision.considered_candidates == (
        _identity("source-a", "agent-a"),
        _identity("source-c", "agent-c"),
    )


def test_alternative_strategy_replaceable() -> None:
    identity_a = _identity("source-a", "agent-a")
    identity_b = _identity("source-b", "agent-b")
    request = _request(
        matches=(_eligible_match(identity_a), _eligible_match(identity_b))
    )

    baseline = DeterministicIdentitySelectionStrategy().select(request)
    alternate = _LastIdentitySelectionStrategy().select(request)

    assert baseline.selected_identity == identity_a
    assert alternate.selected_identity == identity_b


def test_selection_module_has_no_discovery_or_lifecycle_imports() -> None:
    source = Path("intergrax/agent_distribution/agent_selection.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden_prefixes = (
        "intergrax.agent_distribution.agent_discovery",
        "intergrax.agent_distribution.installation",
        "intergrax.agent_distribution.binding",
        "intergrax.agent_distribution.activation",
        "intergrax.runtime",
    )
    for name in imported:
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix)
