# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryCandidate,
    AgentDiscoveryIdentityConflict,
    AgentDiscoveryRequest,
    AgentDiscoveryResult,
    AgentDiscoveryStrategy,
    AgentDiscoveryStrategyId,
    StaticAgentDiscoveryStrategy,
    build_agent_discovery_result,
    project_package_contract_capabilities,
    project_to_capability_candidate,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
)
from intergrax.agent_distribution.agent_selection import (
    DeterministicIdentitySelectionStrategy,
    SelectionOutcome,
    build_agent_selection_request,
    require_selected_identity,
)
from intergrax.agent_distribution.capability_matching import (
    CapabilityMatcher,
    build_agent_capability_requirement,
)
from intergrax.agent_distribution.catalog import (
    AgentDiscoveryCandidateIdentity,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.federated_discovery import (
    FEDERATED_DISCOVERY_STRATEGY_ID,
    FederatedAgentDiscoveryResult,
    FederatedAgentDiscoveryStrategy,
    FederatedDiscoveryChildResultError,
    FederatedDiscoveryConfigurationError,
    merge_federated_candidates,
    provenance_for,
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


def _discovery_candidate(
    source_id: str,
    package_id: str,
    *,
    capability_ids: tuple[str, ...] = (),
    kind: CatalogProviderKind = CatalogProviderKind.LOCAL_DEVELOPER,
    catalog_entry_id: str | None = None,
    artifact_locator: str | None = None,
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
        catalog_entry_id=catalog_entry_id or f"{source_id}:{package_id}",
        artifact_locator=artifact_locator,
    )


def _request(
    *,
    required: tuple[str, ...] = ("document.search",),
    optional: tuple[str, ...] = (),
) -> AgentDiscoveryRequest:
    return AgentDiscoveryRequest(
        requirement=build_agent_capability_requirement(
            required=required,
            optional=optional,
        ),
    )


def _strategy_id(value: str) -> AgentDiscoveryStrategyId:
    return AgentDiscoveryStrategyId(value=value)


class _FakeDiscoveryStrategy:
    """Structural Protocol implementation without inheritance."""

    def __init__(
        self,
        *,
        strategy_id: AgentDiscoveryStrategyId,
        candidates: tuple[AgentDiscoveryCandidate, ...],
        request_override: AgentDiscoveryRequest | None = None,
        strategy_id_override: AgentDiscoveryStrategyId | None = None,
    ) -> None:
        self._strategy_id = strategy_id
        self._candidates = candidates
        self._request_override = request_override
        self._strategy_id_override = strategy_id_override

    @property
    def strategy_id(self) -> AgentDiscoveryStrategyId:
        return self._strategy_id

    def discover(self, request: AgentDiscoveryRequest) -> AgentDiscoveryResult:
        return build_agent_discovery_result(
            strategy_id=self._strategy_id_override or self._strategy_id,
            request=self._request_override or request,
            candidates=self._candidates,
        )


def _static(
    strategy_id: str,
    candidates: tuple[AgentDiscoveryCandidate, ...],
) -> StaticAgentDiscoveryStrategy:
    return StaticAgentDiscoveryStrategy(
        strategy_id=_strategy_id(strategy_id),
        candidates=candidates,
    )


def test_zero_child_strategies_rejected() -> None:
    with pytest.raises(
        FederatedDiscoveryConfigurationError,
        match="at least one child strategy",
    ):
        FederatedAgentDiscoveryStrategy(strategies=())


def test_duplicate_child_strategy_ids_rejected() -> None:
    child = _static(
        "enterprise.registry",
        (_discovery_candidate("source-a", "agent-a"),),
    )
    with pytest.raises(
        FederatedDiscoveryConfigurationError,
        match="duplicate child strategy_id",
    ):
        FederatedAgentDiscoveryStrategy(strategies=(child, child))


def test_protocol_boundary_accepts_structural_fake() -> None:
    strategy: AgentDiscoveryStrategy = FederatedAgentDiscoveryStrategy(
        strategies=(
            _FakeDiscoveryStrategy(
                strategy_id=_strategy_id("fake.a"),
                candidates=(
                    _discovery_candidate(
                        "source-a",
                        "agent-a",
                        capability_ids=("document.search",),
                    ),
                ),
            ),
        ),
    )
    result = strategy.discover(_request())
    assert isinstance(result, FederatedAgentDiscoveryResult)
    assert result.strategy_id == FEDERATED_DISCOVERY_STRATEGY_ID


def test_one_child_strategy_equivalent_candidate_universe() -> None:
    candidate = _discovery_candidate(
        "source-a",
        "agent-a",
        capability_ids=("document.search",),
    )
    child = _static("installed.local", (candidate,))
    federated = FederatedAgentDiscoveryStrategy(strategies=(child,))
    request = _request()
    child_result = child.discover(request)
    federated_result = federated.discover(request)
    assert federated_result.candidates == child_result.candidates
    assert federated_result.strategy_id == FEDERATED_DISCOVERY_STRATEGY_ID
    evidence = provenance_for(federated_result, candidate.identity)
    assert evidence is not None
    assert evidence.discovering_strategy_ids == (_strategy_id("installed.local"),)


def test_empty_child_results_valid() -> None:
    federated = FederatedAgentDiscoveryStrategy(
        strategies=(
            _static("strategy.a", ()),
            _static("strategy.b", ()),
        ),
    )
    result = federated.discover(_request())
    assert result.candidates == ()
    assert result.candidate_evidence == ()


def test_multi_source_union_deterministic() -> None:
    candidate_a = _discovery_candidate("source-a", "agent-a")
    candidate_b = _discovery_candidate("source-b", "agent-b")
    candidate_c = _discovery_candidate("source-c", "agent-c")
    federated = FederatedAgentDiscoveryStrategy(
        strategies=(
            _static("strategy.c", (candidate_c,)),
            _static("strategy.a", (candidate_a,)),
            _static("strategy.b", (candidate_b,)),
        ),
    )
    result = federated.discover(_request())
    assert [item.identity.source.catalog_source_id for item in result.candidates] == [
        "source-a",
        "source-b",
        "source-c",
    ]
    assert result.invoked_child_strategy_ids == (
        _strategy_id("strategy.a"),
        _strategy_id("strategy.b"),
        _strategy_id("strategy.c"),
    )


def test_same_package_id_different_sources_remain_distinct() -> None:
    source_a = _discovery_candidate(
        "source-a",
        "package-x",
        kind=CatalogProviderKind.ENTERPRISE_PRIVATE,
    )
    source_b = _discovery_candidate(
        "source-b",
        "package-x",
        kind=CatalogProviderKind.GOVERNED_THIRD_PARTY,
    )
    federated = FederatedAgentDiscoveryStrategy(
        strategies=(
            _static("strategy.a", (source_a,)),
            _static("strategy.b", (source_b,)),
        ),
    )
    result = federated.discover(_request())
    assert len(result.candidates) == 2
    assert result.candidates[0].identity != result.candidates[1].identity


def test_exact_same_candidate_discovered_twice_deduplicates_with_provenance() -> None:
    shared = _discovery_candidate(
        "source-a",
        "agent-x",
        capability_ids=("document.search",),
        artifact_locator="artifact://shared",
    )
    federated = FederatedAgentDiscoveryStrategy(
        strategies=(
            _static("strategy.a", (shared,)),
            _static("strategy.b", (shared,)),
        ),
    )
    result = federated.discover(_request())
    assert len(result.candidates) == 1
    evidence = provenance_for(result, shared.identity)
    assert evidence is not None
    assert evidence.discovering_strategy_ids == (
        _strategy_id("strategy.a"),
        _strategy_id("strategy.b"),
    )


def test_conflicting_same_identity_fails_closed() -> None:
    identity = _identity("source-a", "agent-x")
    first = AgentDiscoveryCandidate(
        identity=identity,
        capabilities=project_package_contract_capabilities(
            AgentPackageContractDeclaration(
                contract_id="contract.v1",
                contract_version="1",
                capabilities=("document.search",),
            ),
        ),
        catalog_entry_id="source-a:agent-x",
    )
    second = AgentDiscoveryCandidate(
        identity=identity,
        capabilities=project_package_contract_capabilities(
            AgentPackageContractDeclaration(
                contract_id="contract.v1",
                contract_version="1",
                capabilities=("citation.produce",),
            ),
        ),
        catalog_entry_id="source-a:agent-x",
    )
    federated = FederatedAgentDiscoveryStrategy(
        strategies=(
            _static("strategy.a", (first,)),
            _static("strategy.b", (second,)),
        ),
    )
    with pytest.raises(
        AgentDiscoveryIdentityConflict,
        match="conflicting discovery facts",
    ):
        federated.discover(_request())


def test_request_mismatch_fails_closed() -> None:
    other_request = _request(required=("citation.produce",))
    federated = FederatedAgentDiscoveryStrategy(
        strategies=(
            _FakeDiscoveryStrategy(
                strategy_id=_strategy_id("strategy.a"),
                candidates=(_discovery_candidate("source-a", "agent-a"),),
                request_override=other_request,
            ),
        ),
    )
    with pytest.raises(
        FederatedDiscoveryChildResultError,
        match="request does not match",
    ):
        federated.discover(_request())


def test_strategy_id_mismatch_fails_closed() -> None:
    federated = FederatedAgentDiscoveryStrategy(
        strategies=(
            _FakeDiscoveryStrategy(
                strategy_id=_strategy_id("enterprise.registry"),
                candidates=(_discovery_candidate("source-a", "agent-a"),),
                strategy_id_override=_strategy_id("marketplace.external"),
            ),
        ),
    )
    with pytest.raises(
        FederatedDiscoveryChildResultError,
        match="strategy_id does not match",
    ):
        federated.discover(_request())


def test_child_strategy_order_does_not_affect_result() -> None:
    candidate_a = _discovery_candidate("source-a", "agent-a")
    candidate_b = _discovery_candidate("source-b", "agent-b")
    strategies_abc = (
        _static("strategy.a", (candidate_a,)),
        _static("strategy.b", (candidate_b,)),
    )
    strategies_cba = (
        _static("strategy.b", (candidate_b,)),
        _static("strategy.a", (candidate_a,)),
    )
    first = FederatedAgentDiscoveryStrategy(strategies=strategies_abc).discover(
        _request(),
    )
    second = FederatedAgentDiscoveryStrategy(strategies=strategies_cba).discover(
        _request(),
    )
    assert first.candidates == second.candidates
    assert first.candidate_evidence == second.candidate_evidence
    assert first.invoked_child_strategy_ids == second.invoked_child_strategy_ids


def test_candidate_order_inside_child_does_not_affect_result() -> None:
    candidate_a = _discovery_candidate("z-source", "z-agent")
    candidate_b = _discovery_candidate("a-source", "a-agent")
    shuffled = _static("strategy.a", (candidate_a, candidate_b))
    ordered = _static("strategy.a", (candidate_b, candidate_a))
    first = FederatedAgentDiscoveryStrategy(strategies=(shuffled,)).discover(
        _request(),
    )
    second = FederatedAgentDiscoveryStrategy(strategies=(ordered,)).discover(
        _request(),
    )
    assert first.candidates == second.candidates


def test_federated_discovery_feeds_capability_matcher() -> None:
    candidates = (
        _discovery_candidate(
            "source-a",
            "agent-x",
            capability_ids=("document.search",),
        ),
        _discovery_candidate("source-b", "agent-y", capability_ids=()),
    )
    federated = FederatedAgentDiscoveryStrategy(
        strategies=(_static("strategy.a", candidates),),
    )
    discovery = federated.discover(_request(required=("document.search",)))
    matcher = CapabilityMatcher()
    matches = matcher.find_matches(
        requirement=discovery.request.requirement,
        candidates=tuple(
            project_to_capability_candidate(candidate)
            for candidate in discovery.candidates
        ),
    )
    eligible = [item for item in matches if item.eligible]
    assert len(eligible) == 1
    assert eligible[0].identity == discovery.candidates[0].identity


def test_federated_pipeline_e2e_matching_and_selection() -> None:
    candidate_a = _discovery_candidate(
        "source-a",
        "agent-w",
        capability_ids=("document.search",),
    )
    candidate_b = _discovery_candidate(
        "source-b",
        "agent-x",
        capability_ids=("document.search",),
    )
    candidate_c = _discovery_candidate(
        "source-c",
        "agent-y",
        capability_ids=("citation.produce",),
    )
    candidate_d = _discovery_candidate(
        "source-d",
        "agent-z",
        capability_ids=("streaming.output",),
    )
    federated = FederatedAgentDiscoveryStrategy(
        strategies=(
            _static("strategy.a", (candidate_a, candidate_c)),
            _static("strategy.b", (candidate_b,)),
            _static("strategy.c", (candidate_d,)),
        ),
    )
    request = _request(
        required=("document.search",),
        optional=("streaming.output",),
    )
    discovery = federated.discover(request)
    assert len(discovery.candidates) == 4

    matcher = CapabilityMatcher()
    matches = matcher.find_matches(
        requirement=discovery.request.requirement,
        candidates=tuple(
            project_to_capability_candidate(candidate)
            for candidate in discovery.candidates
        ),
    )
    eligible = tuple(item for item in matches if item.eligible)
    assert len(eligible) == 2

    selection = DeterministicIdentitySelectionStrategy()
    decision = selection.select(
        build_agent_selection_request(
            requirement=discovery.request.requirement,
            eligible_matches=eligible,
        ),
    )
    selected = require_selected_identity(decision)
    assert decision.outcome is SelectionOutcome.SELECTED
    assert selected == discovery.candidates[0].identity

    evidence = provenance_for(discovery, selected)
    assert evidence is not None
    assert evidence.discovering_strategy_ids == (_strategy_id("strategy.a"),)


def test_marketplace_source_collision_distinct() -> None:
    eu = _discovery_candidate(
        "enterprise-marketplace-eu",
        "legal-analyzer",
        kind=CatalogProviderKind.GOVERNED_THIRD_PARTY,
    )
    us = _discovery_candidate(
        "enterprise-marketplace-us",
        "legal-analyzer",
        kind=CatalogProviderKind.GOVERNED_THIRD_PARTY,
    )
    federated = FederatedAgentDiscoveryStrategy(
        strategies=(
            _static("marketplace.eu", (eu,)),
            _static("marketplace.us", (us,)),
        ),
    )
    result = federated.discover(_request())
    assert len(result.candidates) == 2
    assert {item.identity.source.catalog_source_id for item in result.candidates} == {
        "enterprise-marketplace-eu",
        "enterprise-marketplace-us",
    }


def test_merge_federated_candidates_pure_helper() -> None:
    candidate = _discovery_candidate("source-a", "agent-a")
    candidates, evidence = merge_federated_candidates(
        (
            (_strategy_id("strategy.b"), candidate),
            (_strategy_id("strategy.a"), candidate),
        ),
    )
    assert len(candidates) == 1
    assert evidence[0].discovering_strategy_ids == (
        _strategy_id("strategy.a"),
        _strategy_id("strategy.b"),
    )


def test_federated_module_has_no_lifecycle_imports() -> None:
    import intergrax.agent_distribution.federated_discovery as module

    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    forbidden_prefixes = (
        "intergrax.agent_distribution.installation_service",
        "intergrax.agent_distribution.binding_service",
        "intergrax.agent_distribution.admin_service",
        "intergrax.agent_distribution.activation",
        "intergrax.runtime",
        "applications",
        "agents",
    )
    violations = sorted(
        imported
        for imported in imported_modules
        if any(
            imported == prefix or imported.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    )
    assert not violations, f"unexpected lifecycle imports: {violations}"
