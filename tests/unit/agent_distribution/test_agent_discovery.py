# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryCandidate,
    AgentDiscoveryIdentityConflict,
    AgentDiscoveryRequest,
    AgentDiscoveryScope,
    AgentDiscoveryStrategy,
    AgentDiscoveryStrategyId,
    StaticAgentDiscoveryStrategy,
    build_agent_discovery_result,
    normalize_discovery_candidates,
    project_package_contract_capabilities,
    project_to_capability_candidate,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
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


def _request(
    *,
    required: tuple[str, ...] = ("document.search",),
    optional: tuple[str, ...] = (),
    scope: AgentDiscoveryScope | None = None,
) -> AgentDiscoveryRequest:
    return AgentDiscoveryRequest(
        requirement=build_agent_capability_requirement(
            required=required,
            optional=optional,
        ),
        scope=scope or AgentDiscoveryScope(),
    )


class _FakeDiscoveryStrategy:
    """Structural Protocol implementation without inheritance."""

    def __init__(
        self,
        *,
        strategy_id: AgentDiscoveryStrategyId,
        candidates: tuple[AgentDiscoveryCandidate, ...],
    ) -> None:
        self._strategy_id = strategy_id
        self._candidates = candidates

    @property
    def strategy_id(self) -> AgentDiscoveryStrategyId:
        return self._strategy_id

    def discover(self, request: AgentDiscoveryRequest):
        return build_agent_discovery_result(
            strategy_id=self._strategy_id,
            request=request,
            candidates=self._candidates,
        )


def test_protocol_boundary_accepts_structural_fake() -> None:
    strategy: AgentDiscoveryStrategy = _FakeDiscoveryStrategy(
        strategy_id=AgentDiscoveryStrategyId(value="fake.static"),
        candidates=(
            _discovery_candidate(
                "source-a", "agent-x", capability_ids=("document.search",)
            ),
        ),
    )
    result = strategy.discover(_request())
    assert result.strategy_id.value == "fake.static"
    assert len(result.candidates) == 1


def test_source_independence_for_same_display_package_id() -> None:
    source_a = _discovery_candidate(
        "enterprise-a",
        "same-agent",
        capability_ids=("document.search",),
        kind=CatalogProviderKind.ENTERPRISE_PRIVATE,
    )
    source_b = _discovery_candidate(
        "enterprise-b",
        "same-agent",
        capability_ids=("document.search",),
        kind=CatalogProviderKind.GOVERNED_THIRD_PARTY,
    )
    assert source_a.identity != source_b.identity
    matcher = CapabilityMatcher()
    requirement = build_agent_capability_requirement(required=("document.search",))
    result_a = matcher.match(
        requirement=requirement,
        candidate=project_to_capability_candidate(source_a),
    )
    result_b = matcher.match(
        requirement=requirement,
        candidate=project_to_capability_candidate(source_b),
    )
    assert result_a.identity == source_a.identity
    assert result_b.identity == source_b.identity
    assert result_a.identity != result_b.identity


def test_lossless_identity_through_matcher() -> None:
    discovered = _discovery_candidate(
        "catalog.local",
        "agent-a",
        capability_ids=("document.search", "streaming.output"),
    )
    matcher = CapabilityMatcher()
    requirement = build_agent_capability_requirement(
        required=("document.search",),
        optional=("streaming.output",),
    )
    match = matcher.match(
        requirement=requirement,
        candidate=project_to_capability_candidate(discovered),
    )
    assert match.identity == discovered.identity


def test_discovery_output_is_deterministic() -> None:
    candidates = (
        _discovery_candidate(
            "z-source", "z-agent", capability_ids=("document.search",)
        ),
        _discovery_candidate(
            "a-source", "a-agent", capability_ids=("document.search",)
        ),
        _discovery_candidate("m-source", "m-agent"),
    )
    strategy = StaticAgentDiscoveryStrategy(
        strategy_id=AgentDiscoveryStrategyId(value="static.reference"),
        candidates=candidates,
    )
    request = _request()
    first = strategy.discover(request)
    second = strategy.discover(request)
    assert [item.identity.sort_key for item in first.candidates] == [
        item.identity.sort_key for item in second.candidates
    ]
    assert first.candidates[0].identity.source.catalog_source_id == "a-source"


def test_empty_discovery_result_is_valid() -> None:
    strategy = StaticAgentDiscoveryStrategy(
        strategy_id=AgentDiscoveryStrategyId(value="static.empty"),
        candidates=(),
    )
    result = strategy.discover(_request())
    assert result.candidates == ()


def test_duplicate_identity_fails_closed() -> None:
    duplicate = _discovery_candidate(
        "source-a", "agent-a", capability_ids=("document.search",)
    )
    with pytest.raises(AgentDiscoveryIdentityConflict, match="duplicate canonical"):
        normalize_discovery_candidates((duplicate, duplicate))


def test_identity_conflict_fails_closed() -> None:
    first = _discovery_candidate(
        "source-a", "agent-a", capability_ids=("document.search",)
    )
    second = AgentDiscoveryCandidate(
        identity=first.identity,
        capabilities=project_package_contract_capabilities(
            AgentPackageContractDeclaration(
                contract_id="contract.v1",
                contract_version="1",
                capabilities=("citation.produce",),
            ),
        ),
    )
    with pytest.raises(
        AgentDiscoveryIdentityConflict, match="conflicting discovery facts"
    ):
        normalize_discovery_candidates((first, second))


def test_discovery_matching_pipeline() -> None:
    candidates = (
        _discovery_candidate(
            "source-a",
            "agent-x",
            capability_ids=("document.search",),
        ),
        _discovery_candidate(
            "source-b",
            "agent-x",
            capability_ids=(),
        ),
        _discovery_candidate(
            "source-c",
            "agent-y",
            capability_ids=("document.search", "streaming.output"),
        ),
    )
    strategy = StaticAgentDiscoveryStrategy(
        strategy_id=AgentDiscoveryStrategyId(value="static.pipeline"),
        candidates=candidates,
    )
    discovery = strategy.discover(
        _request(required=("document.search",), optional=("streaming.output",)),
    )
    matcher = CapabilityMatcher()
    matches = matcher.find_matches(
        requirement=discovery.request.requirement,
        candidates=tuple(
            project_to_capability_candidate(candidate)
            for candidate in discovery.candidates
        ),
    )
    eligible = [item for item in matches if item.eligible]
    assert len(eligible) == 2
    assert {item.identity.source.catalog_source_id for item in eligible} == {
        "source-a",
        "source-c",
    }
    assert all(
        match.identity == candidate.identity
        for match, candidate in zip(matches, discovery.candidates, strict=True)
    )


def test_project_package_contract_capabilities_fail_closed_on_empty_string() -> None:
    with pytest.raises(ValueError):
        project_package_contract_capabilities(
            AgentPackageContractDeclaration(
                contract_id="contract.v1",
                contract_version="1",
                capabilities=("  ",),
            ),
        )


def test_discovery_module_has_no_lifecycle_imports() -> None:
    import intergrax.agent_distribution.agent_discovery as module

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
