# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agent_distribution.capability_matching import (
    AgentCapabilityCandidate,
    AgentCapabilityDeclaration,
    AgentCapabilityRequirement,
    CapabilityId,
    CapabilityMatcher,
    CapabilityModelError,
    CapabilityRequirement,
    CapabilityRequirementError,
    build_agent_capability_candidate,
    build_agent_capability_requirement,
)
from intergrax.agent_distribution.catalog import (
    AgentDiscoveryCandidateIdentity,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.identity import AgentPackageCandidate


def _identity(
    source_id: str,
    package_id: str,
    *,
    kind: CatalogProviderKind = CatalogProviderKind.LOCAL_DEVELOPER,
    version: str = "1.0.0",
) -> AgentDiscoveryCandidateIdentity:
    return AgentDiscoveryCandidateIdentity(
        source=CatalogSourceIdentity(
            catalog_source_id=source_id,
            provider_kind=kind,
        ),
        package=AgentPackageCandidate(
            distribution_package_id=package_id,
            package_version=version,
        ),
    )


def _req(
    *,
    required: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
) -> AgentCapabilityRequirement:
    return build_agent_capability_requirement(required=required, optional=optional)


def _candidate(
    source_id: str,
    package_id: str,
    *,
    capability_ids: tuple[str, ...] = (),
    kind: CatalogProviderKind = CatalogProviderKind.LOCAL_DEVELOPER,
) -> AgentCapabilityCandidate:
    return build_agent_capability_candidate(
        identity=_identity(source_id, package_id, kind=kind),
        capability_ids=capability_ids,
    )


def test_exact_required_capability_match() -> None:
    matcher = CapabilityMatcher()
    result = matcher.match(
        requirement=_req(required=("document.search",)),
        candidate=_candidate(
            "source-a", "agent-a", capability_ids=("document.search",)
        ),
    )
    assert result.eligible is True
    assert tuple(item.value for item in result.matched_required) == ("document.search",)
    assert result.missing_required == ()
    assert result.matched_optional == ()


def test_missing_required_capability_rejects_candidate() -> None:
    matcher = CapabilityMatcher()
    result = matcher.match(
        requirement=_req(required=("document.search", "citation.produce")),
        candidate=_candidate(
            "source-a", "agent-a", capability_ids=("document.search",)
        ),
    )
    assert result.eligible is False
    assert tuple(item.value for item in result.missing_required) == (
        "citation.produce",
    )


def test_multiple_required_capabilities_all_must_match() -> None:
    matcher = CapabilityMatcher()
    result = matcher.match(
        requirement=_req(
            required=("document.search", "citation.produce", "streaming.output")
        ),
        candidate=_candidate(
            "source-a",
            "agent-a",
            capability_ids=("document.search", "citation.produce"),
        ),
    )
    assert result.eligible is False
    assert tuple(item.value for item in result.missing_required) == (
        "streaming.output",
    )


def test_optional_capability_missing_still_eligible() -> None:
    matcher = CapabilityMatcher()
    result = matcher.match(
        requirement=_req(
            required=("document.search",),
            optional=("streaming.output",),
        ),
        candidate=_candidate(
            "source-a", "agent-a", capability_ids=("document.search",)
        ),
    )
    assert result.eligible is True
    assert result.matched_optional == ()


def test_optional_capability_present_records_match_evidence() -> None:
    matcher = CapabilityMatcher()
    result = matcher.match(
        requirement=_req(
            required=("document.search",),
            optional=("streaming.output",),
        ),
        candidate=_candidate(
            "source-a",
            "agent-a",
            capability_ids=("document.search", "streaming.output"),
        ),
    )
    assert result.eligible is True
    assert tuple(item.value for item in result.matched_optional) == (
        "streaming.output",
    )


def test_superset_candidate_is_eligible() -> None:
    matcher = CapabilityMatcher()
    result = matcher.match(
        requirement=_req(required=("document.search",)),
        candidate=_candidate(
            "source-a",
            "agent-a",
            capability_ids=(
                "document.search",
                "document.extract.structured",
                "citation.produce",
            ),
        ),
    )
    assert result.eligible is True


def test_unrelated_extra_capabilities_are_harmless() -> None:
    matcher = CapabilityMatcher()
    result = matcher.match(
        requirement=_req(required=("document.search",)),
        candidate=_candidate(
            "source-a",
            "agent-a",
            capability_ids=("document.search", "browser.navigate"),
        ),
    )
    assert result.eligible is True
    assert tuple(item.value for item in result.matched_required) == ("document.search",)


def test_empty_candidate_capabilities_rejects_when_required_present() -> None:
    matcher = CapabilityMatcher()
    result = matcher.match(
        requirement=_req(required=("document.search",)),
        candidate=_candidate("source-a", "agent-a"),
    )
    assert result.eligible is False
    assert tuple(item.value for item in result.missing_required) == ("document.search",)


def test_empty_requirement_is_invalid() -> None:
    with pytest.raises(CapabilityRequirementError, match="at least one capability"):
        AgentCapabilityRequirement(requirements=())


def test_duplicate_capability_declarations_rejected() -> None:
    with pytest.raises(CapabilityModelError, match="duplicate capability declaration"):
        AgentCapabilityCandidate(
            identity=_identity("source-a", "agent-a"),
            capabilities=(
                AgentCapabilityDeclaration(
                    capability_id=CapabilityId(value="document.search"),
                ),
                AgentCapabilityDeclaration(
                    capability_id=CapabilityId(value="document.search"),
                ),
            ),
        )


def test_duplicate_requirement_capability_ids_rejected() -> None:
    with pytest.raises(
        CapabilityRequirementError, match="duplicate capability requirement"
    ):
        AgentCapabilityRequirement(
            requirements=(
                CapabilityRequirement(
                    capability_id=CapabilityId(value="document.search"),
                    required=True,
                ),
                CapabilityRequirement(
                    capability_id=CapabilityId(value="document.search"),
                    required=False,
                ),
            ),
        )


def test_find_matches_is_deterministic_across_input_order() -> None:
    matcher = CapabilityMatcher()
    requirement = _req(required=("document.search",))
    first = (
        _candidate("source-z", "z-agent", capability_ids=("document.search",)),
        _candidate("source-a", "a-agent", capability_ids=("document.search",)),
        _candidate("source-m", "m-agent"),
    )
    second = tuple(reversed(first))
    assert matcher.find_matches(
        requirement=requirement, candidates=first
    ) == matcher.find_matches(
        requirement=requirement,
        candidates=second,
    )


def test_find_eligible_orders_by_identity_sort_key() -> None:
    matcher = CapabilityMatcher()
    requirement = _req(required=("document.search",))
    candidates = (
        _candidate("source-z", "z-agent", capability_ids=("document.search",)),
        _candidate("source-a", "a-agent", capability_ids=("document.search",)),
        _candidate("source-m", "m-agent"),
    )
    eligible = matcher.find_eligible(requirement=requirement, candidates=candidates)
    assert [item.identity.source.catalog_source_id for item in eligible] == [
        "source-a",
        "source-z",
    ]


def test_match_evidence_is_exact_and_auditable() -> None:
    matcher = CapabilityMatcher()
    identity = _identity("source-a", "agent-a")
    result = matcher.match(
        requirement=_req(
            required=("document.search", "citation.produce"),
            optional=("streaming.output",),
        ),
        candidate=_candidate(
            "source-a", "agent-a", capability_ids=("document.search",)
        ),
    )
    assert result.identity == identity
    assert result.eligible is False
    assert tuple(item.value for item in result.matched_required) == ("document.search",)
    assert tuple(item.value for item in result.missing_required) == (
        "citation.produce",
    )
    assert result.matched_optional == ()
    assert result.unsupported_constraints == ()


def test_source_independence_for_identical_capability_sets() -> None:
    matcher = CapabilityMatcher()
    requirement = _req(required=("document.search",))
    github_candidate = _candidate(
        "catalog:github",
        "same-capability-agent",
        capability_ids=("document.search",),
        kind=CatalogProviderKind.GOVERNED_THIRD_PARTY,
    )
    enterprise_candidate = _candidate(
        "catalog:enterprise",
        "same-capability-agent",
        capability_ids=("document.search",),
        kind=CatalogProviderKind.ENTERPRISE_PRIVATE,
    )
    github_result = matcher.match(requirement=requirement, candidate=github_candidate)
    enterprise_result = matcher.match(
        requirement=requirement, candidate=enterprise_candidate
    )
    assert github_result.eligible == enterprise_result.eligible
    assert github_result.matched_required == enterprise_result.matched_required
    assert github_result.missing_required == enterprise_result.missing_required
    assert github_result.identity != enterprise_result.identity


def test_matcher_has_no_lifecycle_side_effects() -> None:
    matcher = CapabilityMatcher()
    requirement = _req(required=("document.search",))
    candidate = _candidate("source-a", "agent-a", capability_ids=("document.search",))
    before_requirement = requirement.model_dump()
    before_candidate = candidate.model_dump()
    matcher.find_matches(requirement=requirement, candidates=(candidate,))
    assert requirement.model_dump() == before_requirement
    assert candidate.model_dump() == before_candidate


def test_match_result_exposes_candidate_identity_for_future_trust_filter() -> None:
    matcher = CapabilityMatcher()
    identity = _identity("source-a", "agent-a")
    result = matcher.match(
        requirement=_req(required=("document.search",)),
        candidate=_candidate(
            "source-a", "agent-a", capability_ids=("document.search",)
        ),
    )
    assert result.identity == identity
    assert result.eligible is True


def test_capability_models_are_immutable() -> None:
    requirement = _req(required=("document.search",))
    candidate = _candidate("source-a", "agent-a", capability_ids=("document.search",))
    with pytest.raises(Exception):
        requirement.requirements = ()  # type: ignore[misc]
    with pytest.raises(Exception):
        candidate.identity = _identity("other", "agent")  # type: ignore[misc]


def test_capability_id_normalizes_whitespace() -> None:
    capability = CapabilityId(value="  document.search  ")
    assert capability.value == "document.search"


def test_capability_id_rejects_empty() -> None:
    with pytest.raises(ValueError):
        CapabilityId(value="   ")


def test_matcher_module_has_no_backend_imports() -> None:
    import ast
    from pathlib import Path

    import intergrax.agent_distribution.capability_matching as module

    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    forbidden_prefixes = (
        "intergrax.agent_distribution.in_memory_stores",
        "intergrax.agent_distribution.installation",
        "intergrax.agent_distribution.binding",
        "intergrax.runtime",
        "applications",
        "agents",
    )
    violations = sorted(
        module
        for module in imported_modules
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    )
    assert not violations, f"unexpected backend imports: {violations}"
