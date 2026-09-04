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
    DeterministicIdentitySelectionStrategy,
    SelectionOutcome,
    build_agent_selection_request,
    require_selected_identity,
)
from intergrax.agent_distribution.capability_matching import CapabilityMatcher
from intergrax.agent_distribution.catalog import (
    AgentDiscoveryCandidateIdentity,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.federated_discovery import (
    FederatedAgentDiscoveryStrategy,
)
from intergrax.agent_distribution.identity import AgentPackageCandidate
from intergrax.agent_distribution.task_capability_resolution import (
    CapabilityRequirementKind,
    DETERMINISTIC_MAPPING_RESOLVER_ID,
    DeterministicTaskCapabilityResolver,
    TaskCapabilityEvidenceRationaleCategory,
    TaskCapabilityResolutionConflict,
    TaskCapabilityResolutionContractError,
    TaskCapabilityResolutionNoMatch,
    TaskCapabilityResolutionRequest,
    TaskCapabilityResolutionResult,
    TaskCapabilityResolverId,
    build_deterministic_task_capability_resolver,
    build_task_capability_resolution_request,
    build_task_capability_rule,
)


def _source(source_id: str, kind: CatalogProviderKind) -> CatalogSourceIdentity:
    return CatalogSourceIdentity(
        catalog_source_id=source_id,
        provider_kind=kind,
    )


def _identity(
    source_id: str,
    package_id: str,
    *,
    kind: CatalogProviderKind = CatalogProviderKind.LOCAL_DEVELOPER,
) -> AgentDiscoveryCandidateIdentity:
    return AgentDiscoveryCandidateIdentity(
        source=_source(source_id, kind),
        package=AgentPackageCandidate(
            distribution_package_id=package_id,
            package_version="1.0.0",
        ),
    )


def _discovery_candidate(
    source_id: str,
    package_id: str,
    *,
    capability_ids: tuple[str, ...] = (),
) -> AgentDiscoveryCandidate:
    return AgentDiscoveryCandidate(
        identity=_identity(source_id, package_id),
        capabilities=project_package_contract_capabilities(
            AgentPackageContractDeclaration(
                contract_id="contract.v1",
                contract_version="1",
                capabilities=capability_ids,
            ),
        ),
        catalog_entry_id=f"{source_id}:{package_id}",
    )


def _baseline_resolver() -> DeterministicTaskCapabilityResolver:
    return build_deterministic_task_capability_resolver(
        rules=(
            build_task_capability_rule(
                rule_id="rule.document.legal_compare.v1",
                task_kind="document.legal_compare",
                required=(
                    "document.read",
                    "legal.analysis",
                    "document.compare",
                ),
                optional=("citation.generate",),
                rule_version="1",
            ),
            build_task_capability_rule(
                rule_id="rule.document.ocr.v1",
                task_kind="document.ocr",
                required=("document.ocr",),
                rule_version="1",
            ),
        ),
    )


class _AlternativeTaskCapabilityResolver:
    """Structural Protocol implementation without inheritance."""

    @property
    def resolver_id(self) -> TaskCapabilityResolverId:
        return TaskCapabilityResolverId(value="test.alternative")

    def resolve(
        self,
        request: TaskCapabilityResolutionRequest,
    ) -> TaskCapabilityResolutionResult:
        from intergrax.agent_distribution.capability_matching import (
            build_agent_capability_requirement,
        )
        from intergrax.agent_distribution.task_capability_resolution import (
            TaskCapabilityEvidence,
            TaskCapabilityRuleId,
            TaskCapabilityRuleVersion,
        )

        requirement = build_agent_capability_requirement(
            required=("document.ocr", "document.read"),
        )
        return TaskCapabilityResolutionResult(
            resolver_id=self.resolver_id,
            request=request,
            capability_requirement=requirement,
            evidence=(
                TaskCapabilityEvidence(
                    capability_id=requirement.requirements[0].capability_id,
                    requirement_kind=CapabilityRequirementKind.REQUIRED,
                    rule_id=TaskCapabilityRuleId(value="test.alternative.rule"),
                    rule_version=TaskCapabilityRuleVersion(value="1"),
                    rationale_category=(
                        TaskCapabilityEvidenceRationaleCategory.DETERMINISTIC_RULE_MATCH
                    ),
                ),
            ),
        )


def test_exact_resolution_document_legal_compare() -> None:
    resolver = _baseline_resolver()
    request = build_task_capability_resolution_request(
        task_kind="document.legal_compare",
        task_text="Compare these contracts and identify legal risks.",
    )
    result = resolver.resolve(request)

    assert result.resolver_id == DETERMINISTIC_MAPPING_RESOLVER_ID
    assert result.request == request
    assert tuple(
        item.capability_id.value
        for item in result.capability_requirement.requirements
        if item.required
    ) == ("document.compare", "document.read", "legal.analysis")
    assert tuple(
        item.capability_id.value
        for item in result.capability_requirement.requirements
        if not item.required
    ) == ("citation.generate",)
    assert tuple(item.capability_id.value for item in result.evidence) == (
        "document.compare",
        "document.read",
        "legal.analysis",
        "citation.generate",
    )
    assert all(
        item.rationale_category
        is TaskCapabilityEvidenceRationaleCategory.DETERMINISTIC_RULE_MATCH
        for item in result.evidence
    )
    assert all(
        item.rule_id.value == "rule.document.legal_compare.v1"
        for item in result.evidence
    )


def test_unknown_task_fails_closed() -> None:
    resolver = _baseline_resolver()
    with pytest.raises(
        TaskCapabilityResolutionNoMatch, match="no task capability rule"
    ):
        resolver.resolve(
            build_task_capability_resolution_request(task_kind="unknown.task"),
        )


def test_multiple_matching_rules_fail_closed() -> None:
    resolver = build_deterministic_task_capability_resolver(
        rules=(
            build_task_capability_rule(
                rule_id="rule.a",
                task_kind="document.ocr",
                required=("document.ocr",),
            ),
            build_task_capability_rule(
                rule_id="rule.b",
                task_kind="document.ocr",
                required=("document.read",),
            ),
        ),
    )
    with pytest.raises(
        TaskCapabilityResolutionConflict, match="multiple task capability rules"
    ):
        resolver.resolve(
            build_task_capability_resolution_request(task_kind="document.ocr")
        )


def test_duplicate_rule_ids_fail_closed() -> None:
    with pytest.raises(
        TaskCapabilityResolutionContractError, match="duplicate task capability rule_id"
    ):
        build_deterministic_task_capability_resolver(
            rules=(
                build_task_capability_rule(
                    rule_id="rule.duplicate",
                    task_kind="document.ocr",
                    required=("document.ocr",),
                ),
                build_task_capability_rule(
                    rule_id="rule.duplicate",
                    task_kind="document.read",
                    required=("document.read",),
                ),
            ),
        )


def test_duplicate_required_capability_in_rule_fails_closed() -> None:
    with pytest.raises(
        TaskCapabilityResolutionContractError, match="duplicate capability"
    ):
        build_task_capability_rule(
            rule_id="rule.duplicate.cap",
            task_kind="document.ocr",
            required=("document.ocr", "document.ocr"),
        )


def test_required_optional_conflict_in_rule_fails_closed() -> None:
    with pytest.raises(
        TaskCapabilityResolutionContractError,
        match="both required and optional",
    ):
        build_task_capability_rule(
            rule_id="rule.conflict",
            task_kind="document.ocr",
            required=("document.ocr",),
            optional=("document.ocr",),
        )


def test_invalid_capability_id_fails_closed() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        build_task_capability_rule(
            rule_id="rule.invalid",
            task_kind="document.ocr",
            required=("   ",),
        )


def test_determinism_same_input_same_output() -> None:
    resolver = _baseline_resolver()
    request = build_task_capability_resolution_request(
        task_kind="document.legal_compare"
    )
    first = resolver.resolve(request)
    second = resolver.resolve(request)
    assert first == second


def test_rule_input_order_does_not_change_output() -> None:
    rules = (
        build_task_capability_rule(
            rule_id="rule.document.ocr.v1",
            task_kind="document.ocr",
            required=("document.ocr",),
        ),
        build_task_capability_rule(
            rule_id="rule.document.legal_compare.v1",
            task_kind="document.legal_compare",
            required=("document.read", "legal.analysis", "document.compare"),
            optional=("citation.generate",),
        ),
    )
    shuffled = build_deterministic_task_capability_resolver(rules=rules[::-1])
    ordered = build_deterministic_task_capability_resolver(rules=rules)
    request = build_task_capability_resolution_request(
        task_kind="document.legal_compare"
    )
    assert shuffled.resolve(request) == ordered.resolve(request)


def test_alternative_resolver_is_structurally_pluggable() -> None:
    resolver = _AlternativeTaskCapabilityResolver()
    request = build_task_capability_resolution_request(task_kind="document.ocr")
    result = resolver.resolve(request)
    assert result.resolver_id.value == "test.alternative"
    assert tuple(
        item.capability_id.value for item in result.capability_requirement.requirements
    ) == ("document.ocr", "document.read")


def _run_pipeline(
    *,
    resolver: DeterministicTaskCapabilityResolver,
    task_kind: str,
    candidates: tuple[AgentDiscoveryCandidate, ...],
) -> AgentDiscoveryCandidateIdentity:
    resolution = resolver.resolve(
        build_task_capability_resolution_request(task_kind=task_kind),
    )
    discovery = FederatedAgentDiscoveryStrategy(
        strategies=(
            StaticAgentDiscoveryStrategy(
                strategy_id=AgentDiscoveryStrategyId(value="static.test"),
                candidates=candidates,
            ),
        ),
    ).discover(
        AgentDiscoveryRequest(requirement=resolution.capability_requirement),
    )
    matcher = CapabilityMatcher()
    matches = matcher.find_matches(
        requirement=resolution.capability_requirement,
        candidates=tuple(
            project_to_capability_candidate(candidate)
            for candidate in discovery.candidates
        ),
    )
    eligible = tuple(item for item in matches if item.eligible)
    decision = DeterministicIdentitySelectionStrategy().select(
        build_agent_selection_request(
            requirement=resolution.capability_requirement,
            eligible_matches=eligible,
        ),
    )
    assert decision.outcome is SelectionOutcome.SELECTED
    return require_selected_identity(decision)


def test_user_task_pipeline_document_legal_compare() -> None:
    selected = _run_pipeline(
        resolver=_baseline_resolver(),
        task_kind="document.legal_compare",
        candidates=(
            _discovery_candidate(
                "source-a",
                "legal-agent",
                capability_ids=(
                    "document.read",
                    "legal.analysis",
                    "document.compare",
                    "citation.generate",
                ),
            ),
            _discovery_candidate(
                "source-b",
                "ocr-only",
                capability_ids=("document.ocr",),
            ),
        ),
    )
    assert selected.package.distribution_package_id == "legal-agent"


def test_delegated_subtask_pipeline_document_ocr() -> None:
    selected = _run_pipeline(
        resolver=_baseline_resolver(),
        task_kind="document.ocr",
        candidates=(
            _discovery_candidate(
                "source-a",
                "ocr-agent",
                capability_ids=("document.ocr",),
            ),
            _discovery_candidate(
                "source-b",
                "legal-agent",
                capability_ids=("legal.analysis", "document.read"),
            ),
        ),
    )
    assert selected.package.distribution_package_id == "ocr-agent"


def test_task_capability_module_has_no_downstream_coupling() -> None:
    import intergrax.agent_distribution.task_capability_resolution as module

    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    forbidden_prefixes = (
        "intergrax.agent_distribution.agent_discovery",
        "intergrax.agent_distribution.federated_discovery",
        "intergrax.agent_distribution.agent_selection",
        "intergrax.agent_distribution.installation_service",
        "intergrax.agent_distribution.binding_service",
        "intergrax.agent_distribution.admin_service",
        "intergrax.agent_distribution.activation",
        "intergrax.runtime",
        "intergrax.harness",
        "intergrax.nexus",
        "openai",
        "anthropic",
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
    assert not violations, f"unexpected downstream imports: {violations}"
