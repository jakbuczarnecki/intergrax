# © Artur Czarnecki. All rights reserved.

"""AW Stage 9 — Capability Catalog discovery bridge tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from intergrax.autonomous_work.capability_discovery_adapters import (
    IntegrationCatalogCapabilityDiscoveryAdapter,
)
from intergrax.autonomous_work.capability_acquisition_ports import (
    AllowAllAuthorityCompatibilityPort,
    NotConfiguredApprovedAlternateDiscovery,
    NotConfiguredConfigurationOpportunityDiscovery,
    StaticCodecraftProfileResolver,
    StaticWorkerCapabilityProfileResolver,
    UnavailableIntegrationCapabilityDiscovery,
    permissive_capability_policy,
)
from intergrax.autonomous_work.capability_acquisition_service import (
    WorkerCapabilityAcquisitionDecisionService,
)
from intergrax.autonomous_work.capability_catalog_discovery_adapters import (
    CapabilityCatalogDiscoveryDependencies,
    CapabilityCatalogToolDiscoveryAdapter,
    _tool_supports_required_operations,
    encode_source_qualified_capability_ref,
    identity_key_from_entry_identity,
    map_worker_capability_need_to_discovery_query,
)
from intergrax.capability_catalog import AvailabilityPreservingGovernanceEvaluator
from intergrax.capability_catalog.adapters.tool_governance import ToolPolicyGovernanceEvaluator
from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityDiscoveryDisposition,
    CapabilityOperationCoverage,
    WorkerAutonomyLevel,
    WorkerCapabilityCandidateKind,
    WorkerCapabilityAuthorityCompatibility,
)
from intergrax.contracts.capability_catalog.evidence import CapabilityDiscoveryAvailabilityEvidence
from intergrax.contracts.capability_catalog.governance import (
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityToolGovernanceEvidence,
)
from intergrax.contracts.capability_catalog.identity import (
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.scope import (
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
)
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from tests.unit.autonomous_work.catalog_discovery_test_support import (
    catalog_discovery_dependencies,
    catalog_snapshot_from_registries,
    catalog_tool_skill_adapters,
    host_availability_for_entries,
    skill_catalog_entry,
    tool_catalog_entry,
)
from tests.unit.autonomous_work.test_worker_capability_acquisition import (
    _OPERATION,
    _request,
    _service,
    _skill_registry,
    _tool_registry,
)

pytestmark = pytest.mark.unit

_CAPABILITY_PROFILE = __import__(
    "tests.unit.autonomous_work.test_worker_capability_acquisition",
    fromlist=["_CAPABILITY_PROFILE"],
)._CAPABILITY_PROFILE

_BUILTIN_TOOL_SOURCE = CapabilitySourceIdentity(
    source_id="tools.catalog.builtin",
    source_kind=CapabilitySourceKind.BUILTIN,
)
_PRIVATE_TOOL_SOURCE = CapabilitySourceIdentity(
    source_id="enterprise.private",
    source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
)


class _CountingCodecraftResolver:
    def __init__(self, *, allowed: bool) -> None:
        self.allowed = allowed
        self.calls = 0

    def is_candidate_consideration_allowed(self, profile_ref) -> bool:
        del profile_ref
        self.calls += 1
        return self.allowed


def _catalog_service(
    *,
    tool_registry: ToolRegistry | None = None,
    skill_registry: SkillRegistry | None = None,
    extra_entries: tuple = (),
    availability_evidence: CapabilityDiscoveryAvailabilityEvidence | None = None,
    host_tool_ids: tuple[str, ...] | None = None,
    host_skill_ids: tuple[str, ...] | None = None,
    codecraft_resolver: _CountingCodecraftResolver | None = None,
    authority=None,
) -> WorkerCapabilityAcquisitionDecisionService:
    resolved_tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
    resolved_skill_registry = skill_registry if skill_registry is not None else SkillRegistry()
    tool_discovery, skill_discovery = catalog_tool_skill_adapters(
        tool_registry=resolved_tool_registry,
        skill_registry=resolved_skill_registry,
        extra_entries=extra_entries,
        availability_evidence=availability_evidence,
        host_tool_ids=host_tool_ids,
        host_skill_ids=host_skill_ids,
    )
    return WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_CAPABILITY_PROFILE),
        ),
        tool_discovery=tool_discovery,
        skill_discovery=skill_discovery,
        integration_discovery=IntegrationCatalogCapabilityDiscoveryAdapter(),
        approved_alternate_discovery=NotConfiguredApprovedAlternateDiscovery(),
        configuration_discovery=NotConfiguredConfigurationOpportunityDiscovery(),
        authority_compatibility=authority or AllowAllAuthorityCompatibilityPort(),
        codecraft_profile_resolver=codecraft_resolver
        or StaticCodecraftProfileResolver(allowed=True),
    )


def test_map_worker_need_to_tool_query_uses_required_operations_as_logical_ids() -> None:
    request = _request()
    query = map_worker_capability_need_to_discovery_query(
        request.need,
        kind=CapabilityKind.TOOL,
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )
    assert query.kinds == (CapabilityKind.TOOL,)
    assert query.logical_identity is not None
    assert query.logical_identity.exact_logical_ids == (_OPERATION,)


def test_catalog_tool_a0_use_existing() -> None:
    service = _catalog_service(tool_registry=_tool_registry(_OPERATION))
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING
    assert result.decision is not None
    assert result.decision.autonomy_level is WorkerAutonomyLevel.A0_KNOWN_CAPABILITY
    assert result.decision.selected_candidate is not None
    assert (
        result.decision.selected_candidate.candidate_kind is WorkerCapabilityCandidateKind.TOOL
    )
    assert result.decision.selected_candidate.operations == (_OPERATION,)
    assert (
        result.decision.selected_candidate.operation_coverage
        is CapabilityOperationCoverage.EXACT
    )


def test_tool_exact_single_operation_adapter_outcome() -> None:
    tool_registry = _tool_registry(_OPERATION)
    snapshot = catalog_snapshot_from_registries(
        tool_registry=tool_registry,
        skill_registry=SkillRegistry(),
    )
    dependencies = catalog_discovery_dependencies(
        snapshot=snapshot,
        availability_evidence=host_availability_for_entries(
            *(
                entry
                for entry in snapshot.entries
                if entry.identity.logical.logical_id == _OPERATION
            ),
        ),
    )
    adapter = CapabilityCatalogToolDiscoveryAdapter(dependencies)
    outcome = adapter.discover(_request())

    assert outcome.disposition is CapabilityDiscoveryDisposition.MATCH_FOUND
    assert outcome.candidates is not None
    assert len(outcome.candidates) == 1
    candidate = outcome.candidates[0]
    assert candidate.operations == (_OPERATION,)
    assert candidate.operation_coverage is CapabilityOperationCoverage.EXACT


def test_tool_multiple_required_operations_partial_tools_no_match() -> None:
    logs_op = "tool.search.logs"
    incident_op = "tool.fetch.incident"
    tool_registry = _tool_registry(logs_op, incident_op)
    snapshot = catalog_snapshot_from_registries(
        tool_registry=tool_registry,
        skill_registry=SkillRegistry(),
    )
    dependencies = catalog_discovery_dependencies(
        snapshot=snapshot,
        availability_evidence=host_availability_for_entries(*snapshot.entries),
    )
    adapter = CapabilityCatalogToolDiscoveryAdapter(dependencies)
    outcome = adapter.discover(_request(required_operations=(logs_op, incident_op)))

    assert outcome.disposition is CapabilityDiscoveryDisposition.NO_MATCH
    assert outcome.candidates == ()


def test_partial_tool_cannot_reach_aw_a0_use_existing() -> None:
    logs_op = "tool.search.logs"
    incident_op = "tool.fetch.incident"
    service = _catalog_service(
        tool_registry=_tool_registry(logs_op),
        skill_registry=SkillRegistry(),
    )
    result = service.decide(_request(required_operations=(logs_op, incident_op)))

    selected = result.decision.selected_candidate if result.decision else None
    assert selected is None or selected.candidate_kind is not WorkerCapabilityCandidateKind.TOOL
    if selected is not None:
        assert selected.capability_ref != encode_source_qualified_capability_ref(
            tool_catalog_entry(logs_op).identity,
        )


def test_tool_support_helper_rejects_none_coverage() -> None:
    entry = tool_catalog_entry("tool.search.logs")
    resolved = _tool_supports_required_operations(
        identity=entry.identity,
        required_operations=("business.operation",),
    )
    assert resolved is None


def test_tool_support_helper_rejects_partial_coverage() -> None:
    entry = tool_catalog_entry("tool.search.logs")
    resolved = _tool_supports_required_operations(
        identity=entry.identity,
        required_operations=("tool.search.logs", "tool.fetch.incident"),
    )
    assert resolved is None


def test_tool_partial_then_skill_exact_selected() -> None:
    logs_op = "tool.search.logs"
    incident_op = "tool.fetch.incident"
    skill_id = "multi.skill"
    skill_registry = SkillRegistry()
    skill_registry.register(
        SkillManifest(
            skill_id=skill_id,
            description="multi",
            tool_ids=(logs_op, incident_op),
        ),
    )
    service = _catalog_service(
        tool_registry=_tool_registry(logs_op, incident_op),
        skill_registry=skill_registry,
        host_skill_ids=(skill_id,),
    )
    result = service.decide(_request(required_operations=(logs_op, incident_op)))

    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING
    assert result.decision is not None
    assert result.decision.selected_candidate is not None
    assert (
        result.decision.selected_candidate.candidate_kind is WorkerCapabilityCandidateKind.SKILL
    )


def test_tool_partial_does_not_block_skill_ladder() -> None:
    logs_op = "tool.search.logs"
    incident_op = "tool.fetch.incident"
    skill_id = "multi.skill"
    skill_registry = SkillRegistry()
    skill_registry.register(
        SkillManifest(
            skill_id=skill_id,
            description="multi",
            tool_ids=(logs_op, incident_op),
        ),
    )
    service = _catalog_service(
        tool_registry=_tool_registry(logs_op),
        skill_registry=skill_registry,
        host_skill_ids=(skill_id,),
    )
    result = service.decide(_request(required_operations=(logs_op, incident_op)))

    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING
    assert result.disposition is not CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY
    assert result.disposition is not CapabilityAcquisitionDisposition.UNAVAILABLE
    assert result.decision is not None
    assert result.decision.selected_candidate is not None
    assert (
        result.decision.selected_candidate.candidate_kind is WorkerCapabilityCandidateKind.SKILL
    )


def test_catalog_skill_fallback_when_tool_no_match() -> None:
    service = _catalog_service(
        tool_registry=ToolRegistry(),
        skill_registry=_skill_registry("csv.skill"),
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING
    assert result.decision is not None
    assert result.decision.selected_candidate is not None
    assert (
        result.decision.selected_candidate.candidate_kind is WorkerCapabilityCandidateKind.SKILL
    )


def test_tool_precedence_over_skill() -> None:
    class _CountingSkillDiscovery:
        def __init__(self, inner) -> None:
            self._inner = inner
            self.calls = 0

        def discover(self, request):
            self.calls += 1
            return self._inner.discover(request)

    tool_registry = _tool_registry(_OPERATION)
    skill_registry = _skill_registry("csv.skill")
    tool_discovery, skill_discovery = catalog_tool_skill_adapters(
        tool_registry=tool_registry,
        skill_registry=skill_registry,
    )
    counting_skill = _CountingSkillDiscovery(skill_discovery)
    service = WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_CAPABILITY_PROFILE),
        ),
        tool_discovery=tool_discovery,
        skill_discovery=counting_skill,
        integration_discovery=IntegrationCatalogCapabilityDiscoveryAdapter(),
        approved_alternate_discovery=NotConfiguredApprovedAlternateDiscovery(),
        configuration_discovery=NotConfiguredConfigurationOpportunityDiscovery(),
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
        codecraft_profile_resolver=StaticCodecraftProfileResolver(allowed=True),
    )
    result = service.decide(_request())

    assert result.decision is not None
    assert result.decision.selected_candidate is not None
    assert (
        result.decision.selected_candidate.candidate_kind is WorkerCapabilityCandidateKind.TOOL
    )
    assert counting_skill.calls == 0


def test_catalog_only_tool_does_not_become_a0() -> None:
    entry = tool_catalog_entry(_OPERATION)
    service = _catalog_service(
        tool_registry=ToolRegistry(),
        extra_entries=(entry,),
        availability_evidence=CapabilityDiscoveryAvailabilityEvidence(),
        host_tool_ids=(),
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.EPHEMERAL_GENERATION_CANDIDATE


def test_governance_blocked_tool_returns_policy_blocked() -> None:
    entry = tool_catalog_entry(_OPERATION)
    evidence = host_availability_for_entries(entry)
    denied_key = identity_key_from_entry_identity(entry.identity)
    dependencies = CapabilityCatalogDiscoveryDependencies(
        snapshot=CapabilityCatalogSnapshot(source_ids=("test",), entries=(entry,)),
        availability_evidence=evidence,
        governance_context=CapabilityGovernanceContext(
            posture=CapabilityGovernancePosture.STRICT,
            tool_evidence=CapabilityToolGovernanceEvidence(
                denied_keys=(denied_key,),
            ),
        ),
        governance_evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            ToolPolicyGovernanceEvaluator(),
        ),
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )
    service = WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_CAPABILITY_PROFILE),
        ),
        tool_discovery=CapabilityCatalogToolDiscoveryAdapter(dependencies),
        skill_discovery=UnavailableIntegrationCapabilityDiscovery(),
        integration_discovery=IntegrationCatalogCapabilityDiscoveryAdapter(),
        approved_alternate_discovery=NotConfiguredApprovedAlternateDiscovery(),
        configuration_discovery=NotConfiguredConfigurationOpportunityDiscovery(),
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY
    assert result.decision is not None
    assert result.decision.reason_code is CapabilityAcquisitionReasonCode.POLICY_BLOCKED


def test_source_failure_layer_returns_unavailable() -> None:
    class _UnavailableToolAdapter:
        def discover(self, request):
            del request
            from intergrax.contracts.autonomous_work.capability_acquisition import (
                WorkerCapabilityDiscoveryLayerOutcome,
            )

            return WorkerCapabilityDiscoveryLayerOutcome(
                disposition=CapabilityDiscoveryDisposition.UNAVAILABLE,
            )

    service = WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_CAPABILITY_PROFILE),
        ),
        tool_discovery=_UnavailableToolAdapter(),
        skill_discovery=UnavailableIntegrationCapabilityDiscovery(),
        integration_discovery=IntegrationCatalogCapabilityDiscoveryAdapter(),
        approved_alternate_discovery=NotConfiguredApprovedAlternateDiscovery(),
        configuration_discovery=NotConfiguredConfigurationOpportunityDiscovery(),
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
    )
    result = service.decide(_request())
    assert result.disposition is CapabilityAcquisitionDisposition.UNAVAILABLE


def test_private_source_identity_preserved() -> None:
    entry = tool_catalog_entry("foo.search", source=_PRIVATE_TOOL_SOURCE, version_label="1.0.0")
    service = _catalog_service(
        tool_registry=ToolRegistry(),
        extra_entries=(entry,),
        host_tool_ids=("foo.search",),
    )
    result = service.decide(_request(required_operations=("foo.search",)))

    assert result.decision is not None
    assert result.decision.selected_candidate is not None
    candidate = result.decision.selected_candidate
    assert candidate.capability_ref == encode_source_qualified_capability_ref(entry.identity)
    assert "enterprise.private" in candidate.capability_ref


def test_cross_source_same_logical_id_distinct_candidates() -> None:
    builtin = tool_catalog_entry("foo.search", source=_BUILTIN_TOOL_SOURCE)
    private = tool_catalog_entry("foo.search", source=_PRIVATE_TOOL_SOURCE)
    snapshot = CapabilityCatalogSnapshot(
        source_ids=("builtin", "private"),
        entries=tuple(sorted((builtin, private), key=lambda entry: entry.identity.sort_key)),
    )
    evidence = host_availability_for_entries(builtin, private)
    dependencies = catalog_discovery_dependencies(
        snapshot=snapshot,
        availability_evidence=evidence,
    )
    adapter = CapabilityCatalogToolDiscoveryAdapter(dependencies)
    request = _request(required_operations=("foo.search",))
    outcome = adapter.discover(request)

    assert outcome.disposition is CapabilityDiscoveryDisposition.MATCH_FOUND
    assert len(outcome.candidates) == 2
    refs = {candidate.capability_ref for candidate in outcome.candidates}
    assert len(refs) == 2
    assert refs == {
        encode_source_qualified_capability_ref(builtin.identity),
        encode_source_qualified_capability_ref(private.identity),
    }


def test_skill_version_preserved() -> None:
    entry = skill_catalog_entry(
        "skill.enterprise.research",
        version_label="2.4.0",
        source=_PRIVATE_TOOL_SOURCE,
    )
    skill_registry = SkillRegistry()
    skill_registry.register(
        SkillManifest(
            skill_id="skill.enterprise.research",
            version="2.4.0",
            description="research",
            tool_ids=(_OPERATION,),
        ),
    )
    service = _catalog_service(
        tool_registry=ToolRegistry(),
        skill_registry=skill_registry,
        extra_entries=(entry,),
        host_skill_ids=("skill.enterprise.research",),
    )
    result = service.decide(_request())

    assert result.decision is not None
    assert result.decision.selected_candidate is not None
    assert result.decision.selected_candidate.version == "2.4.0"


def test_no_registry_mutation_after_decision() -> None:
    tool_registry = _tool_registry(_OPERATION)
    skill_registry = _skill_registry("csv.skill")
    tool_before = list(tool_registry.list())
    skill_before = list(skill_registry.list())
    profile_before = ToolProfile()

    service = _catalog_service(tool_registry=tool_registry, skill_registry=skill_registry)
    service.decide(_request())

    assert list(tool_registry.list()) == tool_before
    assert list(skill_registry.list()) == skill_before
    assert ToolProfile() == profile_before


def test_authority_change_required_a4() -> None:
    class _AuthorityBlocked:
        def assess(self, *, worker_instance_id, candidate):
            del worker_instance_id, candidate
            return WorkerCapabilityAuthorityCompatibility.AUTHORITY_CHANGE_REQUIRED

    service = _catalog_service(
        tool_registry=_tool_registry(_OPERATION),
        authority=_AuthorityBlocked(),
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED
    assert result.decision is not None
    assert result.decision.autonomy_level is WorkerAutonomyLevel.A4_AUTHORITY_CHANGE


def test_codecraft_not_reached_when_tool_exists() -> None:
    resolver = _CountingCodecraftResolver(allowed=True)
    service = _catalog_service(
        tool_registry=_tool_registry(_OPERATION),
        codecraft_resolver=resolver,
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING
    assert resolver.calls == 0


def test_codecraft_not_reached_when_skill_exists() -> None:
    resolver = _CountingCodecraftResolver(allowed=True)
    service = _catalog_service(
        tool_registry=ToolRegistry(),
        skill_registry=_skill_registry("csv.skill"),
        codecraft_resolver=resolver,
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING
    assert resolver.calls == 0


def test_legacy_registry_adapter_still_available_for_explicit_wiring() -> None:
    service = _service(
        tool_registry=_tool_registry(_OPERATION),
        use_legacy_registry_discovery=True,
    )
    result = service.decide(_request())
    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING


def test_stage9_bridge_architecture_import_gates() -> None:
    module = importlib.import_module(
        "intergrax.autonomous_work.capability_catalog_discovery_adapters",
    )
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = "\n".join(imported)
    assert "DynamicAgentAcquisitionService" not in joined
    assert "codecraft" not in joined.lower()
    assert "from intergrax.tools.registry.runtime import ToolRegistry" not in source
