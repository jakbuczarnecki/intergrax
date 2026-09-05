# © Artur Czarnecki. All rights reserved.

"""AW-7A — capability acquisition policy semantic tests."""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from intergrax.autonomous_work.capability_acquisition_ports import (
    AllowAllAuthorityCompatibilityPort,
    StaticCodecraftProfileResolver,
    StaticWorkerCapabilityProfileResolver,
    UnavailableApprovedAlternateDiscovery,
    UnavailableConfigurationOpportunityDiscovery,
    UnavailableIntegrationCapabilityDiscovery,
    UnavailableSkillCapabilityDiscovery,
    UnavailableToolCapabilityDiscovery,
    permissive_capability_policy,
)
from intergrax.autonomous_work.capability_acquisition_service import (
    WorkerCapabilityAcquisitionDecisionService,
)
from intergrax.autonomous_work.capability_discovery_adapters import (
    IntegrationCatalogCapabilityDiscoveryAdapter,
    MappingApprovedAlternateDiscoveryAdapter,
    MappingConfigurationOpportunityDiscoveryAdapter,
    SkillRegistryCapabilityDiscoveryAdapter,
    ToolRegistryCapabilityDiscoveryAdapter,
    normalize_candidates,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityDiscoveryDisposition,
    CapabilityNeedKind,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityAcquisitionRequest,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
    WorkerCapabilityDiscoveryLayerOutcome,
    WorkerCapabilityNeed,
    WorkerCapabilityAuthorityCompatibility,
    derive_worker_capability_candidate_id,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    RecoveryDecisionReasonCode,
    RecoveryStrategy,
    WorkerObstacleKind,
    WorkerObstacleSourceKind,
    WorkerRecoveryDecision,
    derive_recovery_decision_id,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    CodecraftProfileRef,
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.references import ProblemReference
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry
from intergrax.integrations.registry.catalog import clear_catalog, register_integration
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.runtime import ToolRegistry
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_UTC = UTC
_NOW = datetime(2026, 9, 5, 8, 0, tzinfo=_UTC)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_EVIDENCE = ProblemReference("problem/evidence/capability-need-1")
_CAPABILITY_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)
_CODECRAFT_PROFILE = CodecraftProfileRef(
    profile_id="codecraft/default",
    version=initial_profile_version(),
)
_OPERATION = "document.parse_csv"


class _Input(BaseModel):
    value: str


class _Output(BaseModel):
    value: str


def _obstacle_id() -> str:
    return (
        f"{_WORKER_ID}:"
        f"{WorkerObstacleSourceKind.CAPABILITY_RESOLUTION.value}:"
        f"capability/missing/1:occurrence-1"
    )


def _recovery_decision(
    *,
    strategy: RecoveryStrategy = RecoveryStrategy.ACQUIRE_CAPABILITY,
    obstacle_kind: WorkerObstacleKind = WorkerObstacleKind.CAPABILITY_MISSING,
) -> WorkerRecoveryDecision:
    obstacle_id = _obstacle_id()
    return WorkerRecoveryDecision(
        decision_id=derive_recovery_decision_id(obstacle_id),
        obstacle_id=obstacle_id,
        obstacle_kind=obstacle_kind,
        strategy=strategy,
        decision_reason_code=RecoveryDecisionReasonCode.CAPABILITY_ACQUIRE_ALLOWED,
        evidence_refs=(_EVIDENCE,),
        decided_at=_NOW,
        source_ref="recovery/source/1",
    )


def _need(
    recovery: WorkerRecoveryDecision | None = None,
    **overrides,
) -> WorkerCapabilityNeed:
    resolved_recovery = recovery or _recovery_decision()
    base = WorkerCapabilityNeed(
        worker_instance_id=_WORKER_ID,
        obstacle_id=resolved_recovery.obstacle_id,
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=(_OPERATION,),
        capability_profile_ref=_CAPABILITY_PROFILE,
        requested_at=_NOW,
        recovery_decision_id=resolved_recovery.decision_id,
        evidence_refs=(_EVIDENCE,),
        codecraft_profile_ref=_CODECRAFT_PROFILE,
    )
    if overrides:
        return replace(base, **overrides)
    return base


def _request(
    recovery: WorkerRecoveryDecision | None = None,
    **need_overrides,
) -> WorkerCapabilityAcquisitionRequest:
    resolved_recovery = recovery or _recovery_decision()
    need = _need(resolved_recovery, **need_overrides)
    return WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=resolved_recovery,
        capability_profile_ref=_CAPABILITY_PROFILE,
        codecraft_profile_ref=_CODECRAFT_PROFILE,
    )


class _OkToolHandler:
    def execute(self, request):
        return _Output(value=request.input.value)


def _tool_registry(*tool_ids: str) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id in tool_ids:
        registry.register(
            ToolContract(
                tool_id=tool_id,
                name=tool_id,
                description=tool_id,
                input_schema=_Input,
                output_schema=_Output,
                side_effects=False,
                error_mapping={},
            ),
            _OkToolHandler(),
        )
    return registry


def _skill_registry(*skill_ids: str) -> SkillRegistry:
    registry = SkillRegistry()
    for skill_id in skill_ids:
        registry.register(
            SkillManifest(
                skill_id=skill_id,
                description=skill_id,
                tool_ids=(_OPERATION,) if skill_id != _OPERATION else (skill_id,),
            ),
        )
    return registry


def _service(
    *,
    tool_registry: ToolRegistry | None = None,
    skill_registry: SkillRegistry | None = None,
    integration_adapter: IntegrationCatalogCapabilityDiscoveryAdapter | None = None,
    tool_discovery=None,
    skill_discovery=None,
    authority=None,
    codecraft_allowed: bool = True,
    policy=None,
) -> WorkerCapabilityAcquisitionDecisionService:
    resolved_policy = policy or permissive_capability_policy(_CAPABILITY_PROFILE)
    return WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(resolved_policy),
        tool_discovery=tool_discovery
        or ToolRegistryCapabilityDiscoveryAdapter(tool_registry or ToolRegistry()),
        skill_discovery=skill_discovery
        or SkillRegistryCapabilityDiscoveryAdapter(skill_registry or SkillRegistry()),
        integration_discovery=integration_adapter
        or IntegrationCatalogCapabilityDiscoveryAdapter(),
        approved_alternate_discovery=UnavailableApprovedAlternateDiscovery(),
        configuration_discovery=UnavailableConfigurationOpportunityDiscovery(),
        authority_compatibility=authority or AllowAllAuthorityCompatibilityPort(),
        codecraft_profile_resolver=StaticCodecraftProfileResolver(allowed=codecraft_allowed),
    )


def test_existing_tool_selected_use_existing_a0() -> None:
    service = _service(tool_registry=_tool_registry(_OPERATION))
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING
    assert result.decision is not None
    assert result.decision.autonomy_level is WorkerAutonomyLevel.A0_KNOWN_CAPABILITY
    assert result.decision.selected_candidate is not None
    assert (
        result.decision.selected_candidate.candidate_kind is WorkerCapabilityCandidateKind.TOOL
    )
    assert result.decision.reason_code is CapabilityAcquisitionReasonCode.EXISTING_TOOL_SELECTED


def test_existing_skill_selected_when_no_tool() -> None:
    service = _service(
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


def test_existing_integration_selected_when_tool_and_skill_no_match() -> None:
    clear_catalog()
    register_integration(
        IntegrationEntry(
            slug="csv_parser",
            categories=(IntegrationCategory.DOCUMENT_PARSER,),
            factory=lambda: None,
        ),
    )
    service = _service(
        tool_registry=ToolRegistry(),
        skill_registry=SkillRegistry(),
        integration_adapter=IntegrationCatalogCapabilityDiscoveryAdapter(),
    )
    need = _need(required_operations=("integration:csv_parser",))
    recovery = _recovery_decision()
    result = service.decide(
        WorkerCapabilityAcquisitionRequest(
            need=need,
            recovery_decision=recovery,
            capability_profile_ref=_CAPABILITY_PROFILE,
        ),
    )

    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING
    assert result.decision is not None
    assert (
        result.decision.selected_candidate is not None
        and result.decision.selected_candidate.candidate_kind
        is WorkerCapabilityCandidateKind.INTEGRATION
    )
    clear_catalog()


def test_approved_alternate_selected() -> None:
    candidate = WorkerCapabilityCandidate(
        candidate_id=derive_worker_capability_candidate_id(
            candidate_kind=WorkerCapabilityCandidateKind.APPROVED_ALTERNATE,
            capability_ref="alternate:csv-flow",
        ),
        candidate_kind=WorkerCapabilityCandidateKind.APPROVED_ALTERNATE,
        capability_ref="alternate:csv-flow",
        source_domain="workflow",
        operations=(_OPERATION,),
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
    )
    service = WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_CAPABILITY_PROFILE),
        ),
        tool_discovery=ToolRegistryCapabilityDiscoveryAdapter(ToolRegistry()),
        skill_discovery=SkillRegistryCapabilityDiscoveryAdapter(SkillRegistry()),
        integration_discovery=IntegrationCatalogCapabilityDiscoveryAdapter(),
        approved_alternate_discovery=MappingApprovedAlternateDiscoveryAdapter(
            {(_OPERATION,): (candidate,)},
        ),
        configuration_discovery=UnavailableConfigurationOpportunityDiscovery(),
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING
    assert result.decision is not None
    assert result.decision.reason_code is (
        CapabilityAcquisitionReasonCode.APPROVED_ALTERNATE_SELECTED
    )


def test_existing_configuration_selected() -> None:
    candidate = WorkerCapabilityCandidate(
        candidate_id=derive_worker_capability_candidate_id(
            candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
            capability_ref="integration:csv_parser",
            configuration_ref="workspace/config/csv_parser",
        ),
        candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
        capability_ref="integration:csv_parser",
        source_domain="integrations",
        operations=(_OPERATION,),
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
        configuration_ref="workspace/config/csv_parser",
    )
    service = WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_CAPABILITY_PROFILE),
        ),
        tool_discovery=ToolRegistryCapabilityDiscoveryAdapter(ToolRegistry()),
        skill_discovery=SkillRegistryCapabilityDiscoveryAdapter(SkillRegistry()),
        integration_discovery=IntegrationCatalogCapabilityDiscoveryAdapter(),
        approved_alternate_discovery=UnavailableApprovedAlternateDiscovery(),
        configuration_discovery=MappingConfigurationOpportunityDiscoveryAdapter(
            {(_OPERATION,): (candidate,)},
        ),
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.CONFIGURE_EXISTING
    assert result.decision is not None
    assert result.decision.autonomy_level is WorkerAutonomyLevel.A0_KNOWN_CAPABILITY


def test_tool_discovery_unavailable_does_not_fall_through_to_skill() -> None:
    service = _service(
        tool_discovery=UnavailableToolCapabilityDiscovery(),
        skill_registry=_skill_registry("csv.skill"),
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.UNAVAILABLE
    assert result.decision is not None
    assert result.decision.reason_code is CapabilityAcquisitionReasonCode.DISCOVERY_UNAVAILABLE


def test_no_match_with_a1_allowed_returns_ephemeral_candidate_only() -> None:
    service = _service(tool_registry=ToolRegistry(), skill_registry=SkillRegistry())
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.EPHEMERAL_GENERATION_CANDIDATE
    assert result.decision is not None
    assert result.decision.autonomy_level is WorkerAutonomyLevel.A1_EPHEMERAL_SAFE
    assert result.decision.selected_candidate is not None
    assert (
        result.decision.selected_candidate.candidate_kind
        is WorkerCapabilityCandidateKind.CODECRAFT_EPHEMERAL
    )


def test_no_match_a1_forbidden_fails_closed() -> None:
    policy = permissive_capability_policy(_CAPABILITY_PROFILE)
    restricted = replace(policy, generated_capability_allowed=False)
    service = _service(
        tool_registry=ToolRegistry(),
        skill_registry=SkillRegistry(),
        policy=restricted,
        codecraft_allowed=False,
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY


def test_schema_adaptation_returns_a2_candidate_only() -> None:
    recovery = _recovery_decision(strategy=RecoveryStrategy.ADAPT_INTEGRATION)
    service = _service(tool_registry=ToolRegistry(), skill_registry=SkillRegistry())
    result = service.decide(
        _request(
            recovery,
            need_kind=CapabilityNeedKind.SCHEMA_ADAPTATION,
        ),
    )

    assert result.disposition is CapabilityAcquisitionDisposition.SCOPED_ADAPTATION_CANDIDATE
    assert result.decision is not None
    assert result.decision.autonomy_level is WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE


def test_production_change_required_a3() -> None:
    policy = permissive_capability_policy(_CAPABILITY_PROFILE)
    restricted = replace(
        policy,
        generated_capability_allowed=False,
        adaptive_integration_allowed=False,
        durable_change_allowed=True,
    )
    service = _service(
        tool_registry=ToolRegistry(),
        skill_registry=SkillRegistry(),
        policy=restricted,
        codecraft_allowed=False,
    )
    result = service.decide(_request(need_kind=CapabilityNeedKind.EXTERNAL_INTEGRATION))

    assert result.disposition is CapabilityAcquisitionDisposition.PRODUCTION_CHANGE_REQUIRED
    assert result.decision is not None
    assert result.decision.autonomy_level is WorkerAutonomyLevel.A3_PRODUCTION_CHANGE


def test_authority_change_required_a4_without_executable_candidate() -> None:
    class RequiresWriteAuthority:
        def assess(self, *, worker_instance_id, candidate):
            del worker_instance_id, candidate
            return WorkerCapabilityAuthorityCompatibility.AUTHORITY_CHANGE_REQUIRED

    service = _service(
        tool_registry=_tool_registry(_OPERATION),
        authority=RequiresWriteAuthority(),
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED
    assert result.decision is not None
    assert result.decision.autonomy_level is WorkerAutonomyLevel.A4_AUTHORITY_CHANGE
    assert result.decision.selected_candidate is None


def test_policy_denied_defense_zero_acquisition() -> None:
    service = _service(tool_registry=_tool_registry(_OPERATION))
    recovery = _recovery_decision(
        obstacle_kind=WorkerObstacleKind.POLICY_DENIED,
        strategy=RecoveryStrategy.STOP,
    )
    result = service.decide(_request(recovery))

    assert result.disposition is CapabilityAcquisitionDisposition.ESCALATE
    assert result.decision is not None
    assert result.decision.reason_code is CapabilityAcquisitionReasonCode.POLICY_DENIED_DEFENSE


def test_credential_obstacle_defense() -> None:
    service = _service(tool_registry=_tool_registry(_OPERATION))
    recovery = _recovery_decision(
        obstacle_kind=WorkerObstacleKind.CREDENTIAL_UNAVAILABLE,
        strategy=RecoveryStrategy.ESCALATE,
    )
    result = service.decide(_request(recovery))

    assert result.disposition is CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED
    assert result.decision is not None
    assert result.decision.reason_code is CapabilityAcquisitionReasonCode.CREDENTIAL_DEFENSE


def test_wrong_recovery_strategy_rejected() -> None:
    service = _service(tool_registry=_tool_registry(_OPERATION))
    result = service.decide(_request(_recovery_decision(strategy=RecoveryStrategy.WAIT)))

    assert result.disposition is CapabilityAcquisitionDisposition.ESCALATE
    assert result.decision is not None
    assert result.decision.reason_code is (
        CapabilityAcquisitionReasonCode.RECOVERY_STRATEGY_REJECTED
    )


def test_profile_unavailable_fail_closed() -> None:
    service = WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(
                CapabilityProfileRef(profile_id="other", version=initial_profile_version()),
            ),
        ),
        tool_discovery=UnavailableToolCapabilityDiscovery(),
        skill_discovery=UnavailableSkillCapabilityDiscovery(),
        integration_discovery=UnavailableIntegrationCapabilityDiscovery(),
        approved_alternate_discovery=UnavailableApprovedAlternateDiscovery(),
        configuration_discovery=UnavailableConfigurationOpportunityDiscovery(),
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
    )
    result = service.decide(_request())

    assert result.disposition is CapabilityAcquisitionDisposition.UNAVAILABLE


def test_candidate_ordering_is_deterministic() -> None:
    candidates = [
        WorkerCapabilityCandidate(
            candidate_id=f"id-{index}",
            candidate_kind=WorkerCapabilityCandidateKind.TOOL,
            capability_ref=f"tool:{index}",
            source_domain="tools",
            operations=(_OPERATION,),
            risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
            evidence_refs=(_EVIDENCE,),
            discovered_at=_NOW,
        )
        for index in range(5)
    ]
    winners: set[str] = set()
    for _ in range(100):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        normalized = normalize_candidates(tuple(shuffled))
        assert isinstance(normalized, tuple)
        winners.add(normalized[0].candidate_id)
    assert len(winners) == 1


def test_duplicate_candidates_deduplicated() -> None:
    candidate = WorkerCapabilityCandidate(
        candidate_id=derive_worker_capability_candidate_id(
            candidate_kind=WorkerCapabilityCandidateKind.TOOL,
            capability_ref="tool:document.parse_csv",
            version="1.0.0",
        ),
        candidate_kind=WorkerCapabilityCandidateKind.TOOL,
        capability_ref="tool:document.parse_csv",
        source_domain="tools",
        version="1.0.0",
        operations=(_OPERATION,),
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
    )
    normalized = normalize_candidates((candidate, candidate))
    assert isinstance(normalized, tuple)
    assert len(normalized) == 1


def test_conflicting_duplicate_candidates() -> None:
    first = WorkerCapabilityCandidate(
        candidate_id=derive_worker_capability_candidate_id(
            candidate_kind=WorkerCapabilityCandidateKind.TOOL,
            capability_ref="tool:document.parse_csv",
            version="1.0.0",
        ),
        candidate_kind=WorkerCapabilityCandidateKind.TOOL,
        capability_ref="tool:document.parse_csv",
        source_domain="tools",
        version="1.0.0",
        operations=(_OPERATION,),
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
    )
    second = WorkerCapabilityCandidate(
        candidate_id=first.candidate_id,
        candidate_kind=WorkerCapabilityCandidateKind.TOOL,
        capability_ref="tool:document.parse_csv",
        source_domain="tools",
        version="1.0.0",
        operations=("other.operation",),
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
    )
    assert normalize_candidates((first, second)) is CapabilityDiscoveryDisposition.CONFLICT


def test_a4_construction_rejects_executable_candidate() -> None:
    candidate = WorkerCapabilityCandidate(
        candidate_id="tool:bad",
        candidate_kind=WorkerCapabilityCandidateKind.TOOL,
        capability_ref="tool:bad",
        source_domain="tools",
        operations=(_OPERATION,),
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
    )
    with pytest.raises(ValueError, match="A4 decisions must not carry executable"):
        WorkerCapabilityAcquisitionDecision(
            decision_id="decision-1",
            worker_instance_id=_WORKER_ID,
            obstacle_id=_obstacle_id(),
            recovery_decision_id="recovery-1",
            need_id="need-1",
            disposition=CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED,
            selected_candidate=candidate,
            autonomy_level=WorkerAutonomyLevel.A4_AUTHORITY_CHANGE,
            capability_profile_ref=_CAPABILITY_PROFILE,
            reason_code=CapabilityAcquisitionReasonCode.A4_AUTHORITY_CHANGE_REQUIRED,
            evidence_refs=(_EVIDENCE,),
            decided_at=_NOW,
        )


def test_authority_need_in_resource_refs_rejected() -> None:
    service = _service(tool_registry=_tool_registry(_OPERATION))
    result = service.decide(_request(required_resource_refs=("credential/db-write",)))

    assert result.disposition is CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED
    assert result.decision is not None
    assert result.decision.reason_code is CapabilityAcquisitionReasonCode.AUTHORITY_NEED_REJECTED


class _FakeDiscoveryPort:
    def __init__(self, outcome: WorkerCapabilityDiscoveryLayerOutcome) -> None:
        self.outcome = outcome
        self.calls = 0

    def discover(self, request):
        del request
        self.calls += 1
        return self.outcome


def test_plugin_discovery_port_injection() -> None:
    candidate = WorkerCapabilityCandidate(
        candidate_id="tool:fake",
        candidate_kind=WorkerCapabilityCandidateKind.TOOL,
        capability_ref="tool:fake",
        source_domain="tools",
        operations=(_OPERATION,),
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
    )
    fake = _FakeDiscoveryPort(
        WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.MATCH_FOUND,
            candidates=(candidate,),
        ),
    )
    service = WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_CAPABILITY_PROFILE),
        ),
        tool_discovery=fake,
        skill_discovery=UnavailableSkillCapabilityDiscovery(),
        integration_discovery=UnavailableIntegrationCapabilityDiscovery(),
        approved_alternate_discovery=UnavailableApprovedAlternateDiscovery(),
        configuration_discovery=UnavailableConfigurationOpportunityDiscovery(),
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
    )
    result = service.decide(_request())

    assert fake.calls == 1
    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING
