# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded worker capability acquisition decision service (AW-7A).

Accepts correlated recovery evidence and capability need, resolves profile policy,
queries the canonical discovery ladder, enforces A0–A4 invariants, and returns
a typed acquisition decision. Does not execute acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.autonomous_work.capability_acquisition_ports import (
    WorkerApprovedAlternateDiscoveryPort,
    WorkerCapabilityAuthorityCompatibilityPort,
    WorkerCapabilityProfileResolutionError,
    WorkerCapabilityProfileResolver,
    WorkerCodecraftProfileResolutionError,
    WorkerCodecraftProfileResolver,
    WorkerConfigurationOpportunityDiscoveryPort,
    WorkerIntegrationCapabilityDiscoveryPort,
    WorkerSkillCapabilityDiscoveryPort,
    WorkerToolCapabilityDiscoveryPort,
)
from intergrax.autonomous_work.capability_discovery_adapters import normalize_candidates
from intergrax.contracts.autonomous_work.capability_acquisition import (
    ACQUISITION_DECISION_POLICY_VERSION,
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityDiscoveryDisposition,
    CapabilityNeedKind,
    ResolvedWorkerCapabilityPolicy,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityAcquisitionRequest,
    WorkerCapabilityAcquisitionResult,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
    WorkerCapabilityDiscoveryLayerOutcome,
    WorkerCapabilityDiscoveryRequest,
    WorkerCapabilityDiscoveryResult,
    WorkerCapabilityNeed,
    WorkerCapabilityAuthorityCompatibility,
    autonomy_level_allowed,
    derive_worker_capability_acquisition_decision_id,
    derive_worker_capability_candidate_id,
    derive_worker_capability_need_id,
    is_capability_acquisition_recovery_strategy,
    need_implies_authority_expansion,
    operations_allowed_by_policy,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    RecoveryStrategy,
    WorkerObstacleKind,
)

_LADDER_RANK: dict[WorkerCapabilityCandidateKind, int] = {
    WorkerCapabilityCandidateKind.TOOL: 0,
    WorkerCapabilityCandidateKind.SKILL: 1,
    WorkerCapabilityCandidateKind.INTEGRATION: 2,
    WorkerCapabilityCandidateKind.APPROVED_ALTERNATE: 3,
    WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION: 4,
}

_DISPOSITION_REASON: dict[WorkerCapabilityCandidateKind, CapabilityAcquisitionReasonCode] = {
    WorkerCapabilityCandidateKind.TOOL: CapabilityAcquisitionReasonCode.EXISTING_TOOL_SELECTED,
    WorkerCapabilityCandidateKind.SKILL: CapabilityAcquisitionReasonCode.EXISTING_SKILL_SELECTED,
    WorkerCapabilityCandidateKind.INTEGRATION: (
        CapabilityAcquisitionReasonCode.EXISTING_INTEGRATION_SELECTED
    ),
    WorkerCapabilityCandidateKind.APPROVED_ALTERNATE: (
        CapabilityAcquisitionReasonCode.APPROVED_ALTERNATE_SELECTED
    ),
    WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION: (
        CapabilityAcquisitionReasonCode.EXISTING_CONFIGURATION_SELECTED
    ),
}


@dataclass(frozen=True, slots=True)
class _DiscoveryLayer:
    name: str
    port: (
        WorkerToolCapabilityDiscoveryPort
        | WorkerSkillCapabilityDiscoveryPort
        | WorkerIntegrationCapabilityDiscoveryPort
        | WorkerApprovedAlternateDiscoveryPort
        | WorkerConfigurationOpportunityDiscoveryPort
    )


class WorkerCapabilityAcquisitionDecisionService:
    """Need + recovery correlation → deterministic acquisition decision."""

    def __init__(
        self,
        *,
        profile_resolver: WorkerCapabilityProfileResolver,
        tool_discovery: WorkerToolCapabilityDiscoveryPort,
        skill_discovery: WorkerSkillCapabilityDiscoveryPort,
        integration_discovery: WorkerIntegrationCapabilityDiscoveryPort,
        approved_alternate_discovery: WorkerApprovedAlternateDiscoveryPort,
        configuration_discovery: WorkerConfigurationOpportunityDiscoveryPort,
        authority_compatibility: WorkerCapabilityAuthorityCompatibilityPort,
        codecraft_profile_resolver: WorkerCodecraftProfileResolver | None = None,
    ) -> None:
        self._profile_resolver = profile_resolver
        self._authority_compatibility = authority_compatibility
        self._codecraft_profile_resolver = codecraft_profile_resolver
        self._layers = (
            _DiscoveryLayer("tool", tool_discovery),
            _DiscoveryLayer("skill", skill_discovery),
            _DiscoveryLayer("integration", integration_discovery),
            _DiscoveryLayer("approved_alternate", approved_alternate_discovery),
            _DiscoveryLayer("configuration", configuration_discovery),
        )

    def decide(
        self,
        request: WorkerCapabilityAcquisitionRequest,
        *,
        decided_at: datetime | None = None,
    ) -> WorkerCapabilityAcquisitionResult:
        timestamp = decided_at or request.need.requested_at
        defense = _defense_rejection(request)
        if defense is not None:
            return defense

        if need_implies_authority_expansion(request.need):
            return _authority_change_result(
                request=request,
                reason_code=CapabilityAcquisitionReasonCode.AUTHORITY_NEED_REJECTED,
                decided_at=timestamp,
            )

        try:
            policy = self._profile_resolver.resolve(request.capability_profile_ref)
        except WorkerCapabilityProfileResolutionError:
            return _simple_result(
                request=request,
                disposition=CapabilityAcquisitionDisposition.UNAVAILABLE,
                reason_code=CapabilityAcquisitionReasonCode.PROFILE_UNAVAILABLE,
                decided_at=timestamp,
            )

        if policy.profile_ref != request.capability_profile_ref:
            return _simple_result(
                request=request,
                disposition=CapabilityAcquisitionDisposition.CONFLICT,
                reason_code=CapabilityAcquisitionReasonCode.STALE_PROFILE,
                decided_at=timestamp,
            )

        discovery_request = WorkerCapabilityDiscoveryRequest(
            need=request.need,
            profile_ref=policy.profile_ref,
            worker_instance_id=request.need.worker_instance_id,
        )

        all_candidates: list[WorkerCapabilityCandidate] = []
        for layer in self._layers:
            outcome = layer.port.discover(discovery_request)
            layer_result = _handle_layer_outcome(outcome)
            if isinstance(layer_result, WorkerCapabilityAcquisitionResult):
                if layer_result.decision is None:
                    return _simple_result(
                        request=request,
                        disposition=layer_result.disposition,
                        reason_code=CapabilityAcquisitionReasonCode.DISCOVERY_UNAVAILABLE
                        if layer_result.disposition is CapabilityAcquisitionDisposition.UNAVAILABLE
                        else CapabilityAcquisitionReasonCode.CONFLICT,
                        decided_at=timestamp,
                    )
                return layer_result
            if layer_result:
                all_candidates.extend(layer_result)
                selected = self._select_existing_candidate(
                    request=request,
                    policy=policy,
                    candidates=tuple(all_candidates),
                    decided_at=timestamp,
                )
                if selected is not None:
                    return selected

        return self._classify_generated_candidate(
            request=request,
            policy=policy,
            decided_at=timestamp,
        )

    def _select_existing_candidate(
        self,
        *,
        request: WorkerCapabilityAcquisitionRequest,
        policy: ResolvedWorkerCapabilityPolicy,
        candidates: tuple[WorkerCapabilityCandidate, ...],
        decided_at: datetime,
    ) -> WorkerCapabilityAcquisitionResult | None:
        normalized = normalize_candidates(candidates)
        if isinstance(normalized, CapabilityDiscoveryDisposition):
            return _simple_result(
                request=request,
                disposition=CapabilityAcquisitionDisposition.CONFLICT,
                reason_code=CapabilityAcquisitionReasonCode.CONFLICT,
                decided_at=decided_at,
            )

        eligible: list[tuple[WorkerCapabilityCandidate, WorkerCapabilityAuthorityCompatibility]] = []
        authority_blocked: list[WorkerCapabilityCandidate] = []
        policy_blocked_autonomy = False
        for candidate in normalized:
            if candidate.candidate_kind not in policy.allowed_candidate_kinds:
                continue
            if not autonomy_level_allowed(
                candidate.risk_class,
                policy.allowed_autonomy_levels,
            ):
                policy_blocked_autonomy = True
                continue
            if not operations_allowed_by_policy(
                request.need.required_operations,
                policy.allowed_operation_patterns,
            ):
                continue
            compatibility = self._authority_compatibility.assess(
                worker_instance_id=request.need.worker_instance_id,
                candidate=candidate,
            )
            if compatibility is WorkerCapabilityAuthorityCompatibility.UNAVAILABLE:
                return _simple_result(
                    request=request,
                    disposition=CapabilityAcquisitionDisposition.UNAVAILABLE,
                    reason_code=CapabilityAcquisitionReasonCode.DISCOVERY_UNAVAILABLE,
                    decided_at=decided_at,
                )
            if compatibility is WorkerCapabilityAuthorityCompatibility.AUTHORITY_CHANGE_REQUIRED:
                authority_blocked.append(candidate)
                continue
            eligible.append((candidate, compatibility))

        if eligible:
            selected = _deterministic_select(eligible)
            disposition = (
                CapabilityAcquisitionDisposition.CONFIGURE_EXISTING
                if selected.candidate_kind is WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION
                else CapabilityAcquisitionDisposition.USE_EXISTING
            )
            reason = _DISPOSITION_REASON[selected.candidate_kind]
            return _decision_result(
                request=request,
                disposition=disposition,
                reason_code=reason,
                selected_candidate=selected,
                autonomy_level=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
                decided_at=decided_at,
            )

        if authority_blocked:
            return _authority_change_result(
                request=request,
                reason_code=CapabilityAcquisitionReasonCode.A4_AUTHORITY_CHANGE_REQUIRED,
                decided_at=decided_at,
            )
        if policy_blocked_autonomy:
            return _policy_blocked_result(request, decided_at)
        return None

    def _classify_generated_candidate(
        self,
        *,
        request: WorkerCapabilityAcquisitionRequest,
        policy: ResolvedWorkerCapabilityPolicy,
        decided_at: datetime,
    ) -> WorkerCapabilityAcquisitionResult:
        if not operations_allowed_by_policy(
            request.need.required_operations,
            policy.allowed_operation_patterns,
        ):
            return _policy_blocked_result(request, decided_at)

        strategy = request.recovery_decision.strategy
        need_kind = request.need.need_kind

        if strategy is RecoveryStrategy.ADAPT_INTEGRATION or need_kind in {
            CapabilityNeedKind.SCHEMA_ADAPTATION,
            CapabilityNeedKind.PROTOCOL_ADAPTATION,
        }:
            if (
                policy.adaptive_integration_allowed
                and WorkerCapabilityCandidateKind.ADAPTIVE_INTEGRATION
                in policy.allowed_candidate_kinds
            ):
                if autonomy_level_allowed(
                    WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE,
                    policy.allowed_autonomy_levels,
                ):
                    candidate = _synthetic_candidate(
                        request.need,
                        candidate_kind=WorkerCapabilityCandidateKind.ADAPTIVE_INTEGRATION,
                        autonomy=WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE,
                        capability_ref="adaptive:integration",
                    )
                    return _decision_result(
                        request=request,
                        disposition=CapabilityAcquisitionDisposition.SCOPED_ADAPTATION_CANDIDATE,
                        reason_code=CapabilityAcquisitionReasonCode.A2_ADAPTATION_REQUIRED,
                        selected_candidate=candidate,
                        autonomy_level=WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE,
                        decided_at=decided_at,
                    )
                return _policy_blocked_result(request, decided_at)
            return _fail_closed_no_safe(request, decided_at)

        if (
            policy.durable_change_allowed
            and WorkerCapabilityCandidateKind.DURABLE_PRODUCTION_CHANGE
            in policy.allowed_candidate_kinds
            and need_kind is CapabilityNeedKind.EXTERNAL_INTEGRATION
        ):
            if autonomy_level_allowed(
                WorkerAutonomyLevel.A3_PRODUCTION_CHANGE,
                policy.allowed_autonomy_levels,
            ):
                candidate = _synthetic_candidate(
                    request.need,
                    candidate_kind=WorkerCapabilityCandidateKind.DURABLE_PRODUCTION_CHANGE,
                    autonomy=WorkerAutonomyLevel.A3_PRODUCTION_CHANGE,
                    capability_ref="durable:production-change",
                )
                return _decision_result(
                    request=request,
                    disposition=CapabilityAcquisitionDisposition.PRODUCTION_CHANGE_REQUIRED,
                    reason_code=CapabilityAcquisitionReasonCode.A3_PRODUCTION_CHANGE_REQUIRED,
                    selected_candidate=candidate,
                    autonomy_level=WorkerAutonomyLevel.A3_PRODUCTION_CHANGE,
                    decided_at=decided_at,
                )
            return _policy_blocked_result(request, decided_at)

        if policy.generated_capability_allowed:
            if (
                WorkerCapabilityCandidateKind.CODECRAFT_EPHEMERAL
                not in policy.allowed_candidate_kinds
            ):
                return _policy_blocked_result(request, decided_at)
            if autonomy_level_allowed(
                WorkerAutonomyLevel.A1_EPHEMERAL_SAFE,
                policy.allowed_autonomy_levels,
            ) and _codecraft_allowed(
                request,
                self._codecraft_profile_resolver,
            ):
                candidate = _synthetic_candidate(
                    request.need,
                    candidate_kind=WorkerCapabilityCandidateKind.CODECRAFT_EPHEMERAL,
                    autonomy=WorkerAutonomyLevel.A1_EPHEMERAL_SAFE,
                    capability_ref="ephemeral:codecraft",
                )
                return _decision_result(
                    request=request,
                    disposition=CapabilityAcquisitionDisposition.EPHEMERAL_GENERATION_CANDIDATE,
                    reason_code=CapabilityAcquisitionReasonCode.A1_CANDIDATE_ALLOWED,
                    selected_candidate=candidate,
                    autonomy_level=WorkerAutonomyLevel.A1_EPHEMERAL_SAFE,
                    decided_at=decided_at,
                )
            if not autonomy_level_allowed(
                WorkerAutonomyLevel.A1_EPHEMERAL_SAFE,
                policy.allowed_autonomy_levels,
            ):
                return _policy_blocked_result(request, decided_at)

        return _fail_closed_no_safe(request, decided_at)


def _handle_layer_outcome(
    outcome: WorkerCapabilityDiscoveryLayerOutcome,
) -> list[WorkerCapabilityCandidate] | WorkerCapabilityAcquisitionResult | None:
    if outcome.disposition is CapabilityDiscoveryDisposition.CONFLICT:
        return WorkerCapabilityAcquisitionResult(
            disposition=CapabilityAcquisitionDisposition.CONFLICT,
            decision=None,
        )
    if outcome.disposition is CapabilityDiscoveryDisposition.UNAVAILABLE:
        return WorkerCapabilityAcquisitionResult(
            disposition=CapabilityAcquisitionDisposition.UNAVAILABLE,
            decision=None,
        )
    if outcome.disposition is CapabilityDiscoveryDisposition.NOT_CONFIGURED:
        return None
    if outcome.disposition is CapabilityDiscoveryDisposition.NO_MATCH:
        return None
    if outcome.disposition is CapabilityDiscoveryDisposition.MATCH_FOUND:
        return list(outcome.candidates)
    return None


def _defense_rejection(
    request: WorkerCapabilityAcquisitionRequest,
) -> WorkerCapabilityAcquisitionResult | None:
    recovery = request.recovery_decision
    obstacle_kind = recovery.obstacle_kind
    strategy = recovery.strategy

    if obstacle_kind is WorkerObstacleKind.POLICY_DENIED or strategy is RecoveryStrategy.STOP:
        return WorkerCapabilityAcquisitionResult(
            disposition=CapabilityAcquisitionDisposition.ESCALATE,
            decision=_decision_only(
                request=request,
                disposition=CapabilityAcquisitionDisposition.ESCALATE,
                reason_code=CapabilityAcquisitionReasonCode.POLICY_DENIED_DEFENSE,
                decided_at=request.need.requested_at,
            ),
        )

    if obstacle_kind is WorkerObstacleKind.CREDENTIAL_UNAVAILABLE:
        return _authority_change_result(
            request=request,
            reason_code=CapabilityAcquisitionReasonCode.CREDENTIAL_DEFENSE,
            decided_at=request.need.requested_at,
        )

    if not is_capability_acquisition_recovery_strategy(strategy):
        return WorkerCapabilityAcquisitionResult(
            disposition=CapabilityAcquisitionDisposition.ESCALATE,
            decision=_decision_only(
                request=request,
                disposition=CapabilityAcquisitionDisposition.ESCALATE,
                reason_code=CapabilityAcquisitionReasonCode.RECOVERY_STRATEGY_REJECTED,
                decided_at=request.need.requested_at,
            ),
        )
    return None


def _deterministic_select(
    eligible: list[tuple[WorkerCapabilityCandidate, WorkerCapabilityAuthorityCompatibility]],
) -> WorkerCapabilityCandidate:
    def sort_key(item: tuple[WorkerCapabilityCandidate, WorkerCapabilityAuthorityCompatibility]) -> tuple:
        candidate = item[0]
        return (
            _LADDER_RANK.get(candidate.candidate_kind, 99),
            -len(candidate.operations),
            candidate.capability_ref,
            candidate.candidate_id,
        )

    return sorted(eligible, key=sort_key)[0][0]


def _synthetic_candidate(
    need: WorkerCapabilityNeed,
    *,
    candidate_kind: WorkerCapabilityCandidateKind,
    autonomy: WorkerAutonomyLevel,
    capability_ref: str,
) -> WorkerCapabilityCandidate:
    return WorkerCapabilityCandidate(
        candidate_id=derive_worker_capability_candidate_id(
            candidate_kind=candidate_kind,
            capability_ref=capability_ref,
        ),
        candidate_kind=candidate_kind,
        capability_ref=capability_ref,
        source_domain="autonomous_work",
        operations=need.required_operations,
        risk_class=autonomy,
        evidence_refs=need.evidence_refs,
        discovered_at=need.requested_at,
    )


def _codecraft_allowed(
    request: WorkerCapabilityAcquisitionRequest,
    resolver: WorkerCodecraftProfileResolver | None,
) -> bool:
    profile_ref = request.codecraft_profile_ref or request.need.codecraft_profile_ref
    if profile_ref is None or resolver is None:
        return False
    try:
        return resolver.is_candidate_consideration_allowed(profile_ref)
    except WorkerCodecraftProfileResolutionError:
        return False


def _fail_closed_no_safe(
    request: WorkerCapabilityAcquisitionRequest,
    decided_at: datetime,
) -> WorkerCapabilityAcquisitionResult:
    return _simple_result(
        request=request,
        disposition=CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY,
        reason_code=CapabilityAcquisitionReasonCode.NO_SAFE_CANDIDATE,
        decided_at=decided_at,
    )


def _policy_blocked_result(
    request: WorkerCapabilityAcquisitionRequest,
    decided_at: datetime,
) -> WorkerCapabilityAcquisitionResult:
    return _simple_result(
        request=request,
        disposition=CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY,
        reason_code=CapabilityAcquisitionReasonCode.POLICY_BLOCKED,
        decided_at=decided_at,
    )


def _authority_change_result(
    *,
    request: WorkerCapabilityAcquisitionRequest,
    reason_code: CapabilityAcquisitionReasonCode,
    decided_at: datetime,
) -> WorkerCapabilityAcquisitionResult:
    return _decision_result(
        request=request,
        disposition=CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED,
        reason_code=reason_code,
        selected_candidate=None,
        autonomy_level=WorkerAutonomyLevel.A4_AUTHORITY_CHANGE,
        decided_at=decided_at,
    )


def _simple_result(
    *,
    request: WorkerCapabilityAcquisitionRequest,
    disposition: CapabilityAcquisitionDisposition,
    reason_code: CapabilityAcquisitionReasonCode,
    decided_at: datetime,
) -> WorkerCapabilityAcquisitionResult:
    return WorkerCapabilityAcquisitionResult(
        disposition=disposition,
        decision=_decision_only(
            request=request,
            disposition=disposition,
            reason_code=reason_code,
            decided_at=decided_at,
        ),
    )


def _decision_result(
    *,
    request: WorkerCapabilityAcquisitionRequest,
    disposition: CapabilityAcquisitionDisposition,
    reason_code: CapabilityAcquisitionReasonCode,
    selected_candidate: WorkerCapabilityCandidate | None,
    autonomy_level: WorkerAutonomyLevel | None,
    decided_at: datetime,
) -> WorkerCapabilityAcquisitionResult:
    decision = _build_decision(
        request=request,
        disposition=disposition,
        reason_code=reason_code,
        selected_candidate=selected_candidate,
        autonomy_level=autonomy_level,
        decided_at=decided_at,
    )
    discovery = WorkerCapabilityDiscoveryResult(
        need=request.need,
        candidates=(selected_candidate,) if selected_candidate is not None else (),
        disposition=CapabilityDiscoveryDisposition.MATCH_FOUND
        if selected_candidate is not None
        else CapabilityDiscoveryDisposition.NO_MATCH,
        profile_ref=request.capability_profile_ref,
        discovered_at=decided_at,
    )
    return WorkerCapabilityAcquisitionResult(
        disposition=disposition,
        decision=decision,
        discovery=discovery,
    )


def _decision_only(
    *,
    request: WorkerCapabilityAcquisitionRequest,
    disposition: CapabilityAcquisitionDisposition,
    reason_code: CapabilityAcquisitionReasonCode,
    decided_at: datetime,
) -> WorkerCapabilityAcquisitionDecision:
    return _build_decision(
        request=request,
        disposition=disposition,
        reason_code=reason_code,
        selected_candidate=None,
        autonomy_level=None,
        decided_at=decided_at,
    )


def _build_decision(
    *,
    request: WorkerCapabilityAcquisitionRequest,
    disposition: CapabilityAcquisitionDisposition,
    reason_code: CapabilityAcquisitionReasonCode,
    selected_candidate: WorkerCapabilityCandidate | None,
    autonomy_level: WorkerAutonomyLevel | None,
    decided_at: datetime,
) -> WorkerCapabilityAcquisitionDecision:
    need = request.need
    need_id = derive_worker_capability_need_id(need)
    selected_id = selected_candidate.candidate_id if selected_candidate is not None else None
    decision_id = derive_worker_capability_acquisition_decision_id(
        worker_instance_id=need.worker_instance_id,
        obstacle_id=need.obstacle_id,
        recovery_decision_id=need.recovery_decision_id,
        need_id=need_id,
        capability_profile_version=need.capability_profile_ref.version.value,
        selected_candidate_id=selected_id,
        decision_policy_version=ACQUISITION_DECISION_POLICY_VERSION,
    )
    evidence_refs = need.evidence_refs
    if selected_candidate is not None:
        evidence_refs = evidence_refs + selected_candidate.evidence_refs
    return WorkerCapabilityAcquisitionDecision(
        decision_id=decision_id,
        worker_instance_id=need.worker_instance_id,
        obstacle_id=need.obstacle_id,
        recovery_decision_id=need.recovery_decision_id,
        need_id=need_id,
        disposition=disposition,
        selected_candidate=selected_candidate,
        autonomy_level=autonomy_level,
        capability_profile_ref=request.capability_profile_ref,
        codecraft_profile_ref=request.codecraft_profile_ref or need.codecraft_profile_ref,
        reason_code=reason_code,
        evidence_refs=evidence_refs,
        decided_at=decided_at,
        decision_policy_version=ACQUISITION_DECISION_POLICY_VERSION,
    )
