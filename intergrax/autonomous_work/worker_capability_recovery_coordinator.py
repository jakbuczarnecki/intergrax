# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker capability recovery coordinator — canonical discovery/UCA, AW-specific mapping (UCA-6B)."""

from __future__ import annotations

from datetime import datetime

from intergrax.autonomous_work.capability_acquisition_ports import (
    AllowAllAuthorityCompatibilityPort,
    WorkerCapabilityAuthorityCompatibilityPort,
)
from intergrax.autonomous_work.worker_capability_need_projection import (
    DefaultWorkerCapabilityNeedProjection,
    WorkerCapabilityNeedProjection,
)
from intergrax.autonomous_work.worker_capability_recovery_ports import (
    CanonicalCapabilityDiscoveryPort,
    CanonicalCapabilityDiscoveryRequest,
    CapabilityAcquisitionCoordinatorPort,
    CapabilityQualificationCoordinatorPort,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    ACQUISITION_DECISION_POLICY_VERSION,
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityDiscoveryDisposition,
    ResolvedWorkerCapabilityPolicy,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityAcquisitionRequest,
    WorkerCapabilityAcquisitionResult,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
    WorkerCapabilityDiscoveryResult,
    WorkerCapabilityAuthorityCompatibility,
    autonomy_level_allowed,
    derive_worker_capability_acquisition_decision_id,
    derive_worker_capability_candidate_id,
    derive_worker_capability_need_id,
    operations_allowed_by_policy,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import RecoveryStrategy
from intergrax.contracts.autonomous_work.references import ProblemReference
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
    WorkerCapabilityRecoveryPhase,
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
    derive_capability_acquisition_request_id,
)
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletionOutcome,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
    derive_capability_qualification_request_id,
)


def derive_worker_discovery_correlation_id(worker_need_id: str) -> str:
    return f"aw-canonical-discovery:{worker_need_id}"


def _candidate_kind_for_identity(
    key: CapabilityIdentityKey,
) -> WorkerCapabilityCandidateKind:
    if key.kind is CapabilityKind.TOOL:
        return WorkerCapabilityCandidateKind.TOOL
    if key.kind is CapabilityKind.SKILL:
        return WorkerCapabilityCandidateKind.SKILL
    return WorkerCapabilityCandidateKind.APPROVED_ALTERNATE


def _worker_candidate_from_identity_key(
    *,
    key: CapabilityIdentityKey,
    operations: tuple[str, ...],
    discovered_at: datetime,
    evidence_refs: tuple[ProblemReference, ...],
) -> WorkerCapabilityCandidate:
    capability_ref = (
        f"{key.kind.value}:{key.source_kind.value}:{key.source_id}:{key.logical_id}"
    )
    kind = _candidate_kind_for_identity(key)
    return WorkerCapabilityCandidate(
        candidate_id=derive_worker_capability_candidate_id(
            candidate_kind=kind,
            capability_ref=capability_ref,
        ),
        candidate_kind=kind,
        capability_ref=capability_ref,
        source_domain=key.source_kind.value,
        operations=operations,
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=evidence_refs,
        discovered_at=discovered_at,
    )


class WorkerCapabilityRecoveryCoordinator:
    """Maps worker need → canonical discovery/UCA/qualification → worker recovery."""

    def __init__(
        self,
        *,
        discovery: CanonicalCapabilityDiscoveryPort,
        acquisition: CapabilityAcquisitionCoordinatorPort,
        qualification: CapabilityQualificationCoordinatorPort | None = None,
        need_projection: WorkerCapabilityNeedProjection | None = None,
        authority_compatibility: WorkerCapabilityAuthorityCompatibilityPort
        | None = None,
    ) -> None:
        self._discovery = discovery
        self._acquisition = acquisition
        self._qualification = qualification
        self._need_projection = (
            need_projection or DefaultWorkerCapabilityNeedProjection()
        )
        self._authority_compatibility = (
            authority_compatibility or AllowAllAuthorityCompatibilityPort()
        )

    def coordinate_recovery(
        self,
        request: WorkerCapabilityAcquisitionRequest,
        *,
        decided_at: datetime | None = None,
        allow_generic_acquisition: bool = True,
    ) -> WorkerCapabilityRecoveryOutcome:
        timestamp = decided_at or request.need.requested_at
        worker_need_id = derive_worker_capability_need_id(request.need)
        canonical_need = self._need_projection.project(request.need)
        assert canonical_need.need_id is not None
        correlation_id = derive_worker_discovery_correlation_id(worker_need_id)
        completion = self._discovery.complete_discovery(
            CanonicalCapabilityDiscoveryRequest(
                capability_need=canonical_need,
                worker_need=request.need,
                discovery_correlation_id=correlation_id,
                requested_at=timestamp,
            ),
        )
        provenance = WorkerCapabilityRecoveryProvenance(
            worker_need_id=worker_need_id,
            canonical_need_id=canonical_need.need_id,
            discovery_correlation_id=completion.discovery_correlation_id,
            discovery_completion_outcome=completion.outcome.value,
            evidence_refs=request.need.evidence_refs,
        )
        outcome = completion.outcome
        if outcome is DiscoveryCompletionOutcome.DIRECT_REUSE:
            return WorkerCapabilityRecoveryOutcome(
                phase=WorkerCapabilityRecoveryPhase.DIRECT_REUSE,
                provenance=provenance,
                discovery_completion=completion,
                decided_at=timestamp,
            )
        if outcome is DiscoveryCompletionOutcome.REALIZATION_REQUIRED:
            return WorkerCapabilityRecoveryOutcome(
                phase=WorkerCapabilityRecoveryPhase.REALIZATION_REQUIRED,
                provenance=provenance,
                discovery_completion=completion,
                decided_at=timestamp,
            )
        if outcome is not DiscoveryCompletionOutcome.MISSING_CAPABILITY:
            return WorkerCapabilityRecoveryOutcome(
                phase=WorkerCapabilityRecoveryPhase.FAIL_CLOSED,
                provenance=provenance,
                discovery_completion=completion,
                decided_at=timestamp,
            )

        if not allow_generic_acquisition:
            return WorkerCapabilityRecoveryOutcome(
                phase=WorkerCapabilityRecoveryPhase.FAIL_CLOSED,
                provenance=provenance,
                discovery_completion=completion,
                decided_at=timestamp,
            )

        gap = CapabilityGap.from_discovery_completion(completion)
        provenance = WorkerCapabilityRecoveryProvenance(
            worker_need_id=worker_need_id,
            canonical_need_id=canonical_need.need_id,
            discovery_correlation_id=completion.discovery_correlation_id,
            discovery_completion_outcome=completion.outcome.value,
            gap_id=gap.gap_id,
            evidence_refs=provenance.evidence_refs,
        )
        request_nonce = f"{request.need.recovery_decision_id}:acquire"
        acquisition_request = CapabilityAcquisitionRequest(
            request_id=derive_capability_acquisition_request_id(
                gap_id=gap.gap_id,
                request_nonce=request_nonce,
            ),
            request_nonce=request_nonce,
            capability_gap=gap,
            capability_need=canonical_need,
            correlation_id=correlation_id,
            causation_id=request.need.recovery_decision_id,
            requested_at=timestamp,
        )
        acquisition_result = self._acquisition.acquire(acquisition_request)
        provenance = WorkerCapabilityRecoveryProvenance(
            worker_need_id=worker_need_id,
            canonical_need_id=canonical_need.need_id,
            discovery_correlation_id=completion.discovery_correlation_id,
            discovery_completion_outcome=completion.outcome.value,
            gap_id=gap.gap_id,
            acquisition_request_id=acquisition_result.request_id,
            acquisition_strategy_id=acquisition_result.strategy_id,
            evidence_refs=provenance.evidence_refs
            + (
                ProblemReference(
                    f"uca/acquisition/{acquisition_result.request_id}",
                ),
            ),
        )
        if acquisition_result.outcome is CapabilityAcquisitionOutcome.REQUIRES_HITL:
            return WorkerCapabilityRecoveryOutcome(
                phase=WorkerCapabilityRecoveryPhase.PAUSE_HITL,
                provenance=provenance,
                discovery_completion=completion,
                acquisition_result=acquisition_result,
                decided_at=timestamp,
            )
        if acquisition_result.outcome is not CapabilityAcquisitionOutcome.SUCCEEDED:
            return WorkerCapabilityRecoveryOutcome(
                phase=WorkerCapabilityRecoveryPhase.FAIL_CLOSED,
                provenance=provenance,
                discovery_completion=completion,
                acquisition_result=acquisition_result,
                decided_at=timestamp,
            )

        if self._qualification is None:
            return WorkerCapabilityRecoveryOutcome(
                phase=WorkerCapabilityRecoveryPhase.PENDING_QUALIFICATION,
                provenance=provenance,
                discovery_completion=completion,
                acquisition_result=acquisition_result,
                decided_at=timestamp,
            )

        strategy_id = acquisition_result.strategy_id
        if strategy_id is None:
            return WorkerCapabilityRecoveryOutcome(
                phase=WorkerCapabilityRecoveryPhase.FAIL_CLOSED,
                provenance=provenance,
                discovery_completion=completion,
                acquisition_result=acquisition_result,
                decided_at=timestamp,
            )
        qual_nonce = "qual-1"
        qual_request = CapabilityQualificationRequest(
            qualification_request_id=derive_capability_qualification_request_id(
                acquisition_request_id=acquisition_result.request_id,
                qualification_nonce=qual_nonce,
            ),
            qualification_nonce=qual_nonce,
            acquisition_request_id=acquisition_result.request_id,
            gap_id=gap.gap_id,
            strategy_id=strategy_id,
            acquisition_result=acquisition_result,
            correlation_id=acquisition_result.correlation_id,
            causation_id=acquisition_result.causation_id,
            requested_at=timestamp,
        )
        qualification_result = self._qualification.qualify(qual_request)
        provenance = WorkerCapabilityRecoveryProvenance(
            worker_need_id=worker_need_id,
            canonical_need_id=canonical_need.need_id,
            discovery_correlation_id=completion.discovery_correlation_id,
            discovery_completion_outcome=completion.outcome.value,
            gap_id=gap.gap_id,
            acquisition_request_id=acquisition_result.request_id,
            acquisition_strategy_id=acquisition_result.strategy_id,
            qualification_request_id=qual_request.qualification_request_id,
            evidence_refs=provenance.evidence_refs,
        )
        if qualification_result.outcome is CapabilityQualificationOutcome.QUALIFIED:
            return WorkerCapabilityRecoveryOutcome(
                phase=WorkerCapabilityRecoveryPhase.QUALIFICATION_COMPLETE,
                provenance=provenance,
                discovery_completion=completion,
                acquisition_result=acquisition_result,
                qualification_result=qualification_result,
                decided_at=timestamp,
            )
        return WorkerCapabilityRecoveryOutcome(
            phase=WorkerCapabilityRecoveryPhase.FAIL_CLOSED,
            provenance=provenance,
            discovery_completion=completion,
            acquisition_result=acquisition_result,
            qualification_result=qualification_result,
            decided_at=timestamp,
        )

    def coordinate_acquisition_decision(
        self,
        request: WorkerCapabilityAcquisitionRequest,
        *,
        policy: ResolvedWorkerCapabilityPolicy,
        decided_at: datetime | None = None,
    ) -> WorkerCapabilityAcquisitionResult:
        """Map canonical recovery outcome to AW-7A acquisition decision surface."""
        if (
            request.recovery_decision.strategy
            is not RecoveryStrategy.ACQUIRE_CAPABILITY
        ):
            raise ValueError("canonical coordinator only handles ACQUIRE_CAPABILITY")
        timestamp = decided_at or request.need.requested_at
        if not operations_allowed_by_policy(
            request.need.required_operations,
            policy.allowed_operation_patterns,
        ):
            return _simple_result(
                request=request,
                disposition=CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY,
                reason_code=CapabilityAcquisitionReasonCode.POLICY_BLOCKED,
                decided_at=timestamp,
            )
        recovery = self.coordinate_recovery(
            request,
            decided_at=timestamp,
            allow_generic_acquisition=policy.generated_capability_allowed,
        )
        if recovery.phase is WorkerCapabilityRecoveryPhase.DIRECT_REUSE:
            assert recovery.discovery_completion is not None
            if len(request.need.required_operations) > 1:
                return _simple_result(
                    request=request,
                    disposition=CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY,
                    reason_code=CapabilityAcquisitionReasonCode.NO_SAFE_CANDIDATE,
                    decided_at=timestamp,
                )
            host_key = recovery.discovery_completion.suitable_host_allowed_keys[0]
            candidate = _worker_candidate_from_identity_key(
                key=host_key,
                operations=request.need.required_operations,
                discovered_at=timestamp,
                evidence_refs=request.need.evidence_refs,
            )
            if candidate.candidate_kind not in policy.allowed_candidate_kinds:
                return _simple_result(
                    request=request,
                    disposition=CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY,
                    reason_code=CapabilityAcquisitionReasonCode.POLICY_BLOCKED,
                    decided_at=timestamp,
                )
            if not autonomy_level_allowed(
                WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
                policy.allowed_autonomy_levels,
            ):
                return _simple_result(
                    request=request,
                    disposition=CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY,
                    reason_code=CapabilityAcquisitionReasonCode.POLICY_BLOCKED,
                    decided_at=timestamp,
                )
            compatibility = self._authority_compatibility.assess(
                worker_instance_id=request.need.worker_instance_id,
                candidate=candidate,
            )
            if compatibility is WorkerCapabilityAuthorityCompatibility.UNAVAILABLE:
                return _simple_result(
                    request=request,
                    disposition=CapabilityAcquisitionDisposition.UNAVAILABLE,
                    reason_code=CapabilityAcquisitionReasonCode.DISCOVERY_UNAVAILABLE,
                    decided_at=timestamp,
                )
            if (
                compatibility
                is WorkerCapabilityAuthorityCompatibility.AUTHORITY_CHANGE_REQUIRED
            ):
                return WorkerCapabilityAcquisitionResult(
                    disposition=CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED,
                    decision=_build_decision(
                        request=request,
                        disposition=CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED,
                        reason_code=CapabilityAcquisitionReasonCode.A4_AUTHORITY_CHANGE_REQUIRED,
                        selected_candidate=None,
                        autonomy_level=WorkerAutonomyLevel.A4_AUTHORITY_CHANGE,
                        decided_at=timestamp,
                    ),
                )
            return _decision_result(
                request=request,
                disposition=CapabilityAcquisitionDisposition.USE_EXISTING,
                reason_code=_reason_for_candidate(candidate),
                selected_candidate=candidate,
                decided_at=timestamp,
            )
        if recovery.phase is WorkerCapabilityRecoveryPhase.REALIZATION_REQUIRED:
            return _simple_result(
                request=request,
                disposition=CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY,
                reason_code=CapabilityAcquisitionReasonCode.NO_SAFE_CANDIDATE,
                decided_at=timestamp,
            )
        if recovery.phase is WorkerCapabilityRecoveryPhase.PENDING_QUALIFICATION:
            return _simple_result(
                request=request,
                disposition=CapabilityAcquisitionDisposition.PENDING_QUALIFICATION,
                reason_code=CapabilityAcquisitionReasonCode.CANONICAL_GAP_ACQUIRED_PENDING_QUALIFICATION,
                decided_at=timestamp,
            )
        if recovery.phase is WorkerCapabilityRecoveryPhase.PAUSE_HITL:
            return _simple_result(
                request=request,
                disposition=CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY,
                reason_code=CapabilityAcquisitionReasonCode.POLICY_BLOCKED,
                decided_at=timestamp,
            )
        return _simple_result(
            request=request,
            disposition=CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY,
            reason_code=CapabilityAcquisitionReasonCode.NO_SAFE_CANDIDATE,
            decided_at=timestamp,
        )


def _reason_for_candidate(
    candidate: WorkerCapabilityCandidate,
) -> CapabilityAcquisitionReasonCode:
    if candidate.candidate_kind is WorkerCapabilityCandidateKind.TOOL:
        return CapabilityAcquisitionReasonCode.EXISTING_TOOL_SELECTED
    if candidate.candidate_kind is WorkerCapabilityCandidateKind.SKILL:
        return CapabilityAcquisitionReasonCode.EXISTING_SKILL_SELECTED
    return CapabilityAcquisitionReasonCode.APPROVED_ALTERNATE_SELECTED


def _simple_result(
    *,
    request: WorkerCapabilityAcquisitionRequest,
    disposition: CapabilityAcquisitionDisposition,
    reason_code: CapabilityAcquisitionReasonCode,
    decided_at: datetime,
) -> WorkerCapabilityAcquisitionResult:
    return WorkerCapabilityAcquisitionResult(
        disposition=disposition,
        decision=_build_decision(
            request=request,
            disposition=disposition,
            reason_code=reason_code,
            selected_candidate=None,
            autonomy_level=None,
            decided_at=decided_at,
        ),
    )


def _decision_result(
    *,
    request: WorkerCapabilityAcquisitionRequest,
    disposition: CapabilityAcquisitionDisposition,
    reason_code: CapabilityAcquisitionReasonCode,
    selected_candidate: WorkerCapabilityCandidate,
    decided_at: datetime,
) -> WorkerCapabilityAcquisitionResult:
    decision = _build_decision(
        request=request,
        disposition=disposition,
        reason_code=reason_code,
        selected_candidate=selected_candidate,
        autonomy_level=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        decided_at=decided_at,
    )
    discovery = WorkerCapabilityDiscoveryResult(
        need=request.need,
        candidates=(selected_candidate,),
        disposition=CapabilityDiscoveryDisposition.MATCH_FOUND,
        profile_ref=request.capability_profile_ref,
        discovered_at=decided_at,
    )
    return WorkerCapabilityAcquisitionResult(
        disposition=disposition,
        decision=decision,
        discovery=discovery,
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
    selected_id = (
        selected_candidate.candidate_id if selected_candidate is not None else None
    )
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
        codecraft_profile_ref=request.codecraft_profile_ref
        or need.codecraft_profile_ref,
        reason_code=reason_code,
        evidence_refs=evidence_refs,
        decided_at=decided_at,
        decision_policy_version=ACQUISITION_DECISION_POLICY_VERSION,
    )


__all__ = [
    "WorkerCapabilityRecoveryCoordinator",
    "derive_worker_discovery_correlation_id",
]
