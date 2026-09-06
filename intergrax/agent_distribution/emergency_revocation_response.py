# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Active runtime emergency revocation response (AC-6 Phase 4)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.activation import ActivationService
from intergrax.agent_distribution.effective_roster_authority import (
    EffectiveRosterAuthorityService,
)
from intergrax.agent_distribution.errors import (
    AgentDistributionNotFoundError,
    AgentPackageTrustError,
    EffectiveRosterAuthorityNotFound,
    RuntimeActivationConflict,
    RuntimeDrainError,
    RuntimeRollbackError,
)
from intergrax.agent_distribution.events import AgentDistributionEvent, distribution_event
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.package_trust import AgentPackageTrustCoordinator
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.agent_distribution.stores import (
    AgentInstallationStore,
    ApplicationEnvironmentServingRecord,
    ApplicationEnvironmentServingStore,
    RuntimeRevisionStore,
)
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentPackageTrustPolicy,
    AgentPackageTrustRevocationState,
    require_timezone_aware_utc_datetime,
)

_NON_EMPTY = Field(min_length=1)

_ACTIVE_SECURITY_REVOCATION_SCOPES = frozenset(
    {
        "revoked_package_digests",
        "revoked_publisher_ids",
        "revoked_evidence_ids",
    }
)


class ActiveRuntimeTrustImpactReasonCode(StrEnum):
    """Why an enabled package in the active serving revision is security-unsafe."""

    PACKAGE_DIGEST_REVOKED = "package_digest_revoked"
    PUBLISHER_REVOKED = "publisher_revoked"
    EVIDENCE_REVOKED = "evidence_revoked"
    ACTIVE_TRUST_EVIDENCE_UNAVAILABLE = "active_trust_evidence_unavailable"


class ActiveRuntimeTrustImpact(BaseModel):
    """Canonical evidence that active serving revision contains revoked trust."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    runtime_revision_id: str = _NON_EMPTY
    installation_id: str = _NON_EMPTY
    installation_slot_id: str = _NON_EMPTY
    package_identity: AgentPackageIdentity
    reason_code: ActiveRuntimeTrustImpactReasonCode


class EmergencyTrustResponseAction(StrEnum):
    """Explicit emergency response outcome — no hidden kill-switch."""

    NO_ACTION = "no_action"
    ROLLBACK = "rollback"
    BLOCKED_NO_SAFE_TARGET = "blocked_no_safe_target"


class EmergencyTrustResponseReasonCode(StrEnum):
    """Stable machine-readable emergency response codes."""

    NO_ACTIVE_REVOCATION = "no_active_revocation"
    ACTIVE_PACKAGE_REVOKED = "active_package_revoked"
    ACTIVE_PUBLISHER_REVOKED = "active_publisher_revoked"
    ACTIVE_EVIDENCE_REVOKED = "active_evidence_revoked"
    SAFE_ROLLBACK_COMPLETED = "safe_rollback_completed"
    ROLLBACK_TARGET_UNTRUSTED = "rollback_target_untrusted"
    NO_PRIOR_REVISION = "no_prior_revision"
    ROLLBACK_CONFLICT = "rollback_conflict"
    ROLLBACK_FAILED = "rollback_failed"
    DRAIN_RECOVERY_REQUIRED = "drain_recovery_required"
    ACTIVE_TRUST_EVIDENCE_UNAVAILABLE = "active_trust_evidence_unavailable"
    SERVING_POINTER_MISMATCH = "serving_pointer_mismatch"
    NO_SERVING_REVISION = "no_serving_revision"


_IMPACT_TO_RESPONSE_REASON: dict[
    ActiveRuntimeTrustImpactReasonCode, EmergencyTrustResponseReasonCode
] = {
    ActiveRuntimeTrustImpactReasonCode.PACKAGE_DIGEST_REVOKED: (
        EmergencyTrustResponseReasonCode.ACTIVE_PACKAGE_REVOKED
    ),
    ActiveRuntimeTrustImpactReasonCode.PUBLISHER_REVOKED: (
        EmergencyTrustResponseReasonCode.ACTIVE_PUBLISHER_REVOKED
    ),
    ActiveRuntimeTrustImpactReasonCode.EVIDENCE_REVOKED: (
        EmergencyTrustResponseReasonCode.ACTIVE_EVIDENCE_REVOKED
    ),
    ActiveRuntimeTrustImpactReasonCode.ACTIVE_TRUST_EVIDENCE_UNAVAILABLE: (
        EmergencyTrustResponseReasonCode.ACTIVE_TRUST_EVIDENCE_UNAVAILABLE
    ),
}


class AgentEmergencyRevocationRequest(BaseModel):
    """One-shot emergency response invocation with immutable revocation snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    expected_current_traffic_revision_id: str | None = None
    expected_serving_pointer_revision: int | None = None
    evaluated_at: datetime
    revocation_state: AgentPackageTrustRevocationState
    trust_policy: AgentPackageTrustPolicy | None = None

    @field_validator("evaluated_at")
    @classmethod
    def _validate_evaluated_at(cls, value: datetime) -> datetime:
        return require_timezone_aware_utc_datetime(value, field_name="evaluated_at")


class AgentEmergencyRevocationResponse(BaseModel):
    """Immutable emergency response outcome for audit and operator action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    observed_serving_revision_id: str | None = None
    action: EmergencyTrustResponseAction
    affected_packages: tuple[AgentPackageIdentity, ...] = ()
    impacts: tuple[ActiveRuntimeTrustImpact, ...] = ()
    rollback_target_revision_id: str | None = None
    final_serving_revision_id: str | None = None
    response_reason_code: EmergencyTrustResponseReasonCode
    audit_event_types: tuple[str, ...] = ()


def scan_active_security_revocation_impacts(
    *,
    revision: RuntimeRevision,
    roster: EffectiveRoster,
    installation_store: AgentInstallationStore,
    revocation_state: AgentPackageTrustRevocationState,
) -> tuple[ActiveRuntimeTrustImpact, ...]:
    """Determine whether the serving revision contains security-revoked packages."""
    impacts: list[ActiveRuntimeTrustImpact] = []
    for entry in roster.entries:
        if not entry.effective_enablement:
            continue
        impact = _impact_for_enabled_entry(
            revision=revision,
            entry=entry,
            installation_store=installation_store,
            revocation_state=revocation_state,
        )
        if impact is not None:
            impacts.append(impact)
    return tuple(impacts)


def revision_passes_current_trust_admission(
    *,
    revision: RuntimeRevision,
    roster: EffectiveRoster,
    installation_store: AgentInstallationStore,
    trust_coordinator: AgentPackageTrustCoordinator,
    revocation_state: AgentPackageTrustRevocationState,
    policy: AgentPackageTrustPolicy,
    evaluated_at: datetime,
) -> bool:
    """Re-evaluate every enabled roster entry against current trust admission policy."""
    for entry in roster.entries:
        if not entry.effective_enablement:
            continue
        installation = _require_enabled_installation(
            entry=entry,
            installation_store=installation_store,
        )
        if installation.trust_record is None:
            return False
        try:
            trust_coordinator.assert_install_admission(
                trust_record=installation.trust_record,
                package_identity=installation.package_identity,
                revocation_state=revocation_state,
                policy=policy,
                evaluated_at=evaluated_at,
            )
        except AgentPackageTrustError:
            return False
    return True


class AgentEmergencyRevocationService:
    """Orchestrates active-revocation impact scan and canonical rollback — read-only stores."""

    def __init__(
        self,
        *,
        serving_store: ApplicationEnvironmentServingStore,
        revision_store: RuntimeRevisionStore,
        effective_roster_authority: EffectiveRosterAuthorityService,
        installation_store: AgentInstallationStore,
        package_trust_coordinator: AgentPackageTrustCoordinator,
        activation_service: ActivationService,
    ) -> None:
        self._serving_store = serving_store
        self._revision_store = revision_store
        self._effective_roster_authority = effective_roster_authority
        self._installation_store = installation_store
        self._package_trust_coordinator = package_trust_coordinator
        self._activation_service = activation_service

    def respond_to_current_revocation(
        self,
        request: AgentEmergencyRevocationRequest,
    ) -> AgentEmergencyRevocationResponse:
        """Evaluate active serving trust impact once and invoke canonical rollback if safe."""
        serving = self._serving_store.get_serving_record(
            request.application_id,
            request.application_environment_id,
        )
        if serving is None or serving.traffic_serving_revision_id is None:
            return self._response(
                request=request,
                serving=serving,
                action=EmergencyTrustResponseAction.NO_ACTION,
                reason=EmergencyTrustResponseReasonCode.NO_SERVING_REVISION,
            )

        active_revision = self._revision_store.get_revision(serving.traffic_serving_revision_id)
        if active_revision is None:
            raise AgentDistributionNotFoundError(
                f"active serving revision {serving.traffic_serving_revision_id} was not found"
            )

        try:
            active_roster = self._effective_roster_authority.require_for_revision(active_revision)
        except EffectiveRosterAuthorityNotFound as exc:
            raise EffectiveRosterAuthorityNotFound(
                "AC-6-PHASE-4-ACTIVE-HISTORICAL-AUTHORITY-GAP: "
                f"{exc}"
            ) from exc

        impacts = scan_active_security_revocation_impacts(
            revision=active_revision,
            roster=active_roster,
            installation_store=self._installation_store,
            revocation_state=request.revocation_state,
        )
        if not impacts:
            return self._response(
                request=request,
                serving=serving,
                action=EmergencyTrustResponseAction.NO_ACTION,
                reason=EmergencyTrustResponseReasonCode.NO_ACTIVE_REVOCATION,
            )

        pointer_mismatch = self._serving_pointer_mismatch(request, serving)
        if pointer_mismatch is not None:
            return pointer_mismatch

        events = (
            distribution_event(
                "agent_trust.active_revocation_detected",
                request.application_environment_id,
                runtime_revision_id=active_revision.runtime_revision_id,
                impact_count=str(len(impacts)),
            ),
        )
        affected = tuple(impact.package_identity for impact in impacts)

        if any(
            impact.reason_code
            is ActiveRuntimeTrustImpactReasonCode.ACTIVE_TRUST_EVIDENCE_UNAVAILABLE
            for impact in impacts
        ):
            return self._response(
                request=request,
                serving=serving,
                action=EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET,
                reason=EmergencyTrustResponseReasonCode.ACTIVE_TRUST_EVIDENCE_UNAVAILABLE,
                impacts=impacts,
                affected_packages=affected,
                audit_events=events
                + (
                    distribution_event(
                        "agent_trust.emergency_response_blocked",
                        request.application_environment_id,
                        reason_code=(
                            EmergencyTrustResponseReasonCode.ACTIVE_TRUST_EVIDENCE_UNAVAILABLE.value
                        ),
                    ),
                ),
            )

        if serving.prior_traffic_revision_id is None:
            return self._response(
                request=request,
                serving=serving,
                action=EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET,
                reason=EmergencyTrustResponseReasonCode.NO_PRIOR_REVISION,
                impacts=impacts,
                affected_packages=affected,
                audit_events=events
                + (
                    distribution_event(
                        "agent_trust.emergency_response_blocked",
                        request.application_environment_id,
                        reason_code=EmergencyTrustResponseReasonCode.NO_PRIOR_REVISION.value,
                    ),
                ),
            )

        target_revision_id = serving.prior_traffic_revision_id
        target_revision = self._revision_store.get_revision(target_revision_id)
        if target_revision is None:
            return self._blocked_target_response(
                request=request,
                serving=serving,
                reason=EmergencyTrustResponseReasonCode.ROLLBACK_TARGET_UNTRUSTED,
                impacts=impacts,
                affected_packages=affected,
                target_revision_id=target_revision_id,
                audit_events=events,
            )

        try:
            target_roster = self._effective_roster_authority.require_for_revision(
                target_revision
            )
        except EffectiveRosterAuthorityNotFound:
            return self._blocked_target_response(
                request=request,
                serving=serving,
                reason=EmergencyTrustResponseReasonCode.ROLLBACK_TARGET_UNTRUSTED,
                impacts=impacts,
                affected_packages=affected,
                target_revision_id=target_revision_id,
                audit_events=events,
            )

        policy = request.trust_policy or AgentPackageTrustPolicy()
        if not revision_passes_current_trust_admission(
            revision=target_revision,
            roster=target_roster,
            installation_store=self._installation_store,
            trust_coordinator=self._package_trust_coordinator,
            revocation_state=request.revocation_state,
            policy=policy,
            evaluated_at=request.evaluated_at,
        ):
            return self._blocked_target_response(
                request=request,
                serving=serving,
                reason=EmergencyTrustResponseReasonCode.ROLLBACK_TARGET_UNTRUSTED,
                impacts=impacts,
                affected_packages=affected,
                target_revision_id=target_revision_id,
                audit_events=events,
            )

        expected_revision = (
            request.expected_current_traffic_revision_id
            or serving.traffic_serving_revision_id
        )
        expected_pointer = (
            request.expected_serving_pointer_revision
            if request.expected_serving_pointer_revision is not None
            else serving.serving_pointer_revision
        )
        try:
            rolled = self._activation_service.rollback(
                application_id=request.application_id,
                application_environment_id=request.application_environment_id,
                expected_current_traffic_revision_id=expected_revision,
                expected_serving_pointer_revision=expected_pointer,
            )
            return self._response(
                request=request,
                serving=rolled.value.serving_record,
                action=EmergencyTrustResponseAction.ROLLBACK,
                reason=EmergencyTrustResponseReasonCode.SAFE_ROLLBACK_COMPLETED,
                impacts=impacts,
                affected_packages=affected,
                rollback_target_revision_id=target_revision_id,
                audit_events=events
                + (
                    distribution_event(
                        "agent_trust.emergency_rollback_completed",
                        request.application_environment_id,
                        restored_revision_id=target_revision_id,
                        superseded_revision_id=active_revision.runtime_revision_id,
                    ),
                ),
            )
        except RuntimeDrainError:
            serving_after = self._serving_store.get_serving_record(
                request.application_id,
                request.application_environment_id,
            )
            return self._response(
                request=request,
                serving=serving_after,
                action=EmergencyTrustResponseAction.ROLLBACK,
                reason=EmergencyTrustResponseReasonCode.DRAIN_RECOVERY_REQUIRED,
                impacts=impacts,
                affected_packages=affected,
                rollback_target_revision_id=target_revision_id,
                audit_events=events
                + (
                    distribution_event(
                        "agent_trust.emergency_rollback_completed",
                        request.application_environment_id,
                        restored_revision_id=target_revision_id,
                        superseded_revision_id=active_revision.runtime_revision_id,
                    ),
                ),
            )
        except RuntimeActivationConflict:
            return self._response(
                request=request,
                serving=serving,
                action=EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET,
                reason=EmergencyTrustResponseReasonCode.ROLLBACK_CONFLICT,
                impacts=impacts,
                affected_packages=affected,
                rollback_target_revision_id=target_revision_id,
                audit_events=events
                + (
                    distribution_event(
                        "agent_trust.emergency_response_blocked",
                        request.application_environment_id,
                        reason_code=EmergencyTrustResponseReasonCode.ROLLBACK_CONFLICT.value,
                    ),
                ),
            )
        except RuntimeRollbackError:
            return self._response(
                request=request,
                serving=serving,
                action=EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET,
                reason=EmergencyTrustResponseReasonCode.ROLLBACK_FAILED,
                impacts=impacts,
                affected_packages=affected,
                rollback_target_revision_id=target_revision_id,
                audit_events=events
                + (
                    distribution_event(
                        "agent_trust.emergency_response_blocked",
                        request.application_environment_id,
                        reason_code=EmergencyTrustResponseReasonCode.ROLLBACK_FAILED.value,
                    ),
                ),
            )

    def _serving_pointer_mismatch(
        self,
        request: AgentEmergencyRevocationRequest,
        serving: ApplicationEnvironmentServingRecord,
    ) -> AgentEmergencyRevocationResponse | None:
        if (
            request.expected_current_traffic_revision_id is not None
            and serving.traffic_serving_revision_id
            != request.expected_current_traffic_revision_id
        ):
            return self._response(
                request=request,
                serving=serving,
                action=EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET,
                reason=EmergencyTrustResponseReasonCode.SERVING_POINTER_MISMATCH,
            )
        if (
            request.expected_serving_pointer_revision is not None
            and serving.serving_pointer_revision
            != request.expected_serving_pointer_revision
        ):
            return self._response(
                request=request,
                serving=serving,
                action=EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET,
                reason=EmergencyTrustResponseReasonCode.ROLLBACK_CONFLICT,
            )
        return None

    def _blocked_target_response(
        self,
        *,
        request: AgentEmergencyRevocationRequest,
        serving: ApplicationEnvironmentServingRecord,
        reason: EmergencyTrustResponseReasonCode,
        impacts: tuple[ActiveRuntimeTrustImpact, ...],
        affected_packages: tuple[AgentPackageIdentity, ...],
        target_revision_id: str,
        audit_events: tuple[AgentDistributionEvent, ...],
        response_reason: EmergencyTrustResponseReasonCode | None = None,
    ) -> AgentEmergencyRevocationResponse:
        return self._response(
            request=request,
            serving=serving,
            action=EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET,
            reason=response_reason or reason,
            impacts=impacts,
            affected_packages=affected_packages,
            rollback_target_revision_id=target_revision_id,
            audit_events=audit_events
            + (
                distribution_event(
                    "agent_trust.emergency_response_blocked",
                    request.application_environment_id,
                    reason_code=reason.value,
                    rollback_target_revision_id=target_revision_id,
                ),
            ),
        )

    @staticmethod
    def _response(
        *,
        request: AgentEmergencyRevocationRequest,
        serving: ApplicationEnvironmentServingRecord | None,
        action: EmergencyTrustResponseAction,
        reason: EmergencyTrustResponseReasonCode,
        impacts: tuple[ActiveRuntimeTrustImpact, ...] = (),
        affected_packages: tuple[AgentPackageIdentity, ...] = (),
        rollback_target_revision_id: str | None = None,
        audit_events: tuple[AgentDistributionEvent, ...] = (),
    ) -> AgentEmergencyRevocationResponse:
        final_revision_id = (
            serving.traffic_serving_revision_id if serving is not None else None
        )
        return AgentEmergencyRevocationResponse(
            application_id=request.application_id,
            application_environment_id=request.application_environment_id,
            observed_serving_revision_id=final_revision_id,
            action=action,
            affected_packages=affected_packages,
            impacts=impacts,
            rollback_target_revision_id=rollback_target_revision_id,
            final_serving_revision_id=final_revision_id,
            response_reason_code=reason,
            audit_event_types=tuple(event.event_type for event in audit_events),
        )


def _impact_for_enabled_entry(
    *,
    revision: RuntimeRevision,
    entry: EffectiveRosterEntry,
    installation_store: AgentInstallationStore,
    revocation_state: AgentPackageTrustRevocationState,
) -> ActiveRuntimeTrustImpact | None:
    package_identity = AgentPackageIdentity(
        distribution_package_id=entry.distribution_package_id,
        package_version="0.0.0",
        package_digest=entry.package_digest,
    )
    installation_id = entry.active_installation_id or "missing-installation"
    if entry.package_digest in revocation_state.revoked_package_digests:
        return ActiveRuntimeTrustImpact(
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            runtime_revision_id=revision.runtime_revision_id,
            installation_id=installation_id,
            installation_slot_id=entry.installation_slot_id,
            package_identity=package_identity,
            reason_code=ActiveRuntimeTrustImpactReasonCode.PACKAGE_DIGEST_REVOKED,
        )

    if entry.active_installation_id is None:
        return ActiveRuntimeTrustImpact(
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            runtime_revision_id=revision.runtime_revision_id,
            installation_id=installation_id,
            installation_slot_id=entry.installation_slot_id,
            package_identity=package_identity,
            reason_code=ActiveRuntimeTrustImpactReasonCode.ACTIVE_TRUST_EVIDENCE_UNAVAILABLE,
        )

    installation = installation_store.get_installation(entry.active_installation_id)
    if installation is None or installation.trust_record is None:
        return ActiveRuntimeTrustImpact(
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            runtime_revision_id=revision.runtime_revision_id,
            installation_id=installation_id,
            installation_slot_id=entry.installation_slot_id,
            package_identity=installation.package_identity
            if installation is not None
            else package_identity,
            reason_code=ActiveRuntimeTrustImpactReasonCode.ACTIVE_TRUST_EVIDENCE_UNAVAILABLE,
        )

    trust_impact = _trust_record_security_impact(
        trust_record=installation.trust_record,
        revocation_state=revocation_state,
    )
    if trust_impact is None:
        return None
    return ActiveRuntimeTrustImpact(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        installation_id=installation.installation_id,
        installation_slot_id=entry.installation_slot_id,
        package_identity=installation.package_identity,
        reason_code=trust_impact,
    )


def _trust_record_security_impact(
    *,
    trust_record: AgentInstallationTrustRecord,
    revocation_state: AgentPackageTrustRevocationState,
) -> ActiveRuntimeTrustImpactReasonCode | None:
    if trust_record.package_digest in revocation_state.revoked_package_digests:
        return ActiveRuntimeTrustImpactReasonCode.PACKAGE_DIGEST_REVOKED
    if trust_record.publisher_identity_ref in revocation_state.revoked_publisher_ids:
        return ActiveRuntimeTrustImpactReasonCode.PUBLISHER_REVOKED
    for evidence_ref in trust_record.trust_evidence_refs:
        if evidence_ref.evidence_id in revocation_state.revoked_evidence_ids:
            return ActiveRuntimeTrustImpactReasonCode.EVIDENCE_REVOKED
    return None


def _require_enabled_installation(
    *,
    entry: EffectiveRosterEntry,
    installation_store: AgentInstallationStore,
):
    if entry.active_installation_id is None:
        raise AgentPackageTrustError("rollback target roster entry lacks installation")
    installation = installation_store.get_installation(entry.active_installation_id)
    if installation is None:
        raise AgentPackageTrustError("rollback target installation was not found")
    return installation


__all__ = [
    "ActiveRuntimeTrustImpact",
    "ActiveRuntimeTrustImpactReasonCode",
    "AgentEmergencyRevocationRequest",
    "AgentEmergencyRevocationResponse",
    "AgentEmergencyRevocationService",
    "EmergencyTrustResponseAction",
    "EmergencyTrustResponseReasonCode",
    "_ACTIVE_SECURITY_REVOCATION_SCOPES",
    "revision_passes_current_trust_admission",
    "scan_active_security_revocation_impacts",
]
