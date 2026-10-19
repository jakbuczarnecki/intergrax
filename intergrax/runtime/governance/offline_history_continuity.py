# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Offline Spanner primary plus Azure witness behind ``HistoryContinuityAuthority``.

One authority. The platform contract is unchanged. Cloud sessions are not
opened. Recovery never mints an authority incarnation or promotes a rolled-back
head to current.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from intergrax.agent_distribution.control_plane_governance import (
    ApplicationEnvironmentTenantResolver,
)
from intergrax.contracts.agent_run import (
    RequestIdentity,
    canonical_principal_id_from_request_identity,
)
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
)
from intergrax.contracts.governance.history_continuity import (
    ContinuityDenialCode,
    ContinuityHead,
    ContinuityHeadStatus,
    ContinuityOutcome,
    ContinuityPermission,
    ContinuityPermissionGrant,
    ContinuityResult,
    ContinuityScope,
    HistoryBinding,
    HistoryContinuityProviderSelection,
    HistoryContinuityRetentionPolicy,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.continuity_providers.errors import (
    ProviderFault,
    map_provider_fault,
)
from intergrax.runtime.governance.continuity_providers.primary_register import (
    HeadRecord,
    InMemoryPrimaryDatabase,
    PrimaryCommitKind,
    PrimaryStep,
    SpannerPrimaryRegister,
    TransitionRecord,
    request_digest,
)
from intergrax.runtime.governance.continuity_providers.targets import (
    AzureWitnessTarget,
    SpannerPrimaryTarget,
)
from intergrax.runtime.governance.continuity_providers.witness_log import (
    AppendKind,
    AzureAppendWitness,
    InMemoryAppendWitness,
    WitnessView,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.local_history_continuity import (
    continuity_authorization_action,
    continuity_binding_digest,
)

_COMPOSITION_TOKEN = object()
_READ_PERMISSIONS = frozenset(ContinuityPermission)
_COMMITTED = "committed"
_PREPARED = "prepared"
_WITNESS_SEALED = "witness_sealed"
_ACTIONS = {
    ContinuityPermission.WRITE: "history_continuity.write",
    ContinuityPermission.INITIALIZE: "history_continuity.initialize",
    ContinuityPermission.RECONCILE: "history_continuity.reconcile",
    ContinuityPermission.INVALIDATE: "history_continuity.invalidate",
    ContinuityPermission.RECOVER: "history_continuity.recover",
}


@dataclass(frozen=True, slots=True)
class OfflineProviderBundle:
    """Injected offline transports and unqualified targets. Not a live session."""

    primary: InMemoryPrimaryDatabase
    witness: InMemoryAppendWitness
    primary_target: SpannerPrimaryTarget
    witness_target: AzureWitnessTarget
    segment_block_limit: int = 49_000
    recovery_block_bound: int = 128


@dataclass(frozen=True, slots=True)
class _Command:
    revision: int
    fencing_token: str
    binding: HistoryBinding
    status: ContinuityHeadStatus
    identity_epoch: int
    tenant_id: str
    expected_revision: int
    expected_fencing_token: str | None


class OfflineSpannerAzureHistoryContinuity:
    """ELCR-IIRW authority over an offline primary register and append witness."""

    def __init__(
        self,
        *,
        tenant_resolver: ApplicationEnvironmentTenantResolver,
        selection: HistoryContinuityProviderSelection,
        authorization_boundary: ControlPlaneMutationAuthorizationBoundary,
        principal: RequestIdentity,
        issuer_principal_id: str,
        bundle: OfflineProviderBundle,
        composition_token: object,
    ) -> None:
        if composition_token is not _COMPOSITION_TOKEN:
            raise RuntimeError("history continuity requires sanctioned composition")
        if canonical_principal_id_from_request_identity(principal) != issuer_principal_id:
            raise RuntimeError("continuity issuer does not match governance principal")
        self._tenant_resolver = tenant_resolver
        self._selection = selection
        self._authorization_boundary = authorization_boundary
        self._principal = principal
        self._issuer = issuer_principal_id
        self._retention = selection.retention
        self._bundle = bundle
        self._register = SpannerPrimaryRegister()
        self._witness = AzureAppendWitness(
            bundle.witness,
            segment_block_limit=bundle.segment_block_limit,
            recovery_block_bound=bundle.recovery_block_bound,
        )
        self._lock = threading.Lock()

    def sealed_retention(self) -> HistoryContinuityRetentionPolicy | None:
        return self._retention

    def initialize(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        binding: HistoryBinding,
    ) -> ContinuityResult:
        with self._lock:
            denied = self._precheck(grant, scope, ContinuityPermission.INITIALIZE)
            if denied is not None:
                return denied
            head = self._head(scope.tenant_id)
            if head is not None or self._phase(scope.tenant_id) is not None:
                return self._deny(ContinuityDenialCode.ALREADY_INITIALIZED, scope.tenant_id)
            return self._execute(
                scope,
                _Command(
                    revision=1,
                    fencing_token="1",
                    binding=binding,
                    status=ContinuityHeadStatus.ACTIVE,
                    identity_epoch=1,
                    tenant_id=scope.tenant_id,
                    expected_revision=0,
                    expected_fencing_token=None,
                ),
            )

    def advance(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        expected_revision: int,
        expected_fencing_token: str,
        binding: HistoryBinding,
    ) -> ContinuityResult:
        with self._lock:
            denied = self._precheck(grant, scope, ContinuityPermission.WRITE)
            if denied is not None:
                return denied
            blocked = self._require_mutable(scope.tenant_id)
            if blocked is not None:
                return blocked
            compared = self._compare(scope.tenant_id, expected_revision, expected_fencing_token)
            if compared is not None:
                return compared
            head = self._require_record(scope.tenant_id)
            if not _binding_moves_forward(self._binding_of(head), binding, allow_rebind=False):
                return self._deny(ContinuityDenialCode.CONTINUITY_REVISION_STALE, scope.tenant_id)
            return self._execute(
                scope,
                _Command(
                    revision=head.revision + 1,
                    fencing_token=head.fencing_token,
                    binding=binding,
                    status=ContinuityHeadStatus.ACTIVE,
                    identity_epoch=head.identity_epoch,
                    tenant_id=scope.tenant_id,
                    expected_revision=head.revision,
                    expected_fencing_token=head.fencing_token,
                ),
            )

    def reconcile(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        expected_revision: int,
        expected_fencing_token: str,
        binding: HistoryBinding,
    ) -> ContinuityResult:
        with self._lock:
            denied = self._precheck(grant, scope, ContinuityPermission.RECONCILE)
            if denied is not None:
                return denied
            blocked = self._require_present(scope.tenant_id)
            if blocked is not None:
                return blocked
            head = self._require_record(scope.tenant_id)
            if head.status == ContinuityHeadStatus.REVOKED.value:
                return self._deny(ContinuityDenialCode.REVOKED, scope.tenant_id)
            if head.status != ContinuityHeadStatus.QUARANTINED.value:
                return self._deny(ContinuityDenialCode.RECONCILIATION_UNAVAILABLE, scope.tenant_id)
            compared = self._compare(scope.tenant_id, expected_revision, expected_fencing_token)
            if compared is not None:
                return compared
            if not _binding_moves_forward(self._binding_of(head), binding, allow_rebind=True):
                return self._deny(ContinuityDenialCode.CONTINUITY_REVISION_STALE, scope.tenant_id)
            diverged = self._divergence(scope.tenant_id)
            if diverged is not None:
                return diverged
            return self._execute(
                scope,
                _Command(
                    revision=head.revision + 1,
                    fencing_token=_next_fence(head.fencing_token),
                    binding=binding,
                    status=ContinuityHeadStatus.ACTIVE,
                    identity_epoch=head.identity_epoch + 1,
                    tenant_id=scope.tenant_id,
                    expected_revision=head.revision,
                    expected_fencing_token=head.fencing_token,
                ),
            )

    def invalidate(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        expected_revision: int,
        expected_fencing_token: str,
        status: ContinuityHeadStatus,
    ) -> ContinuityResult:
        with self._lock:
            denied = self._precheck(grant, scope, ContinuityPermission.INVALIDATE)
            if denied is not None:
                return denied
            if status not in (ContinuityHeadStatus.QUARANTINED, ContinuityHeadStatus.REVOKED):
                return self._deny(ContinuityDenialCode.PERMISSION_DENIED, scope.tenant_id)
            blocked = self._require_present(scope.tenant_id)
            if blocked is not None:
                return blocked
            head = self._require_record(scope.tenant_id)
            if head.status == ContinuityHeadStatus.REVOKED.value:
                return self._deny(ContinuityDenialCode.REVOKED, scope.tenant_id)
            compared = self._compare(scope.tenant_id, expected_revision, expected_fencing_token)
            if compared is not None:
                return compared
            diverged = self._divergence(scope.tenant_id)
            if diverged is not None:
                return diverged
            return self._execute(
                scope,
                _Command(
                    revision=head.revision + 1,
                    fencing_token=_next_fence(head.fencing_token),
                    binding=self._binding_of(head),
                    status=status,
                    identity_epoch=head.identity_epoch,
                    tenant_id=scope.tenant_id,
                    expected_revision=head.revision,
                    expected_fencing_token=head.fencing_token,
                ),
            )

    def recover(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
    ) -> ContinuityResult:
        with self._lock:
            denied = self._precheck(grant, scope, ContinuityPermission.RECOVER)
            if denied is not None:
                return denied
            phase = self._phase(scope.tenant_id)
            if phase in {_PREPARED, _WITNESS_SEALED}:
                return self._recover_pending(scope)
            chain = self._witness.verify_bounded(full_chain=True)
            if chain.fault is not None:
                return self._deny(map_provider_fault(chain.fault), scope.tenant_id)
            diverged = self._divergence(scope.tenant_id)
            if diverged is not None:
                return diverged
            head = self._head(scope.tenant_id)
            if head is None:
                return self._result(ContinuityOutcome.RECOVERED, tenant_id=scope.tenant_id, head=None)
            return self._result(
                ContinuityOutcome.RECOVERED,
                tenant_id=scope.tenant_id,
                head=self._public(head),
            )

    def read_head(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
    ) -> ContinuityResult:
        with self._lock:
            denied = self._precheck(grant, scope, _READ_PERMISSIONS)
            if denied is not None:
                return denied
            phase = self._phase(scope.tenant_id)
            if phase is not None and phase != _COMMITTED:
                return self._deny(ContinuityDenialCode.TRANSITION_UNRESOLVED, scope.tenant_id)
            head = self._head(scope.tenant_id)
            if head is None:
                return self._deny(ContinuityDenialCode.NOT_INITIALIZED, scope.tenant_id)
            diverged = self._divergence(scope.tenant_id)
            if diverged is not None:
                return diverged
            return self._result(
                ContinuityOutcome.COMMITTED,
                tenant_id=scope.tenant_id,
                head=self._public(head),
            )

    def attempt_retention_change(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        retention_seconds: int,
        ceiling_seconds: int,
        legal_hold: bool,
    ) -> ContinuityResult:
        with self._lock:
            scope_denial = self._check_scope(scope)
            if scope_denial is not None:
                return self._deny(scope_denial, scope.tenant_id)
            if grant.principal_id != self._issuer:
                return self._deny(ContinuityDenialCode.PERMISSION_DENIED, scope.tenant_id)
            weaker = (
                retention_seconds < self._retention.retention_seconds
                or ceiling_seconds < self._retention.ceiling_seconds
                or legal_hold is not True
            )
            if weaker:
                return self._deny(
                    ContinuityDenialCode.RETENTION_WEAKENING_FORBIDDEN,
                    scope.tenant_id,
                )
            return self._deny(ContinuityDenialCode.PERMISSION_DENIED, scope.tenant_id)

    def _execute(self, scope: ContinuityScope, command: _Command) -> ContinuityResult:
        transition = self._transition(command)
        incarnation = self._incarnation(scope.tenant_id)
        prepared = self._register.prepare(
            self._bundle.primary,
            application_id=scope.application_id,
            environment_id=scope.environment_id,
            tenant_id=scope.tenant_id,
            expected_revision=command.expected_revision,
            expected_fencing_token=command.expected_fencing_token,
            transition=transition,
            authority_incarnation=incarnation,
        )
        if prepared.kind is not PrimaryCommitKind.COMMITTED:
            return self._from_step(prepared, scope.tenant_id)
        appended = self._witness.append_seal(
            application_id=scope.application_id,
            environment_id=scope.environment_id,
            tenant_id=scope.tenant_id,
            revision=command.revision,
            fencing_token=command.fencing_token,
            transition_id=transition.transition_id,
            binding_digest=transition.binding_digest,
            status=command.status.value,
            identity_epoch=command.identity_epoch,
        )
        if appended.kind is not AppendKind.COMMITTED or appended.digest is None:
            return self._deny(
                map_provider_fault(appended.fault or ProviderFault.OUTCOME_UNKNOWN),
                scope.tenant_id,
            )
        sealed = self._register.attach_receipt(
            self._bundle.primary,
            application_id=scope.application_id,
            environment_id=scope.environment_id,
            tenant_id=scope.tenant_id,
            transition_id=transition.transition_id,
            request_digest_value=transition.request_digest,
            witness_segment=appended.segment or 0,
            witness_offset=appended.offset or 0,
            seal_digest=appended.digest,
        )
        if sealed.kind is not PrimaryCommitKind.COMMITTED:
            return self._from_step(sealed, scope.tenant_id)
        committed = self._register.commit_transition(
            self._bundle.primary,
            application_id=scope.application_id,
            environment_id=scope.environment_id,
            tenant_id=scope.tenant_id,
            transition_id=transition.transition_id,
            request_digest_value=transition.request_digest,
        )
        if committed.kind is not PrimaryCommitKind.COMMITTED or committed.head is None:
            fresh = self._bundle.primary.read_head(
                scope.application_id,
                scope.environment_id,
                scope.tenant_id,
            )
            if fresh is not None and fresh.phase == _COMMITTED and fresh.revision == command.revision:
                return self._result(
                    ContinuityOutcome.COMMITTED,
                    tenant_id=scope.tenant_id,
                    head=self._public(fresh),
                )
            return self._from_step(committed, scope.tenant_id)
        fresh = self._require_record(scope.tenant_id)
        if fresh.authority_incarnation != incarnation:
            return self._deny(ContinuityDenialCode.WITNESS_MISMATCH, scope.tenant_id)
        agreed = self._divergence(scope.tenant_id)
        if agreed is not None:
            return agreed
        return self._result(
            ContinuityOutcome.COMMITTED,
            tenant_id=scope.tenant_id,
            head=self._public(fresh),
        )

    def _recover_pending(self, scope: ContinuityScope) -> ContinuityResult:
        head = self._head(scope.tenant_id)
        if head is None or not head.transition_id:
            return self._deny(ContinuityDenialCode.TRANSITION_UNRESOLVED, scope.tenant_id)
        transition = self._bundle.primary.read_transition(
            scope.application_id,
            scope.environment_id,
            scope.tenant_id,
            head.transition_id,
        )
        if transition is None:
            return self._deny(ContinuityDenialCode.TRANSITION_UNRESOLVED, scope.tenant_id)
        chain = self._witness.verify_bounded(full_chain=True)
        if chain.fault is ProviderFault.UNAVAILABLE and head.phase == _PREPARED:
            return self._finish_pending_seal(scope, transition)
        if chain.fault is not None:
            return self._deny(map_provider_fault(chain.fault), scope.tenant_id)
        if chain.high_revision > transition.revision:
            return self._deny(ContinuityDenialCode.CONTINUITY_REVISION_STALE, scope.tenant_id)
        if chain.high_revision < transition.revision:
            if head.phase != _PREPARED:
                return self._deny(ContinuityDenialCode.WITNESS_MISMATCH, scope.tenant_id)
            return self._finish_pending_seal(scope, transition)
        if not _tail_matches(chain, transition):
            return self._deny(ContinuityDenialCode.WITNESS_MISMATCH, scope.tenant_id)
        if head.phase == _PREPARED:
            sealed = self._register.attach_receipt(
                self._bundle.primary,
                application_id=scope.application_id,
                environment_id=scope.environment_id,
                tenant_id=scope.tenant_id,
                transition_id=transition.transition_id,
                request_digest_value=transition.request_digest,
                witness_segment=chain.segment or 0,
                witness_offset=chain.offset or 0,
                seal_digest=chain.digest or "",
            )
            if sealed.kind is not PrimaryCommitKind.COMMITTED:
                return self._from_step(sealed, scope.tenant_id)
        committed = self._register.commit_transition(
            self._bundle.primary,
            application_id=scope.application_id,
            environment_id=scope.environment_id,
            tenant_id=scope.tenant_id,
            transition_id=transition.transition_id,
            request_digest_value=transition.request_digest,
        )
        if committed.kind is not PrimaryCommitKind.COMMITTED:
            return self._from_step(committed, scope.tenant_id)
        restored = self._require_record(scope.tenant_id)
        if restored.revision != transition.revision:
            return self._deny(ContinuityDenialCode.WITNESS_MISMATCH, scope.tenant_id)
        diverged = self._divergence(scope.tenant_id)
        if diverged is not None:
            return diverged
        return self._result(
            ContinuityOutcome.RECOVERED,
            tenant_id=scope.tenant_id,
            head=self._public(restored),
        )

    def _finish_pending_seal(self, scope: ContinuityScope, transition: TransitionRecord) -> ContinuityResult:
        """Complete one already prepared seal. Does not allocate a new revision."""

        appended = self._witness.append_seal(
            application_id=scope.application_id,
            environment_id=scope.environment_id,
            tenant_id=scope.tenant_id,
            revision=transition.revision,
            fencing_token=transition.fencing_token,
            transition_id=transition.transition_id,
            binding_digest=transition.binding_digest,
            status=transition.status,
            identity_epoch=transition.identity_epoch,
        )
        if appended.kind is not AppendKind.COMMITTED or appended.digest is None:
            return self._deny(
                map_provider_fault(appended.fault or ProviderFault.OUTCOME_UNKNOWN),
                scope.tenant_id,
            )
        sealed = self._register.attach_receipt(
            self._bundle.primary,
            application_id=scope.application_id,
            environment_id=scope.environment_id,
            tenant_id=scope.tenant_id,
            transition_id=transition.transition_id,
            request_digest_value=transition.request_digest,
            witness_segment=appended.segment or 0,
            witness_offset=appended.offset or 0,
            seal_digest=appended.digest,
        )
        if sealed.kind is not PrimaryCommitKind.COMMITTED:
            return self._from_step(sealed, scope.tenant_id)
        committed = self._register.commit_transition(
            self._bundle.primary,
            application_id=scope.application_id,
            environment_id=scope.environment_id,
            tenant_id=scope.tenant_id,
            transition_id=transition.transition_id,
            request_digest_value=transition.request_digest,
        )
        if committed.kind is not PrimaryCommitKind.COMMITTED:
            return self._from_step(committed, scope.tenant_id)
        restored = self._require_record(scope.tenant_id)
        if restored.revision != transition.revision or restored.phase != _COMMITTED:
            return self._deny(ContinuityDenialCode.WITNESS_MISMATCH, scope.tenant_id)
        diverged = self._divergence(scope.tenant_id)
        if diverged is not None:
            return diverged
        return self._result(
            ContinuityOutcome.RECOVERED,
            tenant_id=scope.tenant_id,
            head=self._public(restored),
        )

    def _transition(self, command: _Command) -> TransitionRecord:
        binding_digest = continuity_binding_digest(command.binding)
        transition_id = f"{self._selection.environment_id}:{command.revision}:{command.fencing_token}"
        digest = request_digest(
            application_id=self._selection.application_id,
            environment_id=self._selection.environment_id,
            tenant_id=command.tenant_id,
            revision=command.revision,
            fencing_token=command.fencing_token,
            transition_id=transition_id,
            binding_digest=binding_digest,
            status=command.status.value,
            identity_epoch=command.identity_epoch,
            pis_history_id=command.binding.pis_history_id,
            pis_watermark=command.binding.pis_committed_watermark,
            gca_history_id=command.binding.gca_history_id,
            gca_watermark=command.binding.gca_committed_watermark,
        )
        return TransitionRecord(
            application_id=self._selection.application_id,
            environment_id=self._selection.environment_id,
            tenant_id=command.tenant_id,
            transition_id=transition_id,
            request_digest=digest,
            revision=command.revision,
            fencing_token=command.fencing_token,
            binding_digest=binding_digest,
            status=command.status.value,
            identity_epoch=command.identity_epoch,
            phase=_PREPARED,
            pis_history_id=command.binding.pis_history_id,
            pis_watermark=command.binding.pis_committed_watermark,
            gca_history_id=command.binding.gca_history_id,
            gca_watermark=command.binding.gca_committed_watermark,
        )

    def _incarnation(self, tenant_id: str) -> str:
        head = self._head(tenant_id)
        if head is not None and head.authority_incarnation:
            return head.authority_incarnation
        return (
            f"{self._selection.application_id}:{self._selection.environment_id}:"
            "offline-incarnation:1"
        )

    def _divergence(self, tenant_id: str) -> ContinuityResult | None:
        head = self._head(tenant_id)
        tail = self._witness.verify_bounded(full_chain=False)
        if tail.fault is not None:
            return self._deny(map_provider_fault(tail.fault), tenant_id)
        if head is None:
            if tail.high_revision:
                return self._deny(ContinuityDenialCode.CONTINUITY_REVISION_STALE, tenant_id)
            return None
        if tail.high_revision > head.revision:
            return self._deny(ContinuityDenialCode.CONTINUITY_REVISION_STALE, tenant_id)
        if tail.high_revision < head.revision:
            return self._deny(ContinuityDenialCode.WITNESS_MISMATCH, tenant_id)
        if tail.binding_digest != continuity_binding_digest(self._binding_of(head)):
            return self._deny(ContinuityDenialCode.WITNESS_MISMATCH, tenant_id)
        if (
            tail.fencing_token != head.fencing_token
            or tail.transition_id != head.transition_id
            or tail.status != head.status
            or tail.identity_epoch != head.identity_epoch
            or tail.application_id != head.application_id
            or tail.environment_id != head.environment_id
            or tail.tenant_id != head.tenant_id
        ):
            return self._deny(ContinuityDenialCode.WITNESS_MISMATCH, tenant_id)
        return None

    def _require_mutable(self, tenant_id: str) -> ContinuityResult | None:
        blocked = self._require_present(tenant_id)
        if blocked is not None:
            return blocked
        head = self._require_record(tenant_id)
        if head.status == ContinuityHeadStatus.REVOKED.value:
            return self._deny(ContinuityDenialCode.REVOKED, tenant_id)
        if head.status == ContinuityHeadStatus.QUARANTINED.value:
            return self._deny(ContinuityDenialCode.QUARANTINED, tenant_id)
        return self._divergence(tenant_id)

    def _require_present(self, tenant_id: str) -> ContinuityResult | None:
        phase = self._phase(tenant_id)
        if phase is not None and phase != _COMMITTED:
            return self._deny(ContinuityDenialCode.TRANSITION_UNRESOLVED, tenant_id)
        if self._head(tenant_id) is None:
            return self._deny(ContinuityDenialCode.NOT_INITIALIZED, tenant_id)
        return None

    def _compare(
        self,
        tenant_id: str,
        expected_revision: int,
        expected_fencing_token: str,
    ) -> ContinuityResult | None:
        head = self._require_record(tenant_id)
        if expected_revision != head.revision:
            return self._deny(ContinuityDenialCode.REVISION_STALE, tenant_id)
        if expected_fencing_token != head.fencing_token:
            return self._deny(ContinuityDenialCode.FENCE_STALE, tenant_id)
        return None

    def _precheck(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        allowed: ContinuityPermission | frozenset[ContinuityPermission],
    ) -> ContinuityResult | None:
        scope_denial = self._check_scope(scope)
        if scope_denial is not None:
            return self._deny(scope_denial, scope.tenant_id)
        if grant.principal_id != self._issuer:
            return self._deny(ContinuityDenialCode.PERMISSION_DENIED, scope.tenant_id)
        permitted = allowed if isinstance(allowed, frozenset) else frozenset((allowed,))
        if not isinstance(grant.permission, ContinuityPermission) or grant.permission not in permitted:
            return self._deny(ContinuityDenialCode.PERMISSION_DENIED, scope.tenant_id)
        try:
            action = continuity_authorization_action(grant.permission)
        except (TypeError, ValueError):
            return self._deny(ContinuityDenialCode.PERMISSION_DENIED, scope.tenant_id)
        return self._authorize(action, scope.tenant_id)

    def _authorize(self, action: str, tenant_id: str) -> ContinuityResult | None:
        if action not in _ACTIONS.values():
            return self._deny(ContinuityDenialCode.PERMISSION_DENIED, tenant_id)
        head = self._head(tenant_id)
        revision = 0 if head is None or head.revision < 1 else head.revision
        fence = "none" if head is None or not head.fencing_token else head.fencing_token
        try:
            request = ControlPlaneMutationRequest(
                mutation_id=(
                    f"history_continuity:{self._selection.environment_id}:"
                    f"{action}:{revision}:{fence}"
                ),
                mutation_type=action,
                principal=self._principal,
                resource_scope=(
                    f"history_continuity.application:{self._selection.application_id}"
                    f".environment:{self._selection.environment_id}"
                ),
                resource_type="history_continuity.environment",
                resource_id=(
                    f"{self._selection.application_id}:{self._selection.environment_id}"
                ),
                current_revision=f"continuity:{revision}",
                target_revision=f"continuity:{revision + 1}",
                risk_classification=ControlPlaneMutationRisk.HIGH,
            )
            result = self._authorization_boundary.authorize(request)
        except Exception:
            return self._deny(ContinuityDenialCode.PERMISSION_DENIED, tenant_id)
        if (
            not result.permitted
            or result.validation_failed
            or result.requires_governed_continuation
            or result.decision.action is not PolicyAction.ALLOW
        ):
            return self._deny(ContinuityDenialCode.PERMISSION_DENIED, tenant_id)
        evidence = result.evidence
        evidence_principal = (evidence.principal_auth_subject or "").strip() or (
            evidence.principal_user_id or ""
        ).strip()
        if (
            evidence.policy_action is not PolicyAction.ALLOW
            or evidence.mutation_type != action
            or evidence.tenant_id != self._principal.tenant_id
            or evidence_principal != self._issuer
        ):
            return self._deny(ContinuityDenialCode.PERMISSION_DENIED, tenant_id)
        return None

    def _check_scope(self, scope: ContinuityScope) -> ContinuityDenialCode | None:
        if scope.application_id != self._selection.application_id:
            return ContinuityDenialCode.SCOPE_MISMATCH
        if scope.environment_id != self._selection.environment_id:
            return ContinuityDenialCode.ENVIRONMENT_MISMATCH
        try:
            resolved = self._tenant_resolver.resolve_tenant_id(
                application_id=scope.application_id,
                application_environment_id=scope.environment_id,
            )
        except Exception:
            return ContinuityDenialCode.TENANT_MISMATCH
        if scope.tenant_id != resolved or scope.tenant_id != self._principal.tenant_id:
            return ContinuityDenialCode.TENANT_MISMATCH
        head = self._head(scope.tenant_id)
        if head is not None and head.tenant_id != resolved:
            return ContinuityDenialCode.TENANT_MISMATCH
        return None

    def _head(self, tenant_id: str) -> HeadRecord | None:
        return self._bundle.primary.read_head(
            self._selection.application_id,
            self._selection.environment_id,
            tenant_id,
        )

    def _phase(self, tenant_id: str) -> str | None:
        head = self._head(tenant_id)
        if head is None:
            return None
        return head.phase

    def _require_record(self, tenant_id: str) -> HeadRecord:
        head = self._head(tenant_id)
        if head is None:
            raise RuntimeError("continuity head disappeared after a presence check")
        return head

    def _from_step(self, step: PrimaryStep, tenant_id: str) -> ContinuityResult:
        if step.fault is None:
            return self._deny(ContinuityDenialCode.TRANSITION_UNRESOLVED, tenant_id)
        return self._deny(map_provider_fault(step.fault), tenant_id)

    def _deny(self, code: ContinuityDenialCode, tenant_id: str) -> ContinuityResult:
        head = self._head(tenant_id)
        public = None if head is None or head.revision < 1 else self._public(head)
        return ContinuityResult(
            outcome=ContinuityOutcome.DENIED,
            denial=code,
            head=public,
            witness_high_watermark=self._witness_high(),
        )

    def _result(
        self,
        outcome: ContinuityOutcome,
        *,
        tenant_id: str,
        head: ContinuityHead | None,
    ) -> ContinuityResult:
        del tenant_id
        return ContinuityResult(
            outcome=outcome,
            head=head,
            witness_high_watermark=self._witness_high(),
        )

    def _witness_high(self) -> int:
        tail = self._witness.verify_bounded(full_chain=False)
        return tail.high_revision

    def _public(self, head: HeadRecord) -> ContinuityHead:
        return ContinuityHead(
            application_id=head.application_id,
            environment_id=head.environment_id,
            tenant_id=head.tenant_id,
            revision=head.revision,
            fencing_token=head.fencing_token,
            status=ContinuityHeadStatus(head.status),
            binding=self._binding_of(head),
            identity_epoch=head.identity_epoch,
            proof_revision=head.proof_revision,
        )

    def _binding_of(self, head: HeadRecord) -> HistoryBinding:
        return HistoryBinding(
            pis_history_id=head.pis_history_id,
            pis_committed_watermark=head.pis_watermark,
            gca_history_id=head.gca_history_id,
            gca_committed_watermark=head.gca_watermark,
        )


def open_offline_external_history_continuity(
    *,
    tenant_resolver: ApplicationEnvironmentTenantResolver,
    selection: HistoryContinuityProviderSelection,
    authorization_boundary: ControlPlaneMutationAuthorizationBoundary,
    principal: RequestIdentity,
    issuer_principal_id: str,
    bundle: OfflineProviderBundle,
) -> OfflineSpannerAzureHistoryContinuity:
    return OfflineSpannerAzureHistoryContinuity(
        tenant_resolver=tenant_resolver,
        selection=selection,
        authorization_boundary=authorization_boundary,
        principal=principal,
        issuer_principal_id=issuer_principal_id,
        bundle=bundle,
        composition_token=_COMPOSITION_TOKEN,
    )


def _next_fence(current: str) -> str:
    return str(int(current) + 1)


def _binding_moves_forward(
    current: HistoryBinding,
    proposed: HistoryBinding,
    *,
    allow_rebind: bool,
) -> bool:
    if proposed.pis_committed_watermark < current.pis_committed_watermark:
        return False
    if proposed.gca_committed_watermark < current.gca_committed_watermark:
        return False
    identity_same = (
        proposed.pis_history_id == current.pis_history_id
        and proposed.gca_history_id == current.gca_history_id
    )
    if not identity_same and not allow_rebind:
        return False
    return (
        proposed.pis_committed_watermark > current.pis_committed_watermark
        or proposed.gca_committed_watermark > current.gca_committed_watermark
        or not identity_same
    )


def _tail_matches(tail: WitnessView, transition: TransitionRecord) -> bool:
    return (
        tail.high_revision == transition.revision
        and tail.transition_id == transition.transition_id
        and tail.binding_digest == transition.binding_digest
        and tail.fencing_token == transition.fencing_token
        and tail.status == transition.status
        and tail.identity_epoch == transition.identity_epoch
        and tail.digest is not None
    )
