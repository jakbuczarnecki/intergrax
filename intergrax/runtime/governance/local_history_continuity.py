# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic local ELCR-IIRW primary and independent witness (D03-A-LOCAL-01).

Qualification adapter only. Construct it through
``compose_history_continuity_authority``. Capability authorization is the
composed control-plane mutation boundary. Crash and rollback methods are test
seams on this same state owner; they are not a second continuity mechanism.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from enum import StrEnum

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
    HistoryContinuityActivation,
    HistoryContinuityRetentionPolicy,
    WitnessSeal,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

_COMPOSITION_TOKEN = object()
_READ_PERMISSIONS = frozenset(ContinuityPermission)
_WITNESS_ALTER_FIELDS = frozenset(
    {
        "environment_id",
        "revision",
        "fencing_token",
        "transition_id",
        "binding_digest",
        "status",
        "identity_epoch",
    }
)
_CONTINUITY_AUTHORIZATION_ACTIONS: dict[ContinuityPermission, str] = {
    ContinuityPermission.WRITE: "history_continuity.write",
    ContinuityPermission.INITIALIZE: "history_continuity.initialize",
    ContinuityPermission.RECONCILE: "history_continuity.reconcile",
    ContinuityPermission.INVALIDATE: "history_continuity.invalidate",
    ContinuityPermission.RECOVER: "history_continuity.recover",
}


def continuity_authorization_action(permission: ContinuityPermission) -> str:
    """Map one continuity capability to the canonical mutation action."""

    if not isinstance(permission, ContinuityPermission):
        raise ValueError("unsupported continuity permission")
    return _CONTINUITY_AUTHORIZATION_ACTIONS[permission]


class LocalContinuityCrashPoint(StrEnum):
    """One-shot interruption of the next mutating transition."""

    AFTER_PREPARE = "after_prepare"
    AFTER_WITNESS_SEAL = "after_witness_seal"


@dataclass(frozen=True, slots=True)
class _PreparedTransition:
    transition_id: str
    revision: int
    fencing_token: str
    binding: HistoryBinding
    status: ContinuityHeadStatus
    identity_epoch: int
    tenant_id: str


def continuity_binding_digest(binding: HistoryBinding) -> str:
    payload = (
        f"{binding.pis_history_id}|{binding.pis_committed_watermark}|"
        f"{binding.gca_history_id}|{binding.gca_committed_watermark}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _result(
    outcome: ContinuityOutcome,
    *,
    denial: ContinuityDenialCode | None = None,
    head: ContinuityHead | None = None,
    witness_high: int | None = None,
) -> ContinuityResult:
    return ContinuityResult(
        outcome=outcome,
        denial=denial,
        head=head,
        witness_high_watermark=witness_high,
    )


def _deny(
    code: ContinuityDenialCode,
    *,
    head: ContinuityHead | None = None,
    witness_high: int | None = None,
) -> ContinuityResult:
    return _result(
        ContinuityOutcome.DENIED,
        denial=code,
        head=head,
        witness_high=witness_high,
    )


def _next_fence(current: str | None) -> str:
    if current is None:
        return "1"
    return str(int(current) + 1)


def _transition_id(environment_id: str, revision: int, fencing_token: str) -> str:
    return f"{environment_id}:{revision}:{fencing_token}"


def _binding_moves_forward(
    current: HistoryBinding,
    proposed: HistoryBinding,
    *,
    allow_identity_rebind: bool,
) -> bool:
    if proposed.pis_committed_watermark < current.pis_committed_watermark:
        return False
    if proposed.gca_committed_watermark < current.gca_committed_watermark:
        return False
    identity_same = (
        proposed.pis_history_id == current.pis_history_id
        and proposed.gca_history_id == current.gca_history_id
    )
    if not identity_same and not allow_identity_rebind:
        return False
    return (
        proposed.pis_committed_watermark > current.pis_committed_watermark
        or proposed.gca_committed_watermark > current.gca_committed_watermark
        or not identity_same
    )


class InactiveHistoryContinuityAuthority:
    """Fail-closed authority used when activation is missing or rejected."""

    def sealed_retention(self) -> HistoryContinuityRetentionPolicy | None:
        return None

    def initialize(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        binding: HistoryBinding,
    ) -> ContinuityResult:
        del grant, scope, binding
        return _deny(ContinuityDenialCode.ACTIVATION_ABSENT)

    def advance(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        expected_revision: int,
        expected_fencing_token: str,
        binding: HistoryBinding,
    ) -> ContinuityResult:
        del grant, scope, expected_revision, expected_fencing_token, binding
        return _deny(ContinuityDenialCode.ACTIVATION_ABSENT)

    def reconcile(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        expected_revision: int,
        expected_fencing_token: str,
        binding: HistoryBinding,
    ) -> ContinuityResult:
        del grant, scope, expected_revision, expected_fencing_token, binding
        return _deny(ContinuityDenialCode.ACTIVATION_ABSENT)

    def invalidate(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        expected_revision: int,
        expected_fencing_token: str,
        status: ContinuityHeadStatus,
    ) -> ContinuityResult:
        del grant, scope, expected_revision, expected_fencing_token, status
        return _deny(ContinuityDenialCode.ACTIVATION_ABSENT)

    def recover(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
    ) -> ContinuityResult:
        del grant, scope
        return _deny(ContinuityDenialCode.ACTIVATION_ABSENT)

    def read_head(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
    ) -> ContinuityResult:
        del grant, scope
        return _deny(ContinuityDenialCode.ACTIVATION_ABSENT)

    def attempt_retention_change(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        retention_seconds: int,
        ceiling_seconds: int,
        legal_hold: bool,
    ) -> ContinuityResult:
        del grant, scope, retention_seconds, ceiling_seconds, legal_hold
        return _deny(ContinuityDenialCode.ACTIVATION_ABSENT)


class LocalDeterministicHistoryContinuity:
    """In-process linearizable head plus an append-only witness."""

    def __init__(
        self,
        *,
        tenant_resolver: ApplicationEnvironmentTenantResolver,
        activation: HistoryContinuityActivation,
        authorization_boundary: ControlPlaneMutationAuthorizationBoundary,
        principal: RequestIdentity,
        issuer_principal_id: str,
        composition_token: object,
    ) -> None:
        if composition_token is not _COMPOSITION_TOKEN:
            raise RuntimeError(
                "history continuity requires compose_history_continuity_authority",
            )
        if canonical_principal_id_from_request_identity(principal) != issuer_principal_id:
            raise RuntimeError("continuity issuer does not match governance principal")
        self._tenant_resolver = tenant_resolver
        self._activation = activation
        self._authorization_boundary = authorization_boundary
        self._principal = principal
        self._issuer = issuer_principal_id
        self._retention = activation.retention
        self._lock = threading.Lock()
        self._heads: dict[int, ContinuityHead] = {}
        self._head: ContinuityHead | None = None
        self._seals: dict[int, WitnessSeal] = {}
        self._pending: _PreparedTransition | None = None
        self._crash: LocalContinuityCrashPoint | None = None

    def sealed_retention(self) -> HistoryContinuityRetentionPolicy | None:
        return self._retention

    def arm_local_crash(self, point: LocalContinuityCrashPoint) -> None:
        """Qualification seam: interrupt the next transition once."""

        with self._lock:
            self._crash = point

    def testing_rollback_primary(self) -> None:
        """Qualification seam: move the primary head backward without the witness."""

        with self._lock:
            if self._pending is not None or self._head is None or self._head.revision < 2:
                raise RuntimeError("primary rollback requires a committed revision above 1")
            self._head = self._heads[self._head.revision - 1]

    def testing_rollback_witness(self) -> None:
        """Qualification seam: drop the highest witness seal without the primary."""

        with self._lock:
            high = self._witness_high()
            if high < 1 or self._pending is not None:
                raise RuntimeError("witness rollback requires a sealed revision and no pending transition")
            del self._seals[high]

    def testing_alter_witness(self, sealed_revision: int, **changes: object) -> None:
        """Qualification seam: change sealed witness metadata without a transition."""

        if not changes or not set(changes).issubset(_WITNESS_ALTER_FIELDS):
            raise RuntimeError("witness alteration must name sealed identity fields")
        with self._lock:
            seal = self._seals.get(sealed_revision)
            if seal is None:
                raise RuntimeError("witness revision is not sealed")
            self._seals[sealed_revision] = seal.model_copy(update=dict(changes))

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
            if self._head is not None or self._seals or self._pending is not None:
                return self._denied(ContinuityDenialCode.ALREADY_INITIALIZED)
            prepared = _PreparedTransition(
                transition_id=_transition_id(scope.environment_id, 1, "1"),
                revision=1,
                fencing_token="1",
                binding=binding,
                status=ContinuityHeadStatus.ACTIVE,
                identity_epoch=1,
                tenant_id=scope.tenant_id,
            )
            return self._seal_and_commit(prepared)

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
            state = self._require_mutable_head()
            if state is not None:
                return state
            compared = self._compare(expected_revision, expected_fencing_token)
            if compared is not None:
                return compared
            assert self._head is not None
            if not _binding_moves_forward(
                self._head.binding,
                binding,
                allow_identity_rebind=False,
            ):
                return self._denied(ContinuityDenialCode.CONTINUITY_REVISION_STALE)
            fence = self._head.fencing_token
            revision = self._head.revision + 1
            prepared = _PreparedTransition(
                transition_id=_transition_id(scope.environment_id, revision, fence),
                revision=revision,
                fencing_token=fence,
                binding=binding,
                status=ContinuityHeadStatus.ACTIVE,
                identity_epoch=self._head.identity_epoch,
                tenant_id=self._head.tenant_id,
            )
            return self._seal_and_commit(prepared)

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
            state = self._require_head_for_privileged_mutation()
            if state is not None:
                return state
            assert self._head is not None
            if self._head.status is ContinuityHeadStatus.REVOKED:
                return self._denied(ContinuityDenialCode.REVOKED)
            if self._head.status is not ContinuityHeadStatus.QUARANTINED:
                return self._denied(ContinuityDenialCode.RECONCILIATION_UNAVAILABLE)
            compared = self._compare(expected_revision, expected_fencing_token)
            if compared is not None:
                return compared
            if not _binding_moves_forward(
                self._head.binding,
                binding,
                allow_identity_rebind=True,
            ):
                return self._denied(ContinuityDenialCode.CONTINUITY_REVISION_STALE)
            fence = _next_fence(self._head.fencing_token)
            revision = self._head.revision + 1
            prepared = _PreparedTransition(
                transition_id=_transition_id(scope.environment_id, revision, fence),
                revision=revision,
                fencing_token=fence,
                binding=binding,
                status=ContinuityHeadStatus.ACTIVE,
                identity_epoch=self._head.identity_epoch + 1,
                tenant_id=self._head.tenant_id,
            )
            return self._seal_and_commit(prepared)

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
            if status not in (
                ContinuityHeadStatus.QUARANTINED,
                ContinuityHeadStatus.REVOKED,
            ):
                return self._denied(ContinuityDenialCode.PERMISSION_DENIED)
            state = self._require_head_for_privileged_mutation()
            if state is not None:
                return state
            assert self._head is not None
            if self._head.status is ContinuityHeadStatus.REVOKED:
                return self._denied(ContinuityDenialCode.REVOKED)
            compared = self._compare(expected_revision, expected_fencing_token)
            if compared is not None:
                return compared
            assert self._head is not None
            fence = _next_fence(self._head.fencing_token)
            revision = self._head.revision + 1
            prepared = _PreparedTransition(
                transition_id=_transition_id(scope.environment_id, revision, fence),
                revision=revision,
                fencing_token=fence,
                binding=self._head.binding,
                status=status,
                identity_epoch=self._head.identity_epoch,
                tenant_id=self._head.tenant_id,
            )
            return self._seal_and_commit(prepared)

    def recover(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
    ) -> ContinuityResult:
        with self._lock:
            denied = self._precheck(grant, scope, ContinuityPermission.RECOVER)
            if denied is not None:
                return denied
            if self._pending is not None:
                return self._recover_pending()
            return self._recover_consistent_head()

    def read_head(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
    ) -> ContinuityResult:
        with self._lock:
            denied = self._precheck(grant, scope, _READ_PERMISSIONS)
            if denied is not None:
                return denied
            if self._pending is not None:
                return self._denied(ContinuityDenialCode.TRANSITION_UNRESOLVED)
            if self._head is None:
                return self._denied(ContinuityDenialCode.NOT_INITIALIZED)
            diverged = self._divergence_denial()
            if diverged is not None:
                return diverged
            return self._committed(self._head)

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
                return self._denied(scope_denial)
            if grant.principal_id != self._issuer:
                return self._denied(ContinuityDenialCode.PERMISSION_DENIED)
            weaker = (
                retention_seconds < self._retention.retention_seconds
                or ceiling_seconds < self._retention.ceiling_seconds
                or legal_hold is not True
            )
            if weaker:
                return self._denied(ContinuityDenialCode.RETENTION_WEAKENING_FORBIDDEN)
            return self._denied(ContinuityDenialCode.PERMISSION_DENIED)

    def _recover_pending(self) -> ContinuityResult:
        prepared = self._pending
        assert prepared is not None
        seal = self._seals.get(prepared.revision)
        if seal is None:
            self._pending = None
            return self._recovered(self._head)
        if _seal_matches_prepared(
            seal,
            prepared,
            environment_id=self._activation.environment_id,
        ):
            head = self._commit_prepared(prepared)
            return self._recovered(head)
        return self._denied(ContinuityDenialCode.WITNESS_MISMATCH)

    def _recover_consistent_head(self) -> ContinuityResult:
        if self._head is None:
            if self._witness_high() > 0:
                return self._denied(ContinuityDenialCode.WITNESS_MISMATCH)
            return self._recovered(None)
        diverged = self._divergence_denial()
        if diverged is not None:
            return diverged
        return self._recovered(self._head)

    def _seal_and_commit(self, prepared: _PreparedTransition) -> ContinuityResult:
        self._pending = prepared
        if self._consume_crash(LocalContinuityCrashPoint.AFTER_PREPARE):
            return self._denied(ContinuityDenialCode.TRANSITION_UNRESOLVED)
        if not self._append_seal(prepared):
            return self._denied(ContinuityDenialCode.WITNESS_MISMATCH)
        if self._consume_crash(LocalContinuityCrashPoint.AFTER_WITNESS_SEAL):
            return self._denied(ContinuityDenialCode.TRANSITION_UNRESOLVED)
        head = self._commit_prepared(prepared)
        return self._committed(head)

    def _append_seal(self, prepared: _PreparedTransition) -> bool:
        if prepared.revision != self._witness_high() + 1:
            return False
        self._seals[prepared.revision] = WitnessSeal(
            environment_id=self._activation.environment_id,
            revision=prepared.revision,
            fencing_token=prepared.fencing_token,
            transition_id=prepared.transition_id,
            binding_digest=continuity_binding_digest(prepared.binding),
            status=prepared.status,
            identity_epoch=prepared.identity_epoch,
        )
        return True

    def _commit_prepared(self, prepared: _PreparedTransition) -> ContinuityHead:
        head = ContinuityHead(
            application_id=self._activation.application_id,
            environment_id=self._activation.environment_id,
            tenant_id=prepared.tenant_id,
            revision=prepared.revision,
            fencing_token=prepared.fencing_token,
            status=prepared.status,
            binding=prepared.binding,
            identity_epoch=prepared.identity_epoch,
            proof_revision=prepared.revision,
        )
        self._heads[head.revision] = head
        self._head = head
        self._pending = None
        return head

    def _precheck(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        allowed: ContinuityPermission | frozenset[ContinuityPermission],
    ) -> ContinuityResult | None:
        scope_denial = self._check_scope(scope)
        if scope_denial is not None:
            return self._denied(scope_denial)
        if grant.principal_id != self._issuer:
            return self._denied(ContinuityDenialCode.PERMISSION_DENIED)
        permitted = allowed if isinstance(allowed, frozenset) else frozenset((allowed,))
        if (
            not isinstance(grant.permission, ContinuityPermission)
            or grant.permission not in permitted
        ):
            return self._denied(ContinuityDenialCode.PERMISSION_DENIED)
        try:
            action = continuity_authorization_action(grant.permission)
        except (TypeError, ValueError):
            return self._denied(ContinuityDenialCode.PERMISSION_DENIED)
        return self._authorize_action(action)

    def _authorize_action(self, action: str) -> ContinuityResult | None:
        if action not in _CONTINUITY_AUTHORIZATION_ACTIONS.values():
            return self._denied(ContinuityDenialCode.PERMISSION_DENIED)
        revision = 0 if self._head is None else self._head.revision
        fence = "none" if self._head is None else self._head.fencing_token
        try:
            request = ControlPlaneMutationRequest(
                mutation_id=(
                    f"history_continuity:{self._activation.environment_id}:"
                    f"{action}:{revision}:{fence}"
                ),
                mutation_type=action,
                principal=self._principal,
                resource_scope=(
                    f"history_continuity.application:{self._activation.application_id}"
                    f".environment:{self._activation.environment_id}"
                ),
                resource_type="history_continuity.environment",
                resource_id=(
                    f"{self._activation.application_id}:{self._activation.environment_id}"
                ),
                current_revision=f"continuity:{revision}",
                target_revision=f"continuity:{revision + 1}",
                risk_classification=ControlPlaneMutationRisk.HIGH,
            )
            result = self._authorization_boundary.authorize(request)
        except Exception:
            return self._denied(ContinuityDenialCode.PERMISSION_DENIED)
        if (
            not result.permitted
            or result.validation_failed
            or result.requires_governed_continuation
            or result.decision.action is not PolicyAction.ALLOW
        ):
            return self._denied(ContinuityDenialCode.PERMISSION_DENIED)
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
            return self._denied(ContinuityDenialCode.PERMISSION_DENIED)
        return None

    def _check_scope(self, scope: ContinuityScope) -> ContinuityDenialCode | None:
        if scope.application_id != self._activation.application_id:
            return ContinuityDenialCode.SCOPE_MISMATCH
        if scope.environment_id != self._activation.environment_id:
            return ContinuityDenialCode.ENVIRONMENT_MISMATCH
        try:
            resolved = self._tenant_resolver.resolve_tenant_id(
                application_id=scope.application_id,
                application_environment_id=scope.environment_id,
            )
        except Exception:
            return ContinuityDenialCode.TENANT_MISMATCH
        if scope.tenant_id != resolved:
            return ContinuityDenialCode.TENANT_MISMATCH
        if self._head is not None and self._head.tenant_id != resolved:
            return ContinuityDenialCode.TENANT_MISMATCH
        return None

    def _require_mutable_head(self) -> ContinuityResult | None:
        blocked = self._require_head_for_privileged_mutation()
        if blocked is not None:
            return blocked
        assert self._head is not None
        if self._head.status is ContinuityHeadStatus.REVOKED:
            return self._denied(ContinuityDenialCode.REVOKED)
        if self._head.status is ContinuityHeadStatus.QUARANTINED:
            return self._denied(ContinuityDenialCode.QUARANTINED)
        return self._divergence_denial()

    def _require_head_for_privileged_mutation(self) -> ContinuityResult | None:
        if self._pending is not None:
            return self._denied(ContinuityDenialCode.TRANSITION_UNRESOLVED)
        if self._head is None:
            return self._denied(ContinuityDenialCode.NOT_INITIALIZED)
        return None

    def _divergence_denial(self) -> ContinuityResult | None:
        assert self._head is not None
        witness_high = self._witness_high()
        if witness_high > self._head.revision:
            return self._denied(ContinuityDenialCode.CONTINUITY_REVISION_STALE)
        if witness_high < self._head.revision:
            return self._denied(ContinuityDenialCode.WITNESS_MISMATCH)
        if not self._committed_witnesses_match():
            return self._denied(ContinuityDenialCode.WITNESS_MISMATCH)
        return None

    def _committed_witnesses_match(self) -> bool:
        assert self._head is not None
        expected = set(range(1, self._head.revision + 1))
        if set(self._seals) != expected or set(self._heads) != expected:
            return False
        for revision in expected:
            if not _seal_matches_committed_head(
                self._seals[revision],
                self._heads[revision],
                application_id=self._activation.application_id,
                environment_id=self._activation.environment_id,
            ):
                return False
        return True

    def _compare(
        self,
        expected_revision: int,
        expected_fencing_token: str,
    ) -> ContinuityResult | None:
        assert self._head is not None
        if expected_revision != self._head.revision:
            return self._denied(ContinuityDenialCode.REVISION_STALE)
        if expected_fencing_token != self._head.fencing_token:
            return self._denied(ContinuityDenialCode.FENCE_STALE)
        return None

    def _consume_crash(self, point: LocalContinuityCrashPoint) -> bool:
        if self._crash is not point:
            return False
        self._crash = None
        return True

    def _witness_high(self) -> int:
        if not self._seals:
            return 0
        return max(self._seals)

    def _denied(self, code: ContinuityDenialCode) -> ContinuityResult:
        return _deny(code, head=self._head, witness_high=self._witness_high())

    def _committed(self, head: ContinuityHead) -> ContinuityResult:
        return _result(
            ContinuityOutcome.COMMITTED,
            head=head,
            witness_high=self._witness_high(),
        )

    def _recovered(self, head: ContinuityHead | None) -> ContinuityResult:
        return _result(
            ContinuityOutcome.RECOVERED,
            head=head,
            witness_high=self._witness_high(),
        )


def _seal_matches_committed_head(
    seal: WitnessSeal,
    head: ContinuityHead,
    *,
    application_id: str,
    environment_id: str,
) -> bool:
    if head.application_id != application_id or head.environment_id != environment_id:
        return False
    return _seal_identity_matches(
        seal,
        environment_id=head.environment_id,
        revision=head.revision,
        fencing_token=head.fencing_token,
        transition_id=_transition_id(
            head.environment_id,
            head.revision,
            head.fencing_token,
        ),
        binding=head.binding,
        status=head.status,
        identity_epoch=head.identity_epoch,
    )


def _seal_matches_prepared(
    seal: WitnessSeal,
    prepared: _PreparedTransition,
    *,
    environment_id: str,
) -> bool:
    canonical_transition = _transition_id(
        environment_id,
        prepared.revision,
        prepared.fencing_token,
    )
    if prepared.transition_id != canonical_transition:
        return False
    return _seal_identity_matches(
        seal,
        environment_id=environment_id,
        revision=prepared.revision,
        fencing_token=prepared.fencing_token,
        transition_id=canonical_transition,
        binding=prepared.binding,
        status=prepared.status,
        identity_epoch=prepared.identity_epoch,
    )


def _seal_identity_matches(
    seal: WitnessSeal,
    *,
    environment_id: str,
    revision: int,
    fencing_token: str,
    transition_id: str,
    binding: HistoryBinding,
    status: ContinuityHeadStatus,
    identity_epoch: int,
) -> bool:
    return (
        seal.environment_id == environment_id
        and seal.revision == revision
        and seal.fencing_token == fencing_token
        and seal.transition_id == transition_id
        and seal.binding_digest == continuity_binding_digest(binding)
        and seal.status is status
        and seal.identity_epoch == identity_epoch
    )


def _open_local_history_continuity(
    *,
    tenant_resolver: ApplicationEnvironmentTenantResolver,
    activation: HistoryContinuityActivation,
    authorization_boundary: ControlPlaneMutationAuthorizationBoundary,
    principal: RequestIdentity,
    issuer_principal_id: str,
) -> LocalDeterministicHistoryContinuity:
    return LocalDeterministicHistoryContinuity(
        tenant_resolver=tenant_resolver,
        activation=activation,
        authorization_boundary=authorization_boundary,
        principal=principal,
        issuer_principal_id=issuer_principal_id,
        composition_token=_COMPOSITION_TOKEN,
    )
