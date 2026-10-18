# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-neutral ELCR-IIRW history-continuity contracts (D03-A-LOCAL-01).

The record is a governance continuity head, not a provider-invocation ledger.
No cloud backend, production retention period, or CREATE admission is selected
here. Activation policy is declarative configuration, not IAM evidence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_NON_BLANK = Field(min_length=1)


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must be non-blank")
    return value


class ContinuityPermission(StrEnum):
    """Separated continuity duties. None of these grants retention weakening."""

    WRITE = "write"
    INITIALIZE = "initialize"
    RECONCILE = "reconcile"
    INVALIDATE = "invalidate"
    RECOVER = "recover"


class ContinuityHeadStatus(StrEnum):
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    REVOKED = "revoked"


class ContinuityOutcome(StrEnum):
    COMMITTED = "committed"
    DENIED = "denied"
    RECOVERED = "recovered"


class ContinuityDenialCode(StrEnum):
    ACTIVATION_ABSENT = "activation_absent"
    PERMISSION_DENIED = "permission_denied"
    RETENTION_WEAKENING_FORBIDDEN = "retention_weakening_forbidden"
    REVISION_STALE = "revision_stale"
    FENCE_STALE = "fence_stale"
    SCOPE_MISMATCH = "scope_mismatch"
    TENANT_MISMATCH = "tenant_mismatch"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    TRANSITION_UNRESOLVED = "transition_unresolved"
    ALREADY_INITIALIZED = "already_initialized"
    NOT_INITIALIZED = "not_initialized"
    REVOKED = "revoked"
    QUARANTINED = "quarantined"
    WITNESS_MISMATCH = "witness_mismatch"
    CONTINUITY_REVISION_STALE = "continuity_revision_stale"
    RECONCILIATION_UNAVAILABLE = "reconciliation_unavailable"


class HistoryContinuityAdapterKind(StrEnum):
    """Local qualification adapter only. Cloud adapters are outside this gate."""

    LOCAL_DETERMINISTIC = "local_deterministic"


class HistoryContinuityHistoryScope(StrEnum):
    """Where one continuity history is anchored.

    Tenant scope exists so a tenant-specific anchor can be rejected. It is
    not permission to activate one.
    """

    ENVIRONMENT = "environment"
    TENANT = "tenant"


class HistoryContinuityRuntimeCredentialMode(StrEnum):
    """Runtime credential selection. This contract stores no secret material.

    Only ``LOCAL_NO_RUNTIME_CREDENTIAL`` may activate this gate. The other
    modes are explicit rejections. Naming workload federation does not
    implement federation and does not acquire a credential.
    """

    LOCAL_NO_RUNTIME_CREDENTIAL = "local_no_runtime_credential"
    STATIC_LONG_LIVED = "static_long_lived"
    ADMINISTRATOR = "administrator"
    WORKLOAD_FEDERATION = "workload_federation"


def administrator_identities_are_independent(primary: object, witness: object) -> bool:
    """Declarative distinctness. Not an IAM export or a directory lookup."""

    if type(primary) is not str or type(witness) is not str:
        return False
    if not primary or not witness:
        return False
    if primary != primary.strip() or witness != witness.strip():
        return False
    return primary.casefold() != witness.casefold()


def separated_capabilities_are_exact(capabilities: object) -> bool:
    """True only for the five existing duties, each present once."""

    if type(capabilities) is not tuple:
        return False
    expected = frozenset(ContinuityPermission)
    if len(capabilities) != len(expected):
        return False
    if any(type(item) is not ContinuityPermission for item in capabilities):
        return False
    return frozenset(capabilities) == expected


class HistoryContinuityRetentionPolicy(BaseModel):
    """Configurable retention held inside a hard ceiling. Not a production period."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retention_seconds: int = Field(ge=1)
    ceiling_seconds: int = Field(ge=1)
    legal_hold: bool

    @model_validator(mode="after")
    def _within_hard_limits(self) -> HistoryContinuityRetentionPolicy:
        if self.legal_hold is not True:
            raise ValueError("legal_hold is required")
        if self.retention_seconds > self.ceiling_seconds:
            raise ValueError("retention_seconds exceeds ceiling_seconds")
        return self


class HistoryContinuityAdministrativeIdentities(BaseModel):
    """Primary and witness administrator identifiers. Not verified principals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_administrator_id: str = _NON_BLANK
    witness_administrator_id: str = _NON_BLANK

    @field_validator("primary_administrator_id", "witness_administrator_id")
    @classmethod
    def _unambiguous(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("administrator identity is ambiguous")
        return _reject_blank(value)

    @model_validator(mode="after")
    def _independent(self) -> HistoryContinuityAdministrativeIdentities:
        if not administrator_identities_are_independent(
            self.primary_administrator_id,
            self.witness_administrator_id,
        ):
            raise ValueError("primary and witness administrators must be distinct")
        return self


class HistoryContinuityActivationPolicy(BaseModel):
    """Declarative activation constraints. Not a grant and not cloud IAM proof."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    history_scope: HistoryContinuityHistoryScope
    administrators: HistoryContinuityAdministrativeIdentities
    runtime_credential_mode: HistoryContinuityRuntimeCredentialMode
    separated_capabilities: tuple[ContinuityPermission, ...]

    @field_validator("history_scope")
    @classmethod
    def _environment_scope_only(
        cls,
        value: HistoryContinuityHistoryScope,
    ) -> HistoryContinuityHistoryScope:
        if value is not HistoryContinuityHistoryScope.ENVIRONMENT:
            raise ValueError("only environment-scoped history may activate")
        return value

    @field_validator("runtime_credential_mode")
    @classmethod
    def _supported_credential_mode(
        cls,
        value: HistoryContinuityRuntimeCredentialMode,
    ) -> HistoryContinuityRuntimeCredentialMode:
        if value is not HistoryContinuityRuntimeCredentialMode.LOCAL_NO_RUNTIME_CREDENTIAL:
            raise ValueError("runtime credential mode is not accepted")
        return value

    @field_validator("separated_capabilities")
    @classmethod
    def _capabilities_stay_distinct(
        cls,
        value: tuple[ContinuityPermission, ...],
    ) -> tuple[ContinuityPermission, ...]:
        if not separated_capabilities_are_exact(value):
            raise ValueError("operational capabilities must stay the five distinct duties")
        return value


class HistoryContinuityActivation(BaseModel):
    """Explicit adapter activation. Absence is not representable as a default-on.

    ``policy`` selects whether this gate may open. It does not authorize a
    caller, store a secret, or prove administrative independence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    adapter: HistoryContinuityAdapterKind
    application_id: str = _NON_BLANK
    environment_id: str = _NON_BLANK
    retention: HistoryContinuityRetentionPolicy
    policy: HistoryContinuityActivationPolicy

    @field_validator("enabled")
    @classmethod
    def _must_be_enabled(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("activation must be explicitly enabled")
        return value

    @field_validator("application_id", "environment_id")
    @classmethod
    def _ids_non_blank(cls, value: str) -> str:
        return _reject_blank(value)

    @model_validator(mode="after")
    def _local_adapter_only(self) -> HistoryContinuityActivation:
        if self.adapter is not HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC:
            raise ValueError("adapter is not activated for this gate")
        return self

    @model_validator(mode="after")
    def _admitted(self) -> HistoryContinuityActivation:
        if not history_continuity_activation_is_admitted(self):
            raise ValueError("history continuity activation is not admitted")
        return self


def _present_identifier(value: object) -> bool:
    return type(value) is str and bool(value.strip())


def _retention_shape_holds(retention: object) -> bool:
    if type(retention) is not HistoryContinuityRetentionPolicy:
        return False
    duration = retention.retention_seconds
    ceiling = retention.ceiling_seconds
    if type(duration) is not int or type(ceiling) is not int:
        return False
    if duration < 1 or ceiling < 1 or duration > ceiling:
        return False
    return retention.legal_hold is True


def history_continuity_activation_is_admitted(activation: object) -> bool:
    """Return whether typed activation may open the local qualification adapter.

    This is a configuration predicate. It is not caller authorization, not a
    credential check, and not proof of administrative independence.
    """

    if type(activation) is not HistoryContinuityActivation:
        return False
    if getattr(activation, "enabled", None) is not True:
        return False
    if getattr(activation, "adapter", None) is not HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC:
        return False
    if not _present_identifier(getattr(activation, "application_id", None)):
        return False
    if not _present_identifier(getattr(activation, "environment_id", None)):
        return False
    if not _retention_shape_holds(getattr(activation, "retention", None)):
        return False
    policy = getattr(activation, "policy", None)
    if type(policy) is not HistoryContinuityActivationPolicy:
        return False
    if policy.history_scope is not HistoryContinuityHistoryScope.ENVIRONMENT:
        return False
    if policy.runtime_credential_mode is not (
        HistoryContinuityRuntimeCredentialMode.LOCAL_NO_RUNTIME_CREDENTIAL
    ):
        return False
    administrators = getattr(policy, "administrators", None)
    if type(administrators) is not HistoryContinuityAdministrativeIdentities:
        return False
    if not administrator_identities_are_independent(
        getattr(administrators, "primary_administrator_id", None),
        getattr(administrators, "witness_administrator_id", None),
    ):
        return False
    return separated_capabilities_are_exact(getattr(policy, "separated_capabilities", None))


class ContinuityScope(BaseModel):
    """Environment-level history scope plus the tenant that must match it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_BLANK
    environment_id: str = _NON_BLANK
    tenant_id: str = _NON_BLANK

    @field_validator("application_id", "environment_id", "tenant_id")
    @classmethod
    def _ids_non_blank(cls, value: str) -> str:
        return _reject_blank(value)


class ContinuityPermissionGrant(BaseModel):
    """Caller claim of principal and capability. Not authorization by itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: str = _NON_BLANK
    permission: ContinuityPermission

    @field_validator("principal_id")
    @classmethod
    def _principal_non_blank(cls, value: str) -> str:
        return _reject_blank(value)


class HistoryBinding(BaseModel):
    """Opaque bound history identifiers. Not an invocation list."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pis_history_id: str = _NON_BLANK
    pis_committed_watermark: int = Field(ge=0)
    gca_history_id: str = _NON_BLANK
    gca_committed_watermark: int = Field(ge=0)

    @field_validator("pis_history_id", "gca_history_id")
    @classmethod
    def _ids_non_blank(cls, value: str) -> str:
        return _reject_blank(value)


class ContinuityHead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_BLANK
    environment_id: str = _NON_BLANK
    tenant_id: str = _NON_BLANK
    revision: int = Field(ge=1)
    fencing_token: str = _NON_BLANK
    status: ContinuityHeadStatus
    binding: HistoryBinding
    identity_epoch: int = Field(ge=1)
    proof_revision: int = Field(ge=1)

    @field_validator("application_id", "environment_id", "tenant_id", "fencing_token")
    @classmethod
    def _ids_non_blank(cls, value: str) -> str:
        return _reject_blank(value)


class WitnessSeal(BaseModel):
    """Append-only witness record. Independent from the primary head store.

    ``predecessor_seal_digest`` authenticates the previous seal. Revision 1
    carries the fixed genesis predecessor. The field is part of the seal digest.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment_id: str = _NON_BLANK
    revision: int = Field(ge=1)
    fencing_token: str = _NON_BLANK
    transition_id: str = _NON_BLANK
    binding_digest: str = _NON_BLANK
    status: ContinuityHeadStatus
    identity_epoch: int = Field(ge=1)
    predecessor_seal_digest: str = _NON_BLANK

    @field_validator(
        "environment_id",
        "fencing_token",
        "transition_id",
        "binding_digest",
        "predecessor_seal_digest",
    )
    @classmethod
    def _ids_non_blank(cls, value: str) -> str:
        return _reject_blank(value)


class ContinuityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ContinuityOutcome
    denial: ContinuityDenialCode | None = None
    head: ContinuityHead | None = None
    witness_high_watermark: int | None = None

    @model_validator(mode="after")
    def _denial_matches_outcome(self) -> ContinuityResult:
        if self.outcome is ContinuityOutcome.DENIED and self.denial is None:
            raise ValueError("denied result requires a denial code")
        if self.outcome is not ContinuityOutcome.DENIED and self.denial is not None:
            raise ValueError("non-denied result cannot carry a denial code")
        return self


class HistoryContinuityAuthority(Protocol):
    """Single continuity-state port. Callers do not select a second owner."""

    def sealed_retention(self) -> HistoryContinuityRetentionPolicy | None:
        """Return the immutable activation policy, or None when inactive."""

    def initialize(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        binding: HistoryBinding,
    ) -> ContinuityResult:
        """Create the first sealed head. Refuses an already initialized scope."""

    def advance(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        expected_revision: int,
        expected_fencing_token: str,
        binding: HistoryBinding,
    ) -> ContinuityResult:
        """Compare-and-swap the head and seal the witness."""

    def reconcile(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        expected_revision: int,
        expected_fencing_token: str,
        binding: HistoryBinding,
    ) -> ContinuityResult:
        """Authorized reconciliation from quarantine only."""

    def invalidate(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        expected_revision: int,
        expected_fencing_token: str,
        status: ContinuityHeadStatus,
    ) -> ContinuityResult:
        """Quarantine or revoke. Does not admit further writes."""

    def recover(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
    ) -> ContinuityResult:
        """Resolve a prepared transition or quarantine a diverged history."""

    def read_head(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
    ) -> ContinuityResult:
        """Read the current head. Divergence denies rather than returning an allow."""

    def attempt_retention_change(
        self,
        grant: ContinuityPermissionGrant,
        scope: ContinuityScope,
        *,
        retention_seconds: int,
        ceiling_seconds: int,
        legal_hold: bool,
    ) -> ContinuityResult:
        """Reject weakening and every other retention mutation."""
