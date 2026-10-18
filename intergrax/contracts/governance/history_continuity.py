# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-neutral ELCR-IIRW history-continuity contracts (D03-A-LOCAL-01).

The record is a governance continuity head, not a provider-invocation ledger.
No cloud backend, retention period, or CREATE admission is selected here.
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


class HistoryContinuityActivation(BaseModel):
    """Explicit adapter activation. Absence is not representable as a default-on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    adapter: HistoryContinuityAdapterKind
    application_id: str = _NON_BLANK
    environment_id: str = _NON_BLANK
    retention: HistoryContinuityRetentionPolicy

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
    """Append-only witness record. Independent from the primary head store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment_id: str = _NON_BLANK
    revision: int = Field(ge=1)
    fencing_token: str = _NON_BLANK
    transition_id: str = _NON_BLANK
    binding_digest: str = _NON_BLANK
    status: ContinuityHeadStatus
    identity_epoch: int = Field(ge=1)

    @field_validator(
        "environment_id",
        "fencing_token",
        "transition_id",
        "binding_digest",
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
