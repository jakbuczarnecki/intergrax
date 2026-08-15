# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Installation slot and record contracts (AGENT_DISTRIBUTION §11)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.trust import AgentInstallationTrustRecord

_NON_EMPTY = Field(min_length=1)

SCHEMA_AGENT_INSTALLATION_RECORD_V1: Final = "agent_installation_record.v1"

_INSTALLED_STATES = frozenset(
    {
        "installed_active",
        "installed_previous",
        "revoked",
    }
)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class InstallationState(StrEnum):
    """Installation record substates (§7.2)."""

    CANDIDATE = "candidate"
    VERIFIED = "verified"
    INSTALLED_ACTIVE = "installed_active"
    INSTALLED_PREVIOUS = "installed_previous"
    FAILED_CANDIDATE = "failed_candidate"
    REVOKED = "revoked"
    REMOVED_TOMBSTONE = "removed_tombstone"


def installation_state_is_installed(state: InstallationState) -> bool:
    """Return whether the substate counts as INSTALLED per §7.2."""
    return state.value in _INSTALLED_STATES


class AgentInstallationRecord(BaseModel):
    """Digest-pinned installation revision for one environment slot (§11.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_INSTALLATION_RECORD_V1
    installation_id: str = _NON_EMPTY
    installation_slot_id: str = _NON_EMPTY
    environment_id: str = _NON_EMPTY
    package_identity: AgentPackageIdentity
    installation_state: InstallationState
    active_for_slot: bool = False
    previous_installation_ref: str | None = None
    artifact_store_ref: str | None = None
    materialization_evidence_ref: str | None = None
    trust_record: AgentInstallationTrustRecord | None = None
    created_at: datetime | None = None
    superseded_at: datetime | None = None
    tombstoned_at: datetime | None = None

    @field_validator(
        "installation_id",
        "installation_slot_id",
        "environment_id",
        "previous_installation_ref",
        "artifact_store_ref",
        "materialization_evidence_ref",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @model_validator(mode="after")
    def _validate_lifecycle_invariants(self) -> AgentInstallationRecord:
        if self.installation_state is InstallationState.INSTALLED_ACTIVE and not self.active_for_slot:
            raise ValueError("installed_active records must set active_for_slot=true")
        if self.installation_state is InstallationState.INSTALLED_PREVIOUS and self.active_for_slot:
            raise ValueError("installed_previous records must not be active_for_slot")
        if self.installation_state is InstallationState.REMOVED_TOMBSTONE:
            if self.tombstoned_at is None:
                raise ValueError("removed_tombstone requires tombstoned_at")
            if self.active_for_slot:
                raise ValueError("removed_tombstone records cannot be active_for_slot")
        if self.installation_state in {
            InstallationState.CANDIDATE,
            InstallationState.VERIFIED,
            InstallationState.FAILED_CANDIDATE,
        } and self.active_for_slot:
            raise ValueError("non-installed states cannot be active_for_slot")
        if (
            self.installation_state in {InstallationState.INSTALLED_ACTIVE, InstallationState.INSTALLED_PREVIOUS}
            and self.artifact_store_ref is None
        ):
            raise ValueError("installed records require artifact_store_ref")
        return self
