# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public instance ownership contracts (APP-HOST-4A/4B)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.hosting.contracts.context import HostedApplicationProcessIdentity
from intergrax.hosting.contracts.public_data import validate_bounded_identifier, validate_instance_id


class InstanceAcquisitionClassification(str, Enum):
  FRESH = "fresh"
  ACTIVE_OWNER = "active_owner"
  STALE_OWNER = "stale_owner"
  CORRUPTED_METADATA = "corrupted_metadata"
  INACCESSIBLE_LOCK = "inaccessible_lock"
  OWNERSHIP_MISMATCH = "ownership_mismatch"


class HostedApplicationInstanceIdentity(BaseModel):
  """Immutable instance identity used for lease acquisition."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  application_id: str
  instance_id: str
  profile_digest: str
  process_identity: HostedApplicationProcessIdentity


class HostedApplicationInstanceLeasePublicView(BaseModel):
  """Safe public projection of an instance lease without ownership secrets."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  application_id: str
  instance_id: str
  process_id: int
  process_started_at: datetime
  host_id: str | None = None
  user_scope_id: str | None = None
  profile_digest: str
  acquired_at: datetime

  @field_validator("instance_id")
  @classmethod
  def _validate_instance_id(cls, value: str) -> str:
    return validate_instance_id(value)


class HostedApplicationInstanceConflictSnapshot(BaseModel):
  """Safe snapshot describing an active instance conflict."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  application_id: str
  conflicting_instance_id: str | None = None
  conflicting_process_id: int | None = None
  classification: InstanceAcquisitionClassification
  reason_code: str

  @field_validator("reason_code")
  @classmethod
  def _validate_reason_code(cls, value: str) -> str:
    return validate_bounded_identifier(value, field_name="reason_code")


@runtime_checkable
class HostedApplicationInstanceLeasePort(Protocol):
  """Lease handle returned by the instance guard."""

  def is_valid(self) -> bool: ...

  def verify_ownership(self) -> None: ...

  def public_view(self) -> HostedApplicationInstanceLeasePublicView: ...

  async def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class HostedApplicationInstanceAcquisitionResult:
  """Immutable instance guard acquisition outcome."""

  lease: HostedApplicationInstanceLeasePort
  classification: InstanceAcquisitionClassification
