# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference file-lock instance guard (APP-HOST-4A/4B)."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

from intergrax.hosting.contracts.context import HostedApplicationClock, HostedApplicationProcessIdentity
from intergrax.hosting.contracts.identity import normalize_application_id
from intergrax.hosting.contracts.policies import InstancePolicy
from intergrax.hosting.contracts.public_data import validate_bounded_identifier, validate_instance_id
from intergrax.hosting.instance.contracts import HostedApplicationInstanceAcquisitionResult
from intergrax.hosting.errors import (
  HostedApplicationInstanceConflictError,
  HostedApplicationInstanceGuardError,
  HostedApplicationInstanceOwnershipError,
)
from intergrax.hosting.instance._native_lock import (
  NativeFileLock,
  NativeFileLockError,
  open_lock_fd,
  stat_is_regular_file,
  try_acquire_exclusive,
)
from intergrax.hosting.instance.contracts import (
  HostedApplicationInstanceConflictSnapshot,
  HostedApplicationInstanceIdentity,
  HostedApplicationInstanceLeasePublicView,
  InstanceAcquisitionClassification,
)

_METADATA_SCHEMA_VERSION = "1.0"
_MAX_METADATA_BYTES = 4096
_LOCK_SUFFIX = ".lock"

PublishEventCallback = Callable[..., Awaitable[None] | None]


@runtime_checkable
class HostedApplicationProcessProbe(Protocol):
  """Probe whether a process identifier is still alive."""

  def is_alive(self, process_id: int) -> bool: ...


class OsProcessProbe:
  """Default process liveness probe using OS primitives."""

  def is_alive(self, process_id: int) -> bool:
    if process_id <= 0:
      return False
    if os.name == "nt":
      import ctypes

      PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
      STILL_ACTIVE = 259
      handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
      if not handle:
        return False
      try:
        exit_code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
          return False
        return exit_code.value == STILL_ACTIVE
      finally:
        ctypes.windll.kernel32.CloseHandle(handle)
    try:
      os.kill(process_id, 0)
    except ProcessLookupError:
      return False
    except PermissionError:
      return True
    except OSError:
      return False
    return True


def _validate_timezone_aware(value: datetime, *, field_name: str) -> datetime:
  if value.tzinfo is None:
    raise HostedApplicationInstanceGuardError(f"{field_name} must be timezone-aware")
  return value


def _derive_lock_filename(application_id: str) -> str:
  normalized = normalize_application_id(application_id)
  if "/" in normalized or "\\" in normalized or ".." in normalized:
    raise HostedApplicationInstanceGuardError("application_id contains path separators")
  return f"{normalized}{_LOCK_SUFFIX}"


def _path_has_symlink_ancestor(path: Path) -> bool:
  current = path
  while True:
    try:
      if current.is_symlink():
        return True
    except OSError:
      return True
    parent = current.parent
    if parent == current:
      return False
    current = parent


def _validate_run_directory(run_directory: Path) -> Path:
  if not run_directory.is_absolute():
    raise HostedApplicationInstanceGuardError("run_directory must be absolute")
  if _path_has_symlink_ancestor(run_directory):
    raise HostedApplicationInstanceGuardError("run_directory path contains a symlink ancestor")
  try:
    if run_directory.lstat().st_mode and stat.S_ISLNK(run_directory.lstat().st_mode):
      raise HostedApplicationInstanceGuardError("run_directory must not be a symlink")
  except OSError as exc:
    raise HostedApplicationInstanceGuardError("run_directory inaccessible") from exc
  resolved = run_directory.resolve(strict=False)
  if _path_has_symlink_ancestor(resolved):
    raise HostedApplicationInstanceGuardError("run_directory resolves through a symlink")
  resolved.mkdir(parents=True, exist_ok=True)
  return resolved


def _validate_lock_path_parent(lock_path: Path) -> None:
  parent = lock_path.parent
  if _path_has_symlink_ancestor(parent):
    raise HostedApplicationInstanceGuardError("lock path contains a symlink ancestor")
  try:
    parent_stat = parent.lstat()
  except OSError as exc:
    raise HostedApplicationInstanceGuardError("lock path parent inaccessible") from exc
  if stat.S_ISLNK(parent_stat.st_mode):
    raise HostedApplicationInstanceGuardError("lock path parent must not be a symlink")


@dataclass(frozen=True, slots=True)
class _LeaseMetadata:
  schema_version: str
  application_id: str
  instance_id: str
  process_id: int
  process_started_at: datetime
  host_id: str | None
  user_scope_id: str | None
  profile_digest: str
  acquired_at: datetime
  ownership_token: str

  def public_view(self) -> HostedApplicationInstanceLeasePublicView:
    return HostedApplicationInstanceLeasePublicView(
      application_id=self.application_id,
      instance_id=self.instance_id,
      process_id=self.process_id,
      process_started_at=self.process_started_at,
      host_id=self.host_id,
      user_scope_id=self.user_scope_id,
      profile_digest=self.profile_digest,
      acquired_at=self.acquired_at,
    )

  def to_json_bytes(self) -> bytes:
    payload = {
      "schema_version": self.schema_version,
      "application_id": self.application_id,
      "instance_id": self.instance_id,
      "process_id": self.process_id,
      "process_started_at": self.process_started_at.isoformat(),
      "host_id": self.host_id,
      "user_scope_id": self.user_scope_id,
      "profile_digest": self.profile_digest,
      "acquired_at": self.acquired_at.isoformat(),
      "ownership_token": self.ownership_token,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) > _MAX_METADATA_BYTES:
      raise HostedApplicationInstanceGuardError("metadata exceeds size bound")
    return encoded

  @classmethod
  def from_json_bytes(cls, raw: bytes) -> _LeaseMetadata:
    if len(raw) > _MAX_METADATA_BYTES:
      raise HostedApplicationInstanceGuardError("metadata exceeds size bound")
    if not raw:
      raise HostedApplicationInstanceGuardError("corrupted metadata")
    try:
      payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise HostedApplicationInstanceGuardError("corrupted metadata") from exc
    if not isinstance(payload, dict):
      raise HostedApplicationInstanceGuardError("corrupted metadata")
    required = (
      "schema_version",
      "application_id",
      "instance_id",
      "process_id",
      "process_started_at",
      "profile_digest",
      "acquired_at",
      "ownership_token",
    )
    for key in required:
      if key not in payload:
        raise HostedApplicationInstanceGuardError(f"metadata missing field: {key}")
    schema_version = str(payload["schema_version"])
    if schema_version != _METADATA_SCHEMA_VERSION:
      raise HostedApplicationInstanceGuardError("unsupported metadata schema version")
    application_id = str(payload["application_id"])
    validate_bounded_identifier(application_id, field_name="application_id")
    instance_id = validate_instance_id(str(payload["instance_id"]))
    try:
      process_id = int(payload["process_id"])
    except (TypeError, ValueError) as exc:
      raise HostedApplicationInstanceGuardError("invalid metadata process_id") from exc
    if process_id <= 0:
      raise HostedApplicationInstanceGuardError("invalid metadata process_id")
    profile_digest = str(payload["profile_digest"])
    validate_bounded_identifier(profile_digest, field_name="profile_digest")
    ownership_token = str(payload["ownership_token"])
    if not ownership_token:
      raise HostedApplicationInstanceGuardError("invalid metadata ownership_token")
    try:
      process_started_at = datetime.fromisoformat(str(payload["process_started_at"]))
      acquired_at = datetime.fromisoformat(str(payload["acquired_at"]))
    except (TypeError, ValueError) as exc:
      raise HostedApplicationInstanceGuardError("invalid metadata timestamp") from exc
    process_started_at = _validate_timezone_aware(process_started_at, field_name="process_started_at")
    acquired_at = _validate_timezone_aware(acquired_at, field_name="acquired_at")
    host_id = payload.get("host_id")
    user_scope_id = payload.get("user_scope_id")
    if host_id is not None:
      host_id = str(host_id)
    if user_scope_id is not None:
      user_scope_id = str(user_scope_id)
    return cls(
      schema_version=schema_version,
      application_id=application_id,
      instance_id=instance_id,
      process_id=process_id,
      process_started_at=process_started_at,
      host_id=host_id,
      user_scope_id=user_scope_id,
      profile_digest=profile_digest,
      acquired_at=acquired_at,
      ownership_token=ownership_token,
    )


@dataclass
class FileHostedApplicationInstanceLease:
  """Lease handle for a file-backed instance guard."""

  _lock: NativeFileLock
  _lock_path: Path
  _metadata: _LeaseMetadata
  _released_verified: bool = field(default=False, repr=False)

  def is_valid(self) -> bool:
    if self._released_verified or not self._lock.held:
      return False
    try:
      on_disk = _read_metadata_from_lock(self._lock)
    except HostedApplicationInstanceGuardError:
      return False
    if on_disk.ownership_token != self._metadata.ownership_token:
      return False
    if on_disk.instance_id != self._metadata.instance_id:
      return False
    return True

  def verify_ownership(self) -> None:
    if self._released_verified:
      raise HostedApplicationInstanceOwnershipError("lease already released")
    if not self._lock.held:
      raise HostedApplicationInstanceOwnershipError("native lock not held")
    try:
      on_disk = _read_metadata_from_lock(self._lock)
    except HostedApplicationInstanceGuardError as exc:
      raise HostedApplicationInstanceOwnershipError("ownership verification failed") from exc
    if on_disk.ownership_token != self._metadata.ownership_token:
      raise HostedApplicationInstanceOwnershipError("ownership token mismatch")
    if on_disk.instance_id != self._metadata.instance_id:
      raise HostedApplicationInstanceOwnershipError("instance identity mismatch")

  def public_view(self) -> HostedApplicationInstanceLeasePublicView:
    return self._metadata.public_view()

  async def release(self) -> None:
    if self._released_verified:
      return
    released = False
    try:
      self.verify_ownership()
      assert self._lock.fd is not None
      try:
        os.ftruncate(self._lock.fd, 0)
      except OSError as exc:
        raise HostedApplicationInstanceGuardError("lease_truncate_failed") from exc
      released = True
      self._released_verified = True
    except HostedApplicationInstanceOwnershipError:
      raise
    finally:
      self._lock.close()
      if not released:
        self._released_verified = False


def _read_metadata_from_lock(lock: NativeFileLock) -> _LeaseMetadata:
  if lock.fd is None:
    raise HostedApplicationInstanceGuardError("inaccessible lock metadata")
  os.lseek(lock.fd, 0, os.SEEK_SET)
  raw = os.read(lock.fd, _MAX_METADATA_BYTES)
  if not raw:
    raise HostedApplicationInstanceGuardError("corrupted metadata")
  return _LeaseMetadata.from_json_bytes(raw)


def _best_effort_read_metadata(lock_path: Path) -> _LeaseMetadata | None:
  if not lock_path.exists():
    return None
  try:
    fd = os.open(str(lock_path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
  except OSError:
    return None
  try:
    raw = os.read(fd, _MAX_METADATA_BYTES)
    if not raw:
      return None
    return _LeaseMetadata.from_json_bytes(raw)
  except HostedApplicationInstanceGuardError:
    return None
  finally:
    os.close(fd)


def _write_metadata_file(lock: NativeFileLock, metadata: _LeaseMetadata) -> None:
  encoded = metadata.to_json_bytes()
  lock.write_bytes(encoded)


def _classify_prior_metadata(
  prior: _LeaseMetadata | None,
  *,
  prior_corrupt: bool,
  identity: HostedApplicationInstanceIdentity,
  process_probe: HostedApplicationProcessProbe,
  allow_stale_recovery: bool,
) -> InstanceAcquisitionClassification:
  if prior is None and not prior_corrupt:
    return InstanceAcquisitionClassification.FRESH
  if prior is not None and process_probe.is_alive(prior.process_id):
    raise HostedApplicationInstanceConflictError(
      "ownership inconsistency with live process",
      HostedApplicationInstanceConflictSnapshot(
        application_id=identity.application_id,
        conflicting_instance_id=prior.instance_id,
        conflicting_process_id=prior.process_id,
        classification=InstanceAcquisitionClassification.OWNERSHIP_MISMATCH,
        reason_code="live_process_metadata",
      ),
    )
  if prior is not None and not allow_stale_recovery:
    raise HostedApplicationInstanceConflictError(
      "stale recovery disabled",
      HostedApplicationInstanceConflictSnapshot(
        application_id=identity.application_id,
        conflicting_instance_id=prior.instance_id,
        conflicting_process_id=prior.process_id,
        classification=InstanceAcquisitionClassification.STALE_OWNER,
        reason_code="stale_recovery_disabled",
      ),
    )
  if prior_corrupt and not allow_stale_recovery:
    raise HostedApplicationInstanceConflictError(
      "corrupt recovery disabled",
      HostedApplicationInstanceConflictSnapshot(
        application_id=identity.application_id,
        classification=InstanceAcquisitionClassification.CORRUPTED_METADATA,
        reason_code="corrupt_recovery_disabled",
      ),
    )
  if prior_corrupt:
    return InstanceAcquisitionClassification.CORRUPTED_METADATA
  return InstanceAcquisitionClassification.STALE_OWNER


@dataclass
class FileHostedApplicationInstanceGuard:
  """Portable reference instance guard using an exclusive file lock."""

  run_directory: Path
  instance_policy: InstancePolicy
  process_identity: HostedApplicationProcessIdentity
  clock: HostedApplicationClock
  process_probe: HostedApplicationProcessProbe = field(default_factory=OsProcessProbe)

  def __post_init__(self) -> None:
    self._run_directory = _validate_run_directory(self.run_directory)

  def lock_path_for(self, application_id: str) -> Path:
    filename = _derive_lock_filename(application_id)
    return self._run_directory / filename

  async def acquire(
    self,
    identity: HostedApplicationInstanceIdentity,
  ) -> HostedApplicationInstanceAcquisitionResult:
    if self.instance_policy.exclusivity_mode.value == "multi_instance":
      raise HostedApplicationInstanceGuardError("multi_instance mode does not use file guard")
    lock_path = self.lock_path_for(identity.application_id)
    _validate_lock_path_parent(lock_path)
    try:
      if lock_path.exists():
        lock_stat = lock_path.lstat()
        if stat.S_ISLNK(lock_stat.st_mode):
          raise HostedApplicationInstanceGuardError("lock file must not be a symlink")
    except OSError as exc:
      raise HostedApplicationInstanceGuardError("inaccessible lock") from exc

    try:
      fd = open_lock_fd(lock_path)
    except NativeFileLockError as exc:
      raise HostedApplicationInstanceGuardError("inaccessible lock") from exc

    try:
      if not stat_is_regular_file(fd):
        raise HostedApplicationInstanceGuardError("lock file must be a regular file")
      try_acquire_exclusive(fd)
    except NativeFileLockError as exc:
      os.close(fd)
      prior = _best_effort_read_metadata(lock_path)
      snapshot = HostedApplicationInstanceConflictSnapshot(
        application_id=identity.application_id,
        conflicting_instance_id=prior.instance_id if prior is not None else None,
        conflicting_process_id=prior.process_id if prior is not None else None,
        classification=InstanceAcquisitionClassification.ACTIVE_OWNER,
        reason_code="active_owner",
      )
      raise HostedApplicationInstanceConflictError("active instance conflict", snapshot) from exc

    native_lock = NativeFileLock.from_held_fd(lock_path, fd)
    prior_metadata: _LeaseMetadata | None = None
    prior_corrupt = False
    try:
      os.lseek(fd, 0, os.SEEK_SET)
      raw = os.read(fd, _MAX_METADATA_BYTES)
      if raw:
        try:
          prior_metadata = _LeaseMetadata.from_json_bytes(raw)
        except HostedApplicationInstanceGuardError:
          prior_corrupt = True
    except OSError as exc:
      native_lock.close()
      raise HostedApplicationInstanceGuardError("metadata read failed") from exc

    try:
      classification = _classify_prior_metadata(
        prior_metadata,
        prior_corrupt=prior_corrupt,
        identity=identity,
        process_probe=self.process_probe,
        allow_stale_recovery=self.instance_policy.allow_stale_recovery,
      )
    except HostedApplicationInstanceConflictError:
      native_lock.close()
      raise

    acquired_at = _validate_timezone_aware(self.clock.now(), field_name="acquired_at")
    metadata = _LeaseMetadata(
      schema_version=_METADATA_SCHEMA_VERSION,
      application_id=identity.application_id,
      instance_id=identity.instance_id,
      process_id=identity.process_identity.process_id,
      process_started_at=identity.process_identity.started_at,
      host_id=identity.process_identity.host_id,
      user_scope_id=identity.process_identity.user_scope_id,
      profile_digest=identity.profile_digest,
      acquired_at=acquired_at,
      ownership_token=str(uuid4()),
    )
    try:
      _write_metadata_file(native_lock, metadata)
    except OSError as exc:
      native_lock.close()
      raise HostedApplicationInstanceGuardError("metadata write failed") from exc

    lease = FileHostedApplicationInstanceLease(
      _lock=native_lock,
      _lock_path=lock_path,
      _metadata=metadata,
    )
    return HostedApplicationInstanceAcquisitionResult(lease=lease, classification=classification)


def lease_native_lock_for_tests(lease: FileHostedApplicationInstanceLease) -> NativeFileLock:
    """Test seam exposing the native lock for fault injection."""
    return lease._lock


def lease_metadata_for_tests(lease: FileHostedApplicationInstanceLease) -> _LeaseMetadata:
    """Test seam exposing lease metadata for corruption injection."""
    return lease._metadata


def lease_release_verified_for_tests(lease: FileHostedApplicationInstanceLease) -> bool:
    """Test seam reporting whether release was verified."""
    return lease._released_verified
