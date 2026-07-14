# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference file-lock instance guard (APP-HOST-4A/4B)."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

from intergrax.hosting.contracts.context import HostedApplicationClock, HostedApplicationProcessIdentity
from intergrax.hosting.contracts.identity import normalize_application_id
from intergrax.hosting.contracts.policies import InstancePolicy
from intergrax.hosting.instance.contracts import HostedApplicationInstanceIdentity
from intergrax.hosting.errors import (
  HostedApplicationInstanceConflictError,
  HostedApplicationInstanceGuardError,
  HostedApplicationInstanceOwnershipError,
)
from intergrax.hosting.instance._native_lock import NativeFileLock, NativeFileLockError
from intergrax.hosting.instance.contracts import (
  HostedApplicationInstanceConflictSnapshot,
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
    raise ValueError(f"{field_name} must be timezone-aware")
  return value


def _derive_lock_filename(application_id: str) -> str:
  normalized = normalize_application_id(application_id)
  if "/" in normalized or "\\" in normalized or ".." in normalized:
    raise HostedApplicationInstanceGuardError("application_id contains path separators")
  return f"{normalized}{_LOCK_SUFFIX}"


def _validate_run_directory(run_directory: Path) -> Path:
  if not run_directory.is_absolute():
    raise HostedApplicationInstanceGuardError("run_directory must be absolute")
  if run_directory.is_symlink():
    raise HostedApplicationInstanceGuardError("run_directory must not be a symlink")
  resolved = run_directory.resolve(strict=False)
  if resolved.is_symlink():
    raise HostedApplicationInstanceGuardError("run_directory resolves to a symlink")
  resolved.mkdir(parents=True, exist_ok=True)
  return resolved


def _validate_lock_file(lock_path: Path) -> None:
  if lock_path.exists():
    if lock_path.is_symlink():
      raise HostedApplicationInstanceGuardError("lock file must not be a symlink")
    if not lock_path.is_file():
      raise HostedApplicationInstanceGuardError("lock file must be a regular file")


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
    try:
      payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise HostedApplicationInstanceGuardError("corrupted metadata") from exc
    if not isinstance(payload, dict):
      raise HostedApplicationInstanceGuardError("corrupted metadata")
    return cls(
      schema_version=str(payload["schema_version"]),
      application_id=str(payload["application_id"]),
      instance_id=str(payload["instance_id"]),
      process_id=int(payload["process_id"]),
      process_started_at=datetime.fromisoformat(str(payload["process_started_at"])),
      host_id=payload.get("host_id"),
      user_scope_id=payload.get("user_scope_id"),
      profile_digest=str(payload["profile_digest"]),
      acquired_at=datetime.fromisoformat(str(payload["acquired_at"])),
      ownership_token=str(payload["ownership_token"]),
    )


@dataclass
class FileHostedApplicationInstanceLease:
  """Lease handle for a file-backed instance guard."""

  _lock: NativeFileLock
  _lock_path: Path
  _metadata: _LeaseMetadata
  _released: bool = field(default=False, repr=False)

  def is_valid(self) -> bool:
    return self._lock.held and not self._released

  def verify_ownership(self) -> None:
    if self._released:
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
    if self._released:
      return
    try:
      self.verify_ownership()
      _clear_metadata_file(self._lock_path, self._metadata, lock=self._lock)
    except HostedApplicationInstanceOwnershipError:
      pass
    finally:
      self._lock.close()
      self._released = True


def _read_metadata_file(lock_path: Path) -> _LeaseMetadata:
    _validate_lock_file(lock_path)
    fd = os.open(str(lock_path), os.O_RDONLY)
    try:
        raw = os.read(fd, _MAX_METADATA_BYTES)
    finally:
        os.close(fd)
    if not raw:
        raise HostedApplicationInstanceGuardError("corrupted metadata")
    return _LeaseMetadata.from_json_bytes(raw)


def _read_metadata_from_lock(lock: NativeFileLock) -> _LeaseMetadata:
    if lock.fd is None:
        raise HostedApplicationInstanceGuardError("inaccessible lock metadata")
    os.lseek(lock.fd, 0, os.SEEK_SET)
    raw = os.read(lock.fd, _MAX_METADATA_BYTES)
    if not raw:
        raise HostedApplicationInstanceGuardError("corrupted metadata")
    return _LeaseMetadata.from_json_bytes(raw)


def _write_metadata_file(lock: NativeFileLock, metadata: _LeaseMetadata) -> None:
  encoded = metadata.to_json_bytes()
  lock.write_bytes(encoded)


def _clear_metadata_file(lock_path: Path, expected: _LeaseMetadata, *, lock: NativeFileLock | None = None) -> None:
    if lock is not None and lock.fd is not None:
        try:
            on_disk = _read_metadata_from_lock(lock)
        except HostedApplicationInstanceGuardError:
            return
        if on_disk.ownership_token != expected.ownership_token:
            return
        if on_disk.instance_id != expected.instance_id:
            return
        os.ftruncate(lock.fd, 0)
        return
    try:
        on_disk = _read_metadata_file(lock_path)
    except HostedApplicationInstanceGuardError:
        return
    if on_disk.ownership_token != expected.ownership_token:
        return
    if on_disk.instance_id != expected.instance_id:
        return
    fd = os.open(str(lock_path), os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.ftruncate(fd, 0)
    finally:
        os.close(fd)


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
  ) -> tuple[FileHostedApplicationInstanceLease, InstanceAcquisitionClassification]:
    if self.instance_policy.exclusivity_mode.value == "multi_instance":
      raise HostedApplicationInstanceGuardError("multi_instance mode does not use file guard")
    lock_path = self.lock_path_for(identity.application_id)
    if lock_path.exists():
      _validate_lock_file(lock_path)

    prior_metadata: _LeaseMetadata | None = None
    prior_corrupt = False
    if lock_path.exists():
      try:
        prior_metadata = _read_metadata_file(lock_path)
      except HostedApplicationInstanceGuardError:
        prior_corrupt = True

    try:
      native_lock = NativeFileLock(lock_path)
    except NativeFileLockError as exc:
      if prior_metadata is not None and self.process_probe.is_alive(prior_metadata.process_id):
        snapshot = HostedApplicationInstanceConflictSnapshot(
          application_id=identity.application_id,
          conflicting_instance_id=prior_metadata.instance_id,
          conflicting_process_id=prior_metadata.process_id,
          classification=InstanceAcquisitionClassification.ACTIVE_OWNER,
          reason_code="active_owner",
        )
        raise HostedApplicationInstanceConflictError("active instance conflict", snapshot) from exc
      raise HostedApplicationInstanceGuardError("inaccessible lock") from exc

    classification = InstanceAcquisitionClassification.FRESH
    if prior_metadata is not None:
      if self.process_probe.is_alive(prior_metadata.process_id):
        native_lock.close()
        snapshot = HostedApplicationInstanceConflictSnapshot(
          application_id=identity.application_id,
          conflicting_instance_id=prior_metadata.instance_id,
          conflicting_process_id=prior_metadata.process_id,
          classification=InstanceAcquisitionClassification.OWNERSHIP_MISMATCH,
          reason_code="live_process_metadata",
        )
        raise HostedApplicationInstanceConflictError(
          "ownership inconsistency with live process",
          snapshot,
        )
      if not self.instance_policy.allow_stale_recovery:
        native_lock.close()
        snapshot = HostedApplicationInstanceConflictSnapshot(
          application_id=identity.application_id,
          conflicting_instance_id=prior_metadata.instance_id,
          conflicting_process_id=prior_metadata.process_id,
          classification=InstanceAcquisitionClassification.STALE_OWNER,
          reason_code="stale_recovery_disabled",
        )
        raise HostedApplicationInstanceConflictError("stale recovery disabled", snapshot)
      classification = InstanceAcquisitionClassification.STALE_OWNER
    elif prior_corrupt:
      if not self.instance_policy.allow_stale_recovery:
        native_lock.close()
        snapshot = HostedApplicationInstanceConflictSnapshot(
          application_id=identity.application_id,
          classification=InstanceAcquisitionClassification.CORRUPTED_METADATA,
          reason_code="corrupt_recovery_disabled",
        )
        raise HostedApplicationInstanceConflictError("corrupt recovery disabled", snapshot)
      classification = InstanceAcquisitionClassification.CORRUPTED_METADATA

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
    return lease, classification
