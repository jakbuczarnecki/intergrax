# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.hosting import InstancePolicy
from intergrax.hosting.contracts.context import HostedApplicationProcessIdentity
from intergrax.hosting.instance.contracts import HostedApplicationInstanceIdentity
from intergrax.hosting.errors import (
    HostedApplicationInstanceConflictError,
    HostedApplicationInstanceGuardError,
    HostedApplicationInstanceOwnershipError,
)
from intergrax.hosting.instance.file_guard import (
    FileHostedApplicationInstanceGuard,
    OsProcessProbe,
    _LeaseMetadata,
    _METADATA_SCHEMA_VERSION,
)
from intergrax.hosting.instance.contracts import InstanceAcquisitionClassification
from tests.unit.hosting.engine._fakes import FixedClock

pytestmark = pytest.mark.unit


def _identity(instance_id: str = "instance-001") -> HostedApplicationInstanceIdentity:
    clock = FixedClock()
    return HostedApplicationInstanceIdentity(
        application_id="test_app",
        instance_id=instance_id,
        profile_digest="sha256:" + "a" * 64,
        process_identity=HostedApplicationProcessIdentity(
            process_id=os.getpid(),
            host_id="host-test",
            started_at=clock.now(),
        ),
    )


def _guard(tmp_path: Path, *, allow_stale_recovery: bool = True) -> FileHostedApplicationInstanceGuard:
    clock = FixedClock()
    return FileHostedApplicationInstanceGuard(
        run_directory=tmp_path,
        instance_policy=InstancePolicy(allow_stale_recovery=allow_stale_recovery),
        process_identity=HostedApplicationProcessIdentity(
            process_id=os.getpid(),
            host_id="host-test",
            started_at=clock.now(),
        ),
        clock=clock,
    )


@pytest.mark.asyncio
async def test_fresh_acquire_and_public_view(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    lease, classification = await guard.acquire(_identity())
    assert classification is InstanceAcquisitionClassification.FRESH
    public = lease.public_view()
    assert public.instance_id == "instance-001"
    assert "ownership_token" not in public.model_dump()
    await lease.release()
    assert lease.is_valid() is False


@pytest.mark.asyncio
async def test_second_active_owner_rejected(tmp_path: Path) -> None:
    import intergrax.hosting.instance.file_guard as file_guard_module
    from intergrax.hosting.instance._native_lock import NativeFileLockError

    guard = _guard(tmp_path)
    lock_path = guard.lock_path_for("test_app")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _LeaseMetadata(
        schema_version=_METADATA_SCHEMA_VERSION,
        application_id="test_app",
        instance_id="instance-001",
        process_id=424242,
        process_started_at=datetime.now(UTC),
        host_id="host",
        user_scope_id=None,
        profile_digest="sha256:" + "d" * 64,
        acquired_at=datetime.now(UTC),
        ownership_token="existing-token",
    )
    lock_path.write_bytes(metadata.to_json_bytes())
    object.__setattr__(guard, "process_probe", _LiveProbe())

    class _BusyNativeLock:
        def __init__(self, _path: object) -> None:
            raise NativeFileLockError("lock_busy")

    original = file_guard_module.NativeFileLock
    file_guard_module.NativeFileLock = _BusyNativeLock
    try:
        with pytest.raises(HostedApplicationInstanceConflictError):
            await guard.acquire(_identity("instance-002"))
    finally:
        file_guard_module.NativeFileLock = original


@pytest.mark.asyncio
async def test_idempotent_release_and_token_mismatch(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    lease, _ = await guard.acquire(_identity())
    corrupted = _LeaseMetadata(
        schema_version=lease._metadata.schema_version,
        application_id=lease._metadata.application_id,
        instance_id=lease._metadata.instance_id,
        process_id=lease._metadata.process_id,
        process_started_at=lease._metadata.process_started_at,
        host_id=lease._metadata.host_id,
        user_scope_id=lease._metadata.user_scope_id,
        profile_digest=lease._metadata.profile_digest,
        acquired_at=lease._metadata.acquired_at,
        ownership_token="other-token",
    )
    lease._lock.write_bytes(corrupted.to_json_bytes())
    with pytest.raises(HostedApplicationInstanceOwnershipError):
        lease.verify_ownership()
    await lease.release()
    await lease.release()


class _DeadProbe(OsProcessProbe):
    def is_alive(self, process_id: int) -> bool:
        return False


class _LiveProbe(OsProcessProbe):
    def is_alive(self, process_id: int) -> bool:
        return True


@pytest.mark.asyncio
async def test_stale_owner_recovered(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    object.__setattr__(guard, "process_probe", _DeadProbe())
    lease1, _ = await guard.acquire(_identity("instance-old"))
    lease1._lock.close()
    object.__setattr__(lease1, "_released", True)
    lease2, classification = await guard.acquire(_identity("instance-new"))
    assert classification is InstanceAcquisitionClassification.STALE_OWNER
    await lease2.release()


@pytest.mark.asyncio
async def test_stale_recovery_disabled(tmp_path: Path) -> None:
    guard = _guard(tmp_path, allow_stale_recovery=False)
    object.__setattr__(guard, "process_probe", _DeadProbe())
    lease1, _ = await guard.acquire(_identity("instance-old"))
    await lease1.release()
    with pytest.raises(HostedApplicationInstanceConflictError):
        await guard.acquire(_identity("instance-new"))


@pytest.mark.asyncio
async def test_live_process_metadata_with_free_lock_rejected(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    lock_path = guard.lock_path_for("test_app")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _LeaseMetadata(
        schema_version=_METADATA_SCHEMA_VERSION,
        application_id="test_app",
        instance_id="instance-live",
        process_id=999999,
        process_started_at=datetime.now(UTC),
        host_id="host",
        user_scope_id=None,
        profile_digest="sha256:" + "b" * 64,
        acquired_at=datetime.now(UTC),
        ownership_token="token-live",
    )
    lock_path.write_bytes(metadata.to_json_bytes())
    object.__setattr__(guard, "process_probe", _LiveProbe())
    with pytest.raises(HostedApplicationInstanceConflictError):
        await guard.acquire(_identity())


def test_symlinked_run_directory_rejected(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(HostedApplicationInstanceGuardError):
        _guard(link)


def test_metadata_size_bound() -> None:
    metadata = _LeaseMetadata(
        schema_version=_METADATA_SCHEMA_VERSION,
        application_id="test_app",
        instance_id="instance-001",
        process_id=1,
        process_started_at=datetime.now(UTC),
        host_id="h",
        user_scope_id=None,
        profile_digest="sha256:" + "c" * 64,
        acquired_at=datetime.now(UTC),
        ownership_token="t" * 4000,
    )
    with pytest.raises(HostedApplicationInstanceGuardError):
        metadata.to_json_bytes()
