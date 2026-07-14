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
    FileHostedApplicationInstanceLease,
    OsProcessProbe,
    lease_metadata_for_tests,
    lease_native_lock_for_tests,
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
    acquisition = await guard.acquire(_identity())
    assert acquisition.classification is InstanceAcquisitionClassification.FRESH
    lease = acquisition.lease
    public = lease.public_view()
    assert public.instance_id == "instance-001"
    assert "ownership_token" not in public.model_dump()
    await lease.release()
    assert lease.is_valid() is False


@pytest.mark.asyncio
async def test_second_active_owner_rejected(tmp_path: Path) -> None:
    import intergrax.hosting.instance._native_lock as native_lock_module
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

    def _busy_acquire(_fd: int) -> None:
        raise NativeFileLockError("lock_busy")

    original = native_lock_module.try_acquire_exclusive
    native_lock_module.try_acquire_exclusive = _busy_acquire
    try:
        with pytest.raises(HostedApplicationInstanceConflictError):
            await guard.acquire(_identity("instance-002"))
    finally:
        native_lock_module.try_acquire_exclusive = original


@pytest.mark.asyncio
async def test_idempotent_release_and_token_mismatch(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    lease = (await guard.acquire(_identity())).lease
    assert isinstance(lease, FileHostedApplicationInstanceLease)
    metadata = lease_metadata_for_tests(lease)
    corrupted = _LeaseMetadata(
        schema_version=metadata.schema_version,
        application_id=metadata.application_id,
        instance_id=metadata.instance_id,
        process_id=metadata.process_id,
        process_started_at=metadata.process_started_at,
        host_id=metadata.host_id,
        user_scope_id=metadata.user_scope_id,
        profile_digest=metadata.profile_digest,
        acquired_at=metadata.acquired_at,
        ownership_token="other-token",
    )
    lease_native_lock_for_tests(lease).write_bytes(corrupted.to_json_bytes())
    with pytest.raises(HostedApplicationInstanceOwnershipError):
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
    lease1 = (await guard.acquire(_identity("instance-old"))).lease
    assert isinstance(lease1, FileHostedApplicationInstanceLease)
    lease_native_lock_for_tests(lease1).close()
    object.__setattr__(lease1, "_released_verified", True)
    acquisition2 = await guard.acquire(_identity("instance-new"))
    assert acquisition2.classification is InstanceAcquisitionClassification.STALE_OWNER
    lease2 = acquisition2.lease
    await lease2.release()


@pytest.mark.asyncio
async def test_stale_recovery_disabled(tmp_path: Path) -> None:
    guard = _guard(tmp_path, allow_stale_recovery=False)
    object.__setattr__(guard, "process_probe", _DeadProbe())
    lease1 = (await guard.acquire(_identity("instance-old"))).lease
    assert isinstance(lease1, FileHostedApplicationInstanceLease)
    lease_native_lock_for_tests(lease1).close()
    object.__setattr__(lease1, "_released_verified", True)
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
