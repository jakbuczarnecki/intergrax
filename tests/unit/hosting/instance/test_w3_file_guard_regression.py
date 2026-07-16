# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.hosting import InstancePolicy
from intergrax.hosting.contracts.context import HostedApplicationProcessIdentity
from intergrax.hosting.engine.ports import HostedApplicationInstanceGuardPort
from intergrax.hosting.instance.contracts import (
    HostedApplicationInstanceAcquisitionResult,
    HostedApplicationInstanceIdentity,
    InstanceAcquisitionClassification,
)
from intergrax.hosting.errors import (
    HostedApplicationInstanceConflictError,
    HostedApplicationInstanceGuardError,
    HostedApplicationInstanceOwnershipError,
)
from intergrax.hosting.instance.file_guard import (
    FileHostedApplicationInstanceGuard,
    FileHostedApplicationInstanceLease,
    OsProcessProbe,
    lease_native_lock_for_tests,
    lease_release_verified_for_tests,
    _LeaseMetadata,
    _METADATA_SCHEMA_VERSION,
)
from intergrax.hosting.instance._native_lock import NativeFileLockError
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


class _DeadProbe(OsProcessProbe):
    def is_alive(self, process_id: int) -> bool:
        return False


@pytest.mark.asyncio
async def test_port_conformance(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    assert isinstance(guard, HostedApplicationInstanceGuardPort)
    result = await guard.acquire(_identity())
    assert isinstance(result, HostedApplicationInstanceAcquisitionResult)
    await result.lease.release()


@pytest.mark.asyncio
async def test_metadata_reread_after_native_lock(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    result = await guard.acquire(_identity())
    assert result.lease.is_valid() is True
    await result.lease.release()


@pytest.mark.asyncio
async def test_busy_native_lock_conflict(tmp_path: Path) -> None:
    import intergrax.hosting.instance.file_guard as file_guard_module
    import intergrax.hosting.instance._native_lock as native_lock_module

    guard = _guard(tmp_path)
    lock_path = guard.lock_path_for("test_app")
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
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(metadata.to_json_bytes())
    original = file_guard_module.try_acquire_exclusive

    def busy(_fd: int) -> None:
        raise NativeFileLockError("lock_busy")

    file_guard_module.try_acquire_exclusive = busy
    native_lock_module.try_acquire_exclusive = busy
    try:
        with pytest.raises(HostedApplicationInstanceConflictError) as exc_info:
            await guard.acquire(_identity("instance-002"))
        from intergrax.hosting.instance.contracts import HostedApplicationInstanceConflictSnapshot

        snapshot = exc_info.value.snapshot
        assert isinstance(snapshot, HostedApplicationInstanceConflictSnapshot)
        assert snapshot.classification is InstanceAcquisitionClassification.ACTIVE_OWNER
    finally:
        file_guard_module.try_acquire_exclusive = original
        native_lock_module.try_acquire_exclusive = original


def test_unsupported_schema(tmp_path: Path) -> None:
    with pytest.raises(HostedApplicationInstanceGuardError):
        _LeaseMetadata.from_json_bytes(
            b'{"schema_version":"9.9","application_id":"test_app","instance_id":"instance-001",'
            b'"process_id":1,"process_started_at":"2026-07-14T12:00:00+00:00",'
            b'"profile_digest":"sha256:' + b"a" * 64 + b'",'
            b'"acquired_at":"2026-07-14T12:00:00+00:00","ownership_token":"tok"}'
        )


def test_missing_metadata_fields() -> None:
    with pytest.raises(HostedApplicationInstanceGuardError):
        _LeaseMetadata.from_json_bytes(b'{"schema_version":"1.0"}')


def test_invalid_timestamps() -> None:
    raw = (
        '{"schema_version":"1.0","application_id":"test_app","instance_id":"instance-001",'
        '"process_id":1,"process_started_at":"bad","profile_digest":"sha256:'
        + "a" * 64
        + '","acquired_at":"2026-07-14T12:00:00+00:00","ownership_token":"tok"}'
    )
    with pytest.raises(HostedApplicationInstanceGuardError):
        _LeaseMetadata.from_json_bytes(raw.encode())


def test_symlinked_ancestor_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    nested = link / "nested"
    with pytest.raises(HostedApplicationInstanceGuardError):
        _guard(nested)


def test_symlinked_lock_file_rejected(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    lock_path = guard.lock_path_for("test_app")
    target = tmp_path / "target.lock"
    target.write_text("x", encoding="utf-8")
    lock_path.symlink_to(target)
    with pytest.raises(HostedApplicationInstanceGuardError):
        import asyncio

        asyncio.run(guard.acquire(_identity()))


@pytest.mark.asyncio
async def test_ownership_token_mismatch_release(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    lease = (await guard.acquire(_identity())).lease
    assert isinstance(lease, FileHostedApplicationInstanceLease)
    lease_native_lock_for_tests(lease).write_bytes(
        _LeaseMetadata(
            schema_version=_METADATA_SCHEMA_VERSION,
            application_id="test_app",
            instance_id="instance-001",
            process_id=os.getpid(),
            process_started_at=datetime.now(UTC),
            host_id="host",
            user_scope_id=None,
            profile_digest="sha256:" + "a" * 64,
            acquired_at=datetime.now(UTC),
            ownership_token="other",
        ).to_json_bytes()
    )
    assert lease.is_valid() is False
    lease_native_lock_for_tests(lease).close()
    with pytest.raises(HostedApplicationInstanceOwnershipError):
        await lease.release()


@pytest.mark.asyncio
async def test_is_valid_detects_changed_metadata(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    lease = (await guard.acquire(_identity())).lease
    assert isinstance(lease, FileHostedApplicationInstanceLease)
    lease_native_lock_for_tests(lease).write_bytes(
        _LeaseMetadata(
            schema_version=_METADATA_SCHEMA_VERSION,
            application_id="test_app",
            instance_id="instance-001",
            process_id=os.getpid(),
            process_started_at=datetime.now(UTC),
            host_id="host",
            user_scope_id=None,
            profile_digest="sha256:" + "a" * 64,
            acquired_at=datetime.now(UTC),
            ownership_token="changed",
        ).to_json_bytes()
    )
    assert lease.is_valid() is False
    lease_native_lock_for_tests(lease).close()


@pytest.mark.asyncio
async def test_failed_release_not_reported_as_released(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    lease = (await guard.acquire(_identity())).lease
    assert isinstance(lease, FileHostedApplicationInstanceLease)
    lease_native_lock_for_tests(lease).write_bytes(b"corrupt")
    with pytest.raises(HostedApplicationInstanceOwnershipError):
        await lease.release()
    assert lease.is_valid() is False
    lease_native_lock_for_tests(lease).close()


@pytest.mark.asyncio
async def test_stale_recovery_emits_recovered_then_acquired_classification(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    object.__setattr__(guard, "process_probe", _DeadProbe())
    lease1 = (await guard.acquire(_identity("old"))).lease
    assert isinstance(lease1, FileHostedApplicationInstanceLease)
    lease_native_lock_for_tests(lease1).close()
    result = await guard.acquire(_identity("new"))
    assert result.classification is InstanceAcquisitionClassification.STALE_OWNER


@pytest.mark.asyncio
async def test_ftruncate_failure_closes_handle_without_verified_release(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    lease = (await guard.acquire(_identity())).lease
    assert isinstance(lease, FileHostedApplicationInstanceLease)
    native = lease_native_lock_for_tests(lease)
    original_ftruncate = os.ftruncate

    def failing_ftruncate(fd: int, length: int) -> None:
        raise OSError("ftruncate failed")

    os.ftruncate = failing_ftruncate  # type: ignore[assignment]
    try:
        with pytest.raises(HostedApplicationInstanceGuardError):
            await lease.release()
    finally:
        os.ftruncate = original_ftruncate
    assert lease_release_verified_for_tests(lease) is False
    assert native.held is False
