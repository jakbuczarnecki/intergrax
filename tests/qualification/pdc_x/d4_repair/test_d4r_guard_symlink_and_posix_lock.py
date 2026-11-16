# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-D / F-D4-19 (b): the guard's symlink rejection and the POSIX lock branch run on every host.

The real-symlink tests of ``tests/unit/hosting/instance`` skip on a host that does not permit
symlink creation. The tests here stub the symlink answers so the rejection branches are always
exercised, and drive ``_native_lock`` with an injected ``fcntl`` so the POSIX branch is read by a
test even on Windows. They do not claim that Linux or macOS were exercised.
"""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest
from intergrax.hosting import InstancePolicy
from intergrax.hosting.contracts.context import HostedApplicationProcessIdentity
from intergrax.hosting.errors import HostedApplicationInstanceGuardError
from intergrax.hosting.instance import _native_lock
from intergrax.hosting.instance.contracts import HostedApplicationInstanceIdentity
from intergrax.hosting.instance.file_guard import FileHostedApplicationInstanceGuard

pytestmark = [pytest.mark.unit]


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


def _process_identity() -> HostedApplicationProcessIdentity:
    return HostedApplicationProcessIdentity(process_id=os.getpid(), host_id="host-test", started_at=_Clock().now())


def _guard(run_directory: Path) -> FileHostedApplicationInstanceGuard:
    return FileHostedApplicationInstanceGuard(
        run_directory=run_directory,
        instance_policy=InstancePolicy(),
        process_identity=_process_identity(),
        clock=_Clock(),
    )


def _identity() -> HostedApplicationInstanceIdentity:
    return HostedApplicationInstanceIdentity(
        application_id="test_app",
        instance_id="instance-001",
        profile_digest="sha256:" + "a" * 64,
        process_identity=_process_identity(),
    )


def test_a_symlink_ancestor_of_the_run_directory_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    link = tmp_path / "link"
    link.mkdir()
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == link or original(self))
    with pytest.raises(HostedApplicationInstanceGuardError, match="symlink"):
        _guard(link / "nested")


def test_a_run_directory_that_is_itself_a_symlink_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    real_lstat = Path.lstat

    def fake_lstat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        result = real_lstat(self)
        if self == run_directory:
            values = list(result)
            values[0] = stat.S_IFLNK | 0o777
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(HostedApplicationInstanceGuardError, match="symlink"):
        _guard(run_directory)


def test_a_lock_file_that_is_a_symlink_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = _guard(tmp_path)
    lock_path = guard.lock_path_for("test_app")
    lock_path.write_text("x", encoding="utf-8")
    real_lstat = Path.lstat

    def fake_lstat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        result = real_lstat(self)
        if self == lock_path:
            values = list(result)
            values[0] = stat.S_IFLNK | 0o777
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(HostedApplicationInstanceGuardError, match="symlink"):
        asyncio.run(guard.acquire(_identity()))


class _FakeFcntl(types.ModuleType):
    LOCK_EX = 2
    LOCK_NB = 4
    LOCK_UN = 8

    def __init__(self, *, busy: bool = False, unlock_fails: bool = False) -> None:
        super().__init__("fcntl")
        self.busy = busy
        self.unlock_fails = unlock_fails
        self.calls: list[tuple[int, int]] = []

    def flock(self, fd: int, operation: int) -> None:
        self.calls.append((fd, operation))
        if operation == self.LOCK_UN and self.unlock_fails:
            raise OSError("unlock failed")
        if operation != self.LOCK_UN and self.busy:
            raise OSError("would block")


def _use_posix(monkeypatch: pytest.MonkeyPatch, fake: _FakeFcntl) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "fcntl", fake)


def test_posix_branch_takes_a_non_blocking_exclusive_flock(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeFcntl()
    _use_posix(monkeypatch, fake)
    _native_lock.try_acquire_exclusive(7)
    assert fake.calls == [(7, _FakeFcntl.LOCK_EX | _FakeFcntl.LOCK_NB)]


def test_posix_branch_maps_a_busy_lock_to_lock_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_posix(monkeypatch, _FakeFcntl(busy=True))
    with pytest.raises(_native_lock.NativeFileLockError, match="lock_busy"):
        _native_lock.try_acquire_exclusive(7)


def test_posix_branch_releases_with_lock_un_and_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeFcntl()
    _use_posix(monkeypatch, fake)
    _native_lock.release_exclusive(7)
    assert fake.calls == [(7, _FakeFcntl.LOCK_UN)]
    failing = _FakeFcntl(unlock_fails=True)
    _use_posix(monkeypatch, failing)
    _native_lock.release_exclusive(7)  # must not raise
    assert failing.calls == [(7, _FakeFcntl.LOCK_UN)]
