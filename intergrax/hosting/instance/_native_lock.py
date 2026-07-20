# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform-native exclusive file lock primitives (APP-HOST-4A)."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

from intergrax.utils import attribute_access


class NativeFileLockError(OSError):
    """Raised when native file locking fails."""


def open_lock_fd(lock_path: Path) -> int:
    """Open or create the lock file without acquiring the native lock."""
    open_flags = os.O_CREAT | os.O_RDWR
    nofollow = attribute_access.optional(os, "O_NOFOLLOW", 0)
    open_flags |= nofollow
    try:
        return os.open(str(lock_path), open_flags, 0o600)
    except OSError as exc:
        raise NativeFileLockError("lock_open_failed") from exc


def stat_is_regular_file(fd: int) -> bool:
    """Return whether the opened descriptor refers to a regular file."""
    file_stat = os.fstat(fd)
    return stat.S_ISREG(file_stat.st_mode)


def try_acquire_exclusive(fd: int) -> None:
    """Acquire an exclusive non-blocking lock on an open descriptor."""
    if sys.platform == "win32":
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise NativeFileLockError("lock_busy") from exc
    else:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise NativeFileLockError("lock_busy") from exc


def release_exclusive(fd: int) -> None:
    """Release an exclusive lock held on an open descriptor."""
    if sys.platform == "win32":
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


class NativeFileLock:
    """Exclusive advisory file lock held for the lifetime of the object."""

    def __init__(self, lock_path: Path) -> None:
        self._path = lock_path
        self._fd: int | None = None
        self._held = False
        fd = open_lock_fd(lock_path)
        try:
            if not stat_is_regular_file(fd):
                raise NativeFileLockError("not_regular_file")
            try_acquire_exclusive(fd)
            self._fd = fd
            self._held = True
        except NativeFileLockError:
            os.close(fd)
            raise

    @classmethod
    def from_held_fd(cls, lock_path: Path, fd: int) -> NativeFileLock:
        """Wrap an already-open descriptor with an acquired exclusive lock."""
        lock = cls.__new__(cls)
        lock._path = lock_path
        lock._fd = fd
        lock._held = True
        return lock

    @property
    def held(self) -> bool:
        return self._held

    @property
    def fd(self) -> int | None:
        return self._fd

    def write_bytes(self, payload: bytes) -> None:
        if self._fd is None:
            raise NativeFileLockError("lock_not_open")
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, payload)
        if hasattr(os, "fchmod"):
            os.fchmod(self._fd, stat.S_IRUSR | stat.S_IWUSR)  # type: ignore[attr-defined]

    def release(self) -> None:
        if not self._held or self._fd is None:
            return
        release_exclusive(self._fd)
        self._held = False

    def close(self) -> None:
        self.release()
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> NativeFileLock:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
