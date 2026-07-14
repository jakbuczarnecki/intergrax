# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform-native exclusive file lock primitives (APP-HOST-4A)."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


class NativeFileLockError(OSError):
    """Raised when native file locking fails."""


class NativeFileLock:
    """Exclusive advisory file lock held for the lifetime of the object."""

    def __init__(self, lock_path: Path) -> None:
        self._path = lock_path
        self._fd: int | None = None
        self._held = False
        open_flags = os.O_CREAT | os.O_RDWR
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        open_flags |= nofollow
        try:
            self._fd = os.open(str(lock_path), open_flags, 0o600)
        except OSError as exc:
            raise NativeFileLockError("lock_open_failed") from exc
        try:
            self._acquire_nonblocking()
            self._held = True
        except NativeFileLockError:
            os.close(self._fd)
            self._fd = None
            raise

    @property
    def held(self) -> bool:
        return self._held

    @property
    def fd(self) -> int | None:
        return self._fd

    def _acquire_nonblocking(self) -> None:
        if self._fd is None:
            raise NativeFileLockError("lock_not_open")
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise NativeFileLockError("lock_busy") from exc
        else:
            import fcntl

            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise NativeFileLockError("lock_busy") from exc

    def write_bytes(self, payload: bytes) -> None:
        if self._fd is None:
            raise NativeFileLockError("lock_not_open")
        os.ftruncate(self._fd, 0)
        os.write(self._fd, payload)
        if hasattr(os, "fchmod"):
            os.fchmod(self._fd, stat.S_IRUSR | stat.S_IWUSR)  # type: ignore[attr-defined]

    def release(self) -> None:
        if not self._held or self._fd is None:
            return
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
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
