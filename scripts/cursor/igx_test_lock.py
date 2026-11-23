#!/usr/bin/env python3
"""Run a command under one exclusive lock shared by all worktrees of a repo.

Usage: igx_test_lock.py [--log PATH] -- <cmd...>
The lock file lives in the MAIN checkout (.tmp/factory/pytest.lock), so
parallel workers in different worktrees serialise their pytest runs.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def main_root(cwd: Path) -> Path:
    done = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if done.returncode or not done.stdout.strip():
        return cwd
    return Path(done.stdout.strip()).parent


def lock_path(cwd: Path) -> Path:
    return main_root(cwd) / ".tmp" / "factory" / "pytest.lock"


def acquire(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.5)
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def release(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv: list[str]) -> int:
    if "--" not in argv:
        print("usage: igx_test_lock.py [--log PATH] -- <cmd...>", file=sys.stderr)
        return 2
    split = argv.index("--")
    head, cmd = argv[:split], argv[split + 1 :]
    log: Path | None = None
    if head[:1] == ["--log"] and len(head) == 2:
        log = Path(head[1])
    elif head:
        print("usage: igx_test_lock.py [--log PATH] -- <cmd...>", file=sys.stderr)
        return 2
    if not cmd:
        print("no command given", file=sys.stderr)
        return 2
    path = lock_path(Path.cwd())
    path.parent.mkdir(parents=True, exist_ok=True)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        acquire(handle)
        try:
            logfile = open(log, "w", encoding="utf-8") if log else None  # noqa: SIM115
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    if logfile:
                        logfile.write(line)
                        logfile.flush()
                return proc.wait()
            finally:
                if logfile:
                    logfile.close()
        finally:
            release(handle)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
