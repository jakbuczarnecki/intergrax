# © Artur Czarnecki. All rights reserved.

"""Read-only subprocess sandbox for allowlisted audit entrypoints."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import ManifestCheck

_REDACT_MARKERS = ("API_KEY", "SECRET", "PASSWORD", "TOKEN=")


@dataclass(frozen=True, slots=True)
class SandboxResult:
    exit_code: int
    duration_ms: int
    stdout: str
    stderr: str


def _tail(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _redact(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        upper = line.upper()
        if any(marker in upper for marker in _REDACT_MARKERS):
            lines.append("[REDACTED]")
        else:
            lines.append(line)
    return "\n".join(lines)


def _resolve_under_repo(repo_root: Path, entrypoint: str) -> Path:
    if "::" in entrypoint:
        rel = entrypoint.split("::", 1)[0]
    else:
        rel = entrypoint
    if ".." in Path(rel).parts:
        raise ValueError(f"entrypoint must not contain .. segments: {entrypoint}")
    candidate = (repo_root / rel).resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"entrypoint escapes repo_root: {entrypoint}") from exc
    return candidate


class CheckExecutionSandbox:
    """Execute manifest-allowlisted checks only (no shell)."""

    def __init__(self, repo_root: Path, *, timeout_seconds: int = 600) -> None:
        self._repo_root = repo_root.resolve()
        self._timeout = timeout_seconds

    def run_check(self, check: ManifestCheck) -> SandboxResult:
        if check.mutating:
            raise ValueError(f"mutating check not allowed in default sandbox: {check.check_id}")
        _resolve_under_repo(self._repo_root, check.entrypoint)
        started = time.perf_counter()
        if check.entrypoint_kind == "pytest_nodeid":
            cmd = ["uv", "run", "pytest", check.entrypoint, "-q", "--tb=line", "--no-header"]
        elif check.entrypoint_kind == "pytest_file":
            cmd = ["uv", "run", "pytest", check.entrypoint, "-q", "--tb=line", "--no-header"]
        elif check.entrypoint_kind == "python_script":
            cmd = ["uv", "run", "python", check.entrypoint]
        else:
            raise ValueError(f"unsupported entrypoint_kind for sandbox: {check.entrypoint_kind}")
        popen_kwargs: dict = {
            "cwd": self._repo_root,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "shell": False,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=self._timeout)
            exit_code = proc.returncode or 0
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc.pid)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return SandboxResult(
                exit_code=124,
                duration_ms=elapsed_ms,
                stdout="",
                stderr="TIMEOUT",
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return SandboxResult(
            exit_code=exit_code,
            duration_ms=elapsed_ms,
            stdout=_redact(_tail(stdout or "")),
            stderr=_redact(_tail(stderr or "")),
        )


def _kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)
    else:
        try:
            os.killpg(os.getpgid(pid), 9)
        except ProcessLookupError:
            os.kill(pid, 9)
