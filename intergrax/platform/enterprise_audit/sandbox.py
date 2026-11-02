# © Artur Czarnecki. All rights reserved.

"""Read-only subprocess sandbox for allowlisted audit entrypoints."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import ManifestCheck


@dataclass(frozen=True, slots=True)
class SandboxResult:
    exit_code: int
    duration_ms: int
    stdout: str
    stderr: str


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


class CheckExecutionSandbox:
    """Execute manifest-allowlisted checks only (no shell)."""

    def __init__(self, repo_root: Path, *, timeout_seconds: int = 600) -> None:
        self._repo_root = repo_root.resolve()
        self._timeout = timeout_seconds

    def run_check(self, check: ManifestCheck) -> SandboxResult:
        if check.mutating:
            raise ValueError(f"mutating check not allowed in default sandbox: {check.check_id}")
        started = time.perf_counter()
        if check.entrypoint_kind == "pytest_nodeid":
            cmd = ["uv", "run", "pytest", check.entrypoint, "-q", "--tb=line", "--no-header"]
        elif check.entrypoint_kind == "pytest_file":
            cmd = ["uv", "run", "pytest", check.entrypoint, "-q", "--tb=line", "--no-header"]
        elif check.entrypoint_kind == "python_script":
            cmd = ["uv", "run", "python", check.entrypoint]
        else:
            raise ValueError(f"unsupported entrypoint_kind for sandbox: {check.entrypoint_kind}")
        completed = subprocess.run(
            cmd,
            cwd=self._repo_root,
            capture_output=True,
            text=True,
            timeout=self._timeout,
            check=False,
            shell=False,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return SandboxResult(
            exit_code=completed.returncode,
            duration_ms=elapsed_ms,
            stdout=_tail(completed.stdout),
            stderr=_tail(completed.stderr),
        )
