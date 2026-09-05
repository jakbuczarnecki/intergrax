# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Controlled sandbox session for risky operations (architecture §21)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import FrozenSet
from uuid import uuid4

from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities
from intergrax.runtime.sandbox.models import (
    SandboxAuditEntry,
    SandboxExecutionResult,
    SandboxSessionManifest,
)
from intergrax.runtime.sandbox.sandbox_runtime import DEFAULT_SANDBOX_OPERATIONS
from intergrax.utils.time_provider import SystemTimeProvider


def _safe_relative_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {relative_path}")
    return path


class SandboxSession:
    """
    Disposable, permission-controlled execution session.

    Supports allowlisted file and echo operations inside an isolated root directory.
    """

    def __init__(
        self,
        *,
        session_id: str,
        root: Path,
        tenant_id: str,
        task_id: str,
        created_at_utc: str,
        allowed_operations: FrozenSet[str] | None = None,
    ) -> None:
        self.session_id = session_id
        self.root = root
        self.tenant_id = tenant_id
        self.task_id = task_id
        self.created_at_utc = created_at_utc
        self._allowed_operations = allowed_operations or DEFAULT_SANDBOX_OPERATIONS
        self._audit: list[SandboxAuditEntry] = []
        self._cancelled = False
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        base_dir: Path,
        *,
        tenant_id: str,
        task_id: str,
        session_id: str | None = None,
        allowed_operations: FrozenSet[str] | None = None,
    ) -> SandboxSession:
        sid = session_id or f"sbox_{uuid4().hex[:16]}"
        root = base_dir / tenant_id / task_id / sid
        return cls(
            session_id=sid,
            root=root,
            tenant_id=tenant_id,
            task_id=task_id,
            created_at_utc=SystemTimeProvider.utc_now().isoformat(),
            allowed_operations=allowed_operations,
        )

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def allowed_operations(self) -> frozenset[str]:
        """Public allowlist surface for operation-level capability evidence."""
        return frozenset(self._allowed_operations)

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        """Honest local substrate evidence — workspace isolation, not OS-network proof."""
        return SandboxSecurityCapabilities(
            isolation_tier="local",
            provider_id=f"local:{self.session_id}",
            network_egress_deny_enforced="browser_fetch" not in self._allowed_operations,
        )

    @property
    def audit_log(self) -> list[SandboxAuditEntry]:
        return list(self._audit)

    def cancel(self) -> None:
        self._cancelled = True

    def execute(self, operation: str, payload: dict | None = None) -> SandboxExecutionResult:
        started = time.perf_counter()
        payload = dict(payload or {})
        entry_id = f"sbox_audit_{uuid4().hex[:10]}"
        started_at = SystemTimeProvider.utc_now().isoformat()

        if self._cancelled:
            entry = SandboxAuditEntry(
                entry_id=entry_id,
                operation=operation,
                status="cancelled",
                started_at_utc=started_at,
                duration_ms=int((time.perf_counter() - started) * 1000),
                error="sandbox_cancelled",
            )
            self._audit.append(entry)
            return SandboxExecutionResult(
                success=False,
                error="sandbox_cancelled",
                audit_entry=entry,
            )

        if operation not in self._allowed_operations:
            entry = SandboxAuditEntry(
                entry_id=entry_id,
                operation=operation,
                status="denied",
                started_at_utc=started_at,
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=f"operation_not_allowed:{operation}",
            )
            self._audit.append(entry)
            return SandboxExecutionResult(
                success=False,
                error=f"operation_not_allowed:{operation}",
                audit_entry=entry,
            )

        try:
            output = self._dispatch(operation, payload)
        except Exception as exc:  # noqa: BLE001 — sandbox boundary
            entry = SandboxAuditEntry(
                entry_id=entry_id,
                operation=operation,
                status="failed",
                started_at_utc=started_at,
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )
            self._audit.append(entry)
            return SandboxExecutionResult(
                success=False,
                error=str(exc),
                audit_entry=entry,
            )

        entry = SandboxAuditEntry(
            entry_id=entry_id,
            operation=operation,
            status="success",
            started_at_utc=started_at,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        self._audit.append(entry)
        return SandboxExecutionResult(success=True, output=output, audit_entry=entry)

    def manifest(self) -> SandboxSessionManifest:
        return SandboxSessionManifest(
            session_id=self.session_id,
            tenant_id=self.tenant_id,
            task_id=self.task_id,
            root_path=str(self.root),
            created_at_utc=self.created_at_utc,
            allowed_operations=sorted(self._allowed_operations),
            operation_count=len(self._audit),
            cancelled=self._cancelled,
        )

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def exists_on_disk(self) -> bool:
        return self.root.exists()

    def _dispatch(self, operation: str, payload: dict) -> dict:
        if operation == "echo":
            message = str(payload.get("message", ""))
            return {"message": message}
        if operation == "write_file":
            rel = str(payload.get("path", ""))
            content = str(payload.get("content", ""))
            target = self.root / _safe_relative_path(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"path": rel, "size_bytes": len(content.encode("utf-8"))}
        if operation == "read_file":
            rel = str(payload.get("path", ""))
            target = self.root / _safe_relative_path(rel)
            return {"path": rel, "content": target.read_text(encoding="utf-8")}
        if operation == "list_files":
            files: list[str] = []
            if self.root.exists():
                for path in sorted(self.root.rglob("*")):
                    if path.is_file():
                        files.append(path.relative_to(self.root).as_posix())
            return {"files": files}
        if operation == "run_python":
            return self._run_python(payload)
        if operation == "run_script":
            return self._run_script(payload)
        if operation == "browser_fetch":
            return self._browser_fetch(payload)
        raise ValueError(f"unsupported operation: {operation}")

    def _run_python(self, payload: dict) -> dict:
        code = str(payload.get("code", ""))
        language = str(payload.get("language", "python")).strip().lower()
        if language != "python":
            raise ValueError(f"unsupported language: {language}")
        timeout_s = max(1, min(int(payload.get("timeout_s", 30)), 120))
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }

    def _run_script(self, payload: dict) -> dict:
        rel = str(payload.get("path", ""))
        interpreter = str(payload.get("interpreter", sys.executable))
        args = [str(item) for item in payload.get("args", []) if str(item)]
        timeout_s = max(1, min(int(payload.get("timeout_s", 60)), 300))
        script_path = self.root / _safe_relative_path(rel)
        if not script_path.is_file():
            raise FileNotFoundError(f"script not found: {rel}")
        completed = subprocess.run(
            [interpreter, script_path.as_posix(), *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "path": rel,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }

    def _browser_fetch(self, payload: dict) -> dict:
        url = str(payload.get("url", "")).strip()
        if not url:
            raise ValueError("url is required")
        max_chars = max(256, min(int(payload.get("max_chars", 50_000)), 200_000))
        timeout_s = max(1, min(int(payload.get("timeout_s", 30)), 120))
        request = urllib.request.Request(url, headers={"User-Agent": "IntergraxSandbox/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.URLError as exc:
            raise ValueError(f"fetch failed: {exc}") from exc
        text = raw.decode("utf-8", errors="replace")
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]
        return {
            "url": url,
            "content": text,
            "content_type": content_type,
            "truncated": truncated,
            "size_bytes": len(raw),
        }
