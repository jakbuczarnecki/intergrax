# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Remote sandbox session bridge for ``SandboxHostBackend`` integrations."""

from __future__ import annotations

import shlex
import time
from uuid import uuid4

from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
from intergrax.runtime.sandbox.models import SandboxAuditEntry, SandboxExecutionResult
from intergrax.runtime.sandbox.sandbox_runtime import DEFAULT_SANDBOX_OPERATIONS
from intergrax.utils.time_provider import SystemTimeProvider


class HostedSandboxSession:
    """
    Cloud sandbox session delegating operations to ``SandboxHostBackend``.

    Maps local allowlisted operations to remote shell commands.
    """

    def __init__(
        self,
        *,
        session_id: str,
        backend: SandboxHostBackend,
        tenant_id: str,
        task_id: str,
        allowed_operations: frozenset[str] | None = None,
    ) -> None:
        self.session_id = session_id
        self.tenant_id = tenant_id
        self.task_id = task_id
        self._backend = backend
        self._allowed_operations = allowed_operations or DEFAULT_SANDBOX_OPERATIONS
        self._audit: list[SandboxAuditEntry] = []
        self._cancelled = False

    @classmethod
    def open(
        cls,
        backend: SandboxHostBackend,
        *,
        tenant_id: str,
        task_id: str,
        allowed_operations: frozenset[str] | None = None,
    ) -> HostedSandboxSession:
        session = backend.create_session()
        return cls(
            session_id=session.session_id,
            backend=backend,
            tenant_id=tenant_id,
            task_id=task_id,
            allowed_operations=allowed_operations,
        )

    @property
    def cancelled(self) -> bool:
        return self._cancelled

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
            return SandboxExecutionResult(success=False, error="sandbox_cancelled", audit_entry=entry)

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
            command = self._command_for_operation(operation, payload)
            remote = self._backend.exec(self.session_id, command)
            output = {
                "stdout": remote.stdout,
                "stderr": remote.stderr,
                "exit_code": remote.exit_code,
            }
            if operation == "echo":
                output["message"] = remote.stdout.strip()
            success = remote.exit_code == 0
            error = "" if success else remote.stderr or f"remote_exit_code:{remote.exit_code}"
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
            return SandboxExecutionResult(success=False, error=str(exc), audit_entry=entry)

        entry = SandboxAuditEntry(
            entry_id=entry_id,
            operation=operation,
            status="success" if success else "failed",
            started_at_utc=started_at,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=error if not success else "",
        )
        self._audit.append(entry)
        return SandboxExecutionResult(success=success, output=output, error=error, audit_entry=entry)

    def _command_for_operation(self, operation: str, payload: dict[str, object]) -> str:
        if operation == "echo":
            message = str(payload.get("message", ""))
            return f"echo {shlex.quote(message)}"
        if operation == "write_file":
            rel = str(payload.get("path", ""))
            content = str(payload.get("content", ""))
            quoted_path = shlex.quote(rel)
            quoted_content = shlex.quote(content)
            return f"mkdir -p $(dirname {quoted_path}) && printf %s {quoted_content} > {quoted_path}"
        if operation == "read_file":
            rel = str(payload.get("path", ""))
            return f"cat {shlex.quote(rel)}"
        if operation == "list_files":
            return "find . -type f | sort"
        raise ValueError(f"unsupported operation: {operation}")
