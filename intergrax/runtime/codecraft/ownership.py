# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraft session ownership and local execution-gate results (ECC-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.execution_identity import peek_active_execution_identity
from intergrax.tools.registry.wiring import ToolWiringContext

CODECRAFT_EXEC_HITL_NOTES_PREFIX = "codecraft_exec:"


class CodeCraftOwnershipError(Exception):
    """Fail-closed ownership or scope resolution error."""

    def __init__(self, code: str, *, message: str = "") -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True, slots=True)
class CodeCraftSessionOwnership:
    tenant_id: str
    task_id: str
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class CodeCraftExecAuthorization:
    """
    Local CodeCraft profile gate result only.

    When ``authorized`` is True, the local supervised/HITL profile gate does not block
    the attempt; it does **not** mean platform governance, MSE, or ToolRuntime
    authorization exists. Does not mint platform execution permission and never reads
    HumanDecisionPersistence as an authorization source.
    """

    authorized: bool
    pending_hitl: bool = False
    denied: bool = False
    error: str = ""


def codecraft_exec_hitl_notes(craft_id: str) -> str:
    return f"{CODECRAFT_EXEC_HITL_NOTES_PREFIX}{craft_id}"


def _validate_caller_identity_field(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise CodeCraftOwnershipError(f"codecraft_{field}_blank")
    return stripped


def resolve_codecraft_ownership(
    ctx: ToolWiringContext,
    *,
    caller_tenant_id: str | None = None,
    caller_task_id: str | None = None,
    caller_run_id: str | None = None,
) -> CodeCraftSessionOwnership:
    """Resolve trusted tenant/task from sandbox binding and run_id from active execution identity."""
    sandbox = ctx.sandbox_session
    if sandbox is None:
        raise CodeCraftOwnershipError("codecraft_execution_scope_unavailable")

    trusted_tenant = str(sandbox.tenant_id)
    trusted_task = str(sandbox.task_id)

    resolved_caller_tenant = _validate_caller_identity_field(caller_tenant_id, field="tenant")
    resolved_caller_task = _validate_caller_identity_field(caller_task_id, field="task")
    if resolved_caller_tenant is not None and resolved_caller_tenant != trusted_tenant:
        raise CodeCraftOwnershipError("codecraft_tenant_mismatch")
    if resolved_caller_task is not None and resolved_caller_task != trusted_task:
        raise CodeCraftOwnershipError("codecraft_task_mismatch")

    trusted_run: str | None = None
    active = peek_active_execution_identity()
    if active is not None:
        trusted_run = str(active[0])

    if caller_run_id and trusted_run and caller_run_id != trusted_run:
        raise CodeCraftOwnershipError("codecraft_run_mismatch")

    return CodeCraftSessionOwnership(
        tenant_id=trusted_tenant,
        task_id=trusted_task,
        run_id=trusted_run,
    )


def matches_session_ownership(
    session_tenant_id: str,
    session_task_id: str,
    session_run_id: str | None,
    ownership: CodeCraftSessionOwnership,
) -> bool:
    if session_tenant_id != ownership.tenant_id:
        return False
    if session_task_id != ownership.task_id:
        return False
    return session_run_id == ownership.run_id


def resolve_codecraft_exec_authorization(
    ctx: ToolWiringContext,
    *,
    profile: CodeCraftProfile,
    ownership: CodeCraftSessionOwnership,
    craft_id: str,
) -> CodeCraftExecAuthorization:
    """
    Shared supervised-profile gate for iterate and codecraft.run execution paths.

    Persisted human decisions are evidence only; this gate never grants execution from
    the human decision store as an authorization source. When HITL is required by the
    profile, fail closed until a sanctioned typed canonical authority contract exists
    (not implemented on CodeCraft wiring-bound paths in this release).
    """
    _ = ctx
    _ = craft_id
    needs_hitl = profile.mode == "supervised" or profile.require_hitl_before_exec
    if not needs_hitl:
        return CodeCraftExecAuthorization(authorized=True)

    if ownership.run_id is None:
        return CodeCraftExecAuthorization(authorized=False, pending_hitl=True, error="hitl_pending")

    return CodeCraftExecAuthorization(
        authorized=False,
        pending_hitl=True,
        error="hitl_pending",
    )
