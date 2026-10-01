# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraft session ownership and canonical execution authorization (ECC-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.execution_identity import peek_active_execution_identity
from intergrax.runtime.human.models import HumanDecisionRecord, HumanResponseVerdict
from intergrax.tools.registry.runtime_bindings import HumanDecisionStoreBinding
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


def _decision_matches_craft_scope(
    record: HumanDecisionRecord,
    *,
    ownership: CodeCraftSessionOwnership,
    craft_id: str,
) -> bool:
    if record.tenant_id != ownership.tenant_id:
        return False
    if record.task_id != ownership.task_id:
        return False
    if record.run_id != ownership.run_id:
        return False
    expected = codecraft_exec_hitl_notes(craft_id)
    return record.notes == expected or record.notes.startswith(f"{expected}:")


def resolve_codecraft_exec_authorization(
    ctx: ToolWiringContext,
    *,
    profile: CodeCraftProfile,
    ownership: CodeCraftSessionOwnership,
    craft_id: str,
) -> CodeCraftExecAuthorization:
    """Shared HITL gate for iterate and codecraft.run execution paths."""
    needs_hitl = profile.mode == "supervised" or profile.require_hitl_before_exec
    if not needs_hitl:
        return CodeCraftExecAuthorization(authorized=True)

    if ownership.run_id is None:
        return CodeCraftExecAuthorization(authorized=False, pending_hitl=True, error="hitl_pending")

    store = ctx.human_decision_store
    if store is None or not isinstance(store, HumanDecisionStoreBinding):
        return CodeCraftExecAuthorization(authorized=False, pending_hitl=True, error="hitl_pending")

    decisions = store.list_for_task(ownership.task_id, ownership.tenant_id)
    scoped = [
        item
        for item in decisions
        if isinstance(item, HumanDecisionRecord)
        and _decision_matches_craft_scope(item, ownership=ownership, craft_id=craft_id)
    ]

    for record in scoped:
        if record.verdict is HumanResponseVerdict.REJECT:
            return CodeCraftExecAuthorization(authorized=False, denied=True, error="hitl_denied")

    for record in scoped:
        if record.verdict is HumanResponseVerdict.APPROVE:
            return CodeCraftExecAuthorization(authorized=True)

    return CodeCraftExecAuthorization(authorized=False, pending_hitl=True, error="hitl_pending")
