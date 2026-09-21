# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production CodeCraftBoundCapabilityExecutionPort over sandbox code_exec (UCA-6C-R4)."""

from __future__ import annotations

from dataclasses import replace

from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionOutcome,
    CodeCraftBoundCapabilityExecutionRequest,
    CodeCraftBoundCapabilityExecutionResult,
)
from intergrax.contracts.execution_identity import require_active_execution_id
from intergrax.runtime.codecraft.ephemeral_registry import get_ephemeral_registry_store
from intergrax.runtime.codecraft.orchestrator import resolve_codecraft_profile
from intergrax.runtime.codecraft.ownership import (
    CodeCraftOwnershipError,
    CodeCraftSessionOwnership,
    resolve_codecraft_exec_authorization,
)
from intergrax.runtime.codecraft.sandbox_resolver import (
    resolve_craft_sandbox_with_evidence,
)
from intergrax.runtime.codecraft.session_manager import (
    CodeCraftSessionManager,
    get_session_manager,
)
from intergrax.tools.providers.sandbox.contracts import CodeExecInput
from intergrax.tools.providers.sandbox.extended_service import code_exec
from intergrax.tools.registry.wiring import ToolWiringContext


class WiringCodeCraftBoundCapabilityExecution:
    """Execute bound craft artifacts without global ToolRegistry mutation."""

    def __init__(
        self,
        wiring_context: ToolWiringContext,
        *,
        session_manager: CodeCraftSessionManager | None = None,
    ) -> None:
        self._ctx = wiring_context
        self._sessions = session_manager or get_session_manager(wiring_context)
        self.runtime_execution_calls = 0

    def execute(
        self,
        request: CodeCraftBoundCapabilityExecutionRequest,
    ) -> CodeCraftBoundCapabilityExecutionResult:
        active_execution_id = require_active_execution_id()
        if active_execution_id != request.execution_id:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.REJECTED,
                reason_detail="execution_identity_mismatch",
            )

        ownership = CodeCraftSessionOwnership(
            tenant_id=request.tenant_id,
            task_id=str(request.task_id),
            run_id=request.run_id,
        )
        try:
            session = self._sessions.get_owned(request.craft_id, ownership)
        except CodeCraftOwnershipError as exc:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.REJECTED,
                reason_detail=exc.code,
            )

        if session is None or session.disposed:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.UNAVAILABLE,
                reason_detail="codecraft_artifact_unavailable",
            )

        tools = (
            get_ephemeral_registry_store(self._ctx)
            .for_craft(request.craft_id)
            .list_tools()
        )
        if not tools:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.UNAVAILABLE,
                reason_detail="codecraft_ephemeral_artifact_empty",
            )

        profile = resolve_codecraft_profile(self._ctx)
        if profile is None:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.UNAVAILABLE,
                reason_detail="codecraft_profile_missing",
            )

        exec_auth = resolve_codecraft_exec_authorization(
            self._ctx,
            profile=profile,
            ownership=ownership,
            craft_id=request.craft_id,
        )
        if exec_auth.denied:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.REJECTED,
                reason_detail=exec_auth.error or "hitl_denied",
            )
        if exec_auth.pending_hitl or not exec_auth.authorized:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.REJECTED,
                reason_detail=exec_auth.error or "hitl_pending",
            )

        code = session.code
        if not code.strip():
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.FAILED,
                reason_detail="craft_code_empty",
            )

        if not profile.exec_allowed():
            self.runtime_execution_calls += 1
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.SUCCEEDED,
                reason_detail="exec_not_required_for_mode",
            )

        sandbox_resolution = resolve_craft_sandbox_with_evidence(
            self._ctx,
            profile,
            tenant_id=ownership.tenant_id,
            task_id=ownership.task_id,
        )
        sandbox = sandbox_resolution.session
        if sandbox is None:
            detail = sandbox_resolution.error or "sandbox_session_not_configured"
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.UNAVAILABLE,
                reason_detail=detail,
            )

        self.runtime_execution_calls += 1
        exec_ctx = replace(self._ctx, sandbox_session=sandbox)
        sandbox.execute(
            "write_file",
            {"path": "craft_main.py", "content": code},
        )
        effective_timeout = int(
            max(1.0, min(120.0, profile.remaining_exec_time_s(session.total_exec_time_s))),
        )
        exec_out = code_exec(
            exec_ctx,
            CodeExecInput(
                code=code,
                language=session.language,
                timeout_s=effective_timeout,
            ),
        )
        if exec_out.success:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.SUCCEEDED,
            )
        stderr = str((exec_out.output or {}).get("stderr") or exec_out.error or "")
        return CodeCraftBoundCapabilityExecutionResult(
            outcome=CodeCraftBoundCapabilityExecutionOutcome.FAILED,
            reason_detail=stderr or "code_exec_failed",
        )


__all__ = ["WiringCodeCraftBoundCapabilityExecution"]
