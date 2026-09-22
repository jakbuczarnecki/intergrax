# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production CodeCraftBoundCapabilityExecutionPort via canonical ToolRuntime (UCA-6C-R5)."""

from __future__ import annotations

from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionOutcome,
    CodeCraftBoundCapabilityExecutionRequest,
    CodeCraftBoundCapabilityExecutionResult,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    derive_qualified_capability_governance_step_id,
)
from intergrax.contracts.execution_identity import (
    require_active_execution_identity,
    require_active_execution_id,
    validate_run_id,
)
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
from intergrax.runtime.sandbox.isolation_errors import SandboxIsolationRequiredError
from intergrax.tools.execution_models import ToolExecutionResult
from intergrax.tools.invocation_wiring import (
    FixedSandboxSessionWiringResolver,
    ToolWiringResolutionError,
)
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from intergrax.tools.providers.sandbox.contracts import CodeExecInput, SandboxExecOutput
from intergrax.tools.registry.wiring import ToolWiringContext
from pydantic import BaseModel


class WiringCodeCraftBoundCapabilityExecution:
    """Execute bound craft artifacts through catalog ``code.exec`` (no registry mutation)."""

    def __init__(
        self,
        wiring_context: ToolWiringContext,
        *,
        catalog_tool_invoker: ExecutionBoundCatalogToolInvoker | None = None,
        session_manager: CodeCraftSessionManager | None = None,
    ) -> None:
        self._ctx = wiring_context
        self._catalog_tool_invoker = catalog_tool_invoker
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

        if self._catalog_tool_invoker is None:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.UNAVAILABLE,
                reason_detail="canonical_tool_invocation_unconfigured",
            )

        self.runtime_execution_calls += 1
        active_run_id, _ = require_active_execution_identity()
        run_id_str = validate_run_id(str(active_run_id))
        step_id = derive_qualified_capability_governance_step_id(
            request.execution_request_id,
        )
        effective_timeout = int(
            max(
                1.0,
                min(120.0, profile.remaining_exec_time_s(session.total_exec_time_s)),
            ),
        )
        wiring_resolver = FixedSandboxSessionWiringResolver(sandbox_session=sandbox)
        caller_agent_id = self._catalog_tool_invoker.caller_agent_id
        self._catalog_tool_invoker.bind_execution_identity(
            tenant_id=request.tenant_id,
            run_id=run_id_str,
            task_id=str(request.task_id),
            agent_id=caller_agent_id,
        )
        invoke_request = ExecutionBoundCatalogToolInvokeRequest(
            tool_id=CODE_EXEC_TOOL_ID,
            input=CodeExecInput(
                code=code,
                language=session.language,
                timeout_s=effective_timeout,
            ),
            tenant_id=request.tenant_id,
            task_id=str(request.task_id),
            run_id=run_id_str,
            agent_id=caller_agent_id,
            step_id=step_id,
            correlation_request_id=str(request.execution_id),
            wiring_resolver=wiring_resolver,
            governance_approval_evidence=request.governance_approval_evidence,
        )
        try:
            tool_result = self._catalog_tool_invoker.invoke(invoke_request)
        except SandboxIsolationRequiredError as exc:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.UNAVAILABLE,
                reason_detail=str(exc),
            )
        except ToolWiringResolutionError as exc:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.UNAVAILABLE,
                reason_detail=exc.code,
            )
        return _map_tool_execution_result(tool_result)


def _map_tool_execution_result(
    tool_result: ToolExecutionResult[BaseModel],
) -> CodeCraftBoundCapabilityExecutionResult:
    if tool_result.success:
        output = tool_result.output
        if isinstance(output, SandboxExecOutput) and output.success:
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.SUCCEEDED,
            )
        if isinstance(output, SandboxExecOutput):
            stderr = str(output.output.get("stderr") or output.error or "")
            return CodeCraftBoundCapabilityExecutionResult(
                outcome=CodeCraftBoundCapabilityExecutionOutcome.FAILED,
                reason_detail=stderr or "code_exec_failed",
            )
        return CodeCraftBoundCapabilityExecutionResult(
            outcome=CodeCraftBoundCapabilityExecutionOutcome.FAILED,
            reason_detail="code_exec_unexpected_output",
        )

    if tool_result.error is None:
        return CodeCraftBoundCapabilityExecutionResult(
            outcome=CodeCraftBoundCapabilityExecutionOutcome.FAILED,
            reason_detail="code_exec_failed",
        )

    code_value = str(tool_result.error.error_code)
    message = tool_result.error.error_message
    reason_detail = message if message else code_value
    if code_value in {"permission_error", "policy_error"}:
        return CodeCraftBoundCapabilityExecutionResult(
            outcome=CodeCraftBoundCapabilityExecutionOutcome.REJECTED,
            reason_detail=reason_detail,
        )
    if code_value == "validation_error":
        return CodeCraftBoundCapabilityExecutionResult(
            outcome=CodeCraftBoundCapabilityExecutionOutcome.FAILED,
            reason_detail=reason_detail,
        )
    return CodeCraftBoundCapabilityExecutionResult(
        outcome=CodeCraftBoundCapabilityExecutionOutcome.FAILED,
        reason_detail=reason_detail,
    )


__all__ = ["WiringCodeCraftBoundCapabilityExecution"]
