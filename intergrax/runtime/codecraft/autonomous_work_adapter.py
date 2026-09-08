# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraft provider adapter for AW-7B A1 ephemeral capability execution.

Implements ``WorkerEphemeralCapabilityExecutionPort`` using canonical
``CodeCraftOrchestrator`` public APIs. Lifecycle, sandbox, HITL, and ephemeral
registry ownership remain in CodeCraft/runtime — not Autonomous Work core.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from intergrax.codecraft.contracts import CraftResult
from intergrax.contracts.autonomous_work.ephemeral_capability_execution import (
    EPHEMERAL_EXECUTION_POLICY_VERSION,
    WorkerEphemeralCapabilityExecutionReasonCode,
    WorkerEphemeralCapabilityExecutionRequest,
    WorkerEphemeralCapabilityExecutionResult,
    WorkerEphemeralCapabilityExecutionStatus,
    WorkerEphemeralCapabilityReference,
)
from intergrax.runtime.codecraft.ephemeral_registry import get_ephemeral_registry_store
from intergrax.runtime.codecraft.orchestrator import CodeCraftOrchestrator, resolve_codecraft_profile
from intergrax.tools.registry.wiring import ToolWiringContext

_UNAVAILABLE_ERRORS: frozenset[str] = frozenset(
    {
        "codecraft_profile_missing",
        "sandbox_session_not_configured",
        "isolation_requirement_unsatisfied",
        "network_egress_requirement_unsatisfied",
        "network_egress_allowlist_requirement_unsatisfied",
        "hosted_substrate_unavailable",
    },
)

_HITL_PENDING_ERRORS: frozenset[str] = frozenset({"hitl_pending"})

_HITL_DENIED_ERRORS: frozenset[str] = frozenset({"hitl_denied"})


class CodeCraftEphemeralCapabilityExecutionAdapter:
    """Canonical CodeCraft-backed AW-7B execution adapter."""

    def __init__(self, wiring_context: ToolWiringContext) -> None:
        self._ctx = wiring_context

    def execute(
        self,
        request: WorkerEphemeralCapabilityExecutionRequest,
    ) -> WorkerEphemeralCapabilityExecutionResult:
        correlation = request.correlation
        tenant_id = correlation.tenant_id
        task_id = correlation.task_id
        run_id = str(correlation.run_id) if correlation.run_id is not None else None
        trace_correlation = run_id or f"aw7b_{uuid4().hex[:12]}"
        profile = resolve_codecraft_profile(self._ctx)
        if profile is None:
            return _provider_result(
                request,
                status=WorkerEphemeralCapabilityExecutionStatus.UNAVAILABLE,
                reason_code=WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_UNAVAILABLE,
                error_code="codecraft_profile_missing",
                trace_correlation=trace_correlation,
            )

        orch = CodeCraftOrchestrator(self._ctx, run_id=trace_correlation)
        session, deny = orch.start(
            goal=request.generation_goal,
            task_id=task_id,
            tenant_id=tenant_id,
            constraints=request.constraints,
            craft_id=request.idempotency_key,
        )
        if deny is not None:
            return _map_craft_result(
                request,
                deny,
                trace_correlation=trace_correlation,
                dispose=lambda: None,
            )
        if session is None:
            return _provider_result(
                request,
                status=WorkerEphemeralCapabilityExecutionStatus.UNAVAILABLE,
                reason_code=WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_UNAVAILABLE,
                error_code="codecraft_start_failed",
                trace_correlation=trace_correlation,
            )

        craft_id = session.craft_id

        def dispose() -> None:
            orch.dispose(craft_id, tenant_id=tenant_id, task_id=task_id)

        last_result: CraftResult | None = None
        for _ in range(profile.max_iterations):
            _session, last_result = orch.iterate(
                craft_id=craft_id,
                task_id=task_id,
                tenant_id=tenant_id,
            )
            mapped = _map_iterate_progress(
                request,
                ctx=self._ctx,
                craft_id=craft_id,
                result=last_result,
                orch=orch,
                tenant_id=tenant_id,
                task_id=task_id,
                trace_correlation=trace_correlation,
                dispose=dispose,
            )
            if mapped is not None:
                return mapped

        dispose()
        error = last_result.error if last_result is not None else "max_iterations_exceeded"
        return _provider_result(
            request,
            status=WorkerEphemeralCapabilityExecutionStatus.FAILED,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_FAILED,
            error_code=error or "max_iterations_exceeded",
            craft_correlation=craft_id,
            trace_correlation=trace_correlation,
        )


def _map_iterate_progress(
    request: WorkerEphemeralCapabilityExecutionRequest,
    *,
    ctx: ToolWiringContext,
    craft_id: str,
    result: CraftResult,
    orch: CodeCraftOrchestrator,
    tenant_id: str,
    task_id: str,
    trace_correlation: str,
    dispose: Callable[[], None],
) -> WorkerEphemeralCapabilityExecutionResult | None:
    if result.error in _HITL_PENDING_ERRORS:
        return _provider_result(
            request,
            status=WorkerEphemeralCapabilityExecutionStatus.PENDING_HITL,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_PENDING_HITL,
            error_code=result.error,
            craft_correlation=craft_id,
            trace_correlation=trace_correlation,
        )

    if result.success and result.verdict == "promote":
        promote_result = orch.promote(
            craft_id,
            tenant_id=tenant_id,
            task_id=task_id,
        )
        if not promote_result.success:
            dispose()
            return _provider_result(
                request,
                status=WorkerEphemeralCapabilityExecutionStatus.FAILED,
                reason_code=WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_FAILED,
                error_code=promote_result.error or "promotion_failed",
                craft_correlation=craft_id,
                trace_correlation=trace_correlation,
            )
        ephemeral_ref = _resolve_ephemeral_reference(ctx, craft_id)
        return _provider_result(
            request,
            status=WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.A1_EXECUTION_SUCCEEDED,
            ephemeral_capability=ephemeral_ref,
            craft_correlation=craft_id,
            trace_correlation=trace_correlation,
        )

    if result.success and result.verdict == "continue":
        return None

    if result.error in _HITL_DENIED_ERRORS:
        dispose()
        return _provider_result(
            request,
            status=WorkerEphemeralCapabilityExecutionStatus.DENIED,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_DENIED,
            error_code=result.error,
            craft_correlation=craft_id,
            trace_correlation=trace_correlation,
        )

    if result.error in _UNAVAILABLE_ERRORS:
        dispose()
        return _provider_result(
            request,
            status=WorkerEphemeralCapabilityExecutionStatus.UNAVAILABLE,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_UNAVAILABLE,
            error_code=result.error,
            craft_correlation=craft_id,
            trace_correlation=trace_correlation,
        )

    if result.verdict == "revise" or (
        not result.static_gate.passed and result.verdict in {"revise", "continue"}
    ):
        return None

    dispose()
    status = WorkerEphemeralCapabilityExecutionStatus.FAILED
    reason = WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_FAILED
    error = result.error or (result.static_gate.message if not result.static_gate.passed else result.verdict)
    if error in {"codecraft_mode_disabled", "craft_ownership_mismatch", "craft_session_not_found"}:
        status = WorkerEphemeralCapabilityExecutionStatus.DENIED
        reason = WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_DENIED
    return _provider_result(
        request,
        status=status,
        reason_code=reason,
        error_code=error,
        craft_correlation=craft_id,
        trace_correlation=trace_correlation,
    )


def _map_craft_result(
    request: WorkerEphemeralCapabilityExecutionRequest,
    result: CraftResult,
    *,
    trace_correlation: str | None = None,
    dispose: Callable[[], None],
) -> WorkerEphemeralCapabilityExecutionResult:
    craft_id = result.craft_id
    error = result.error or (result.static_gate.message if not result.static_gate.passed else "")

    if error in _HITL_PENDING_ERRORS:
        return _provider_result(
            request,
            status=WorkerEphemeralCapabilityExecutionStatus.PENDING_HITL,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_PENDING_HITL,
            error_code=error,
            craft_correlation=craft_id,
            trace_correlation=trace_correlation,
        )

    dispose()
    if error in _HITL_DENIED_ERRORS:
        return _provider_result(
            request,
            status=WorkerEphemeralCapabilityExecutionStatus.DENIED,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_DENIED,
            error_code=error,
            craft_correlation=craft_id,
            trace_correlation=trace_correlation,
        )
    if error in _UNAVAILABLE_ERRORS:
        return _provider_result(
            request,
            status=WorkerEphemeralCapabilityExecutionStatus.UNAVAILABLE,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_UNAVAILABLE,
            error_code=error,
            craft_correlation=craft_id,
            trace_correlation=trace_correlation,
        )
    status = WorkerEphemeralCapabilityExecutionStatus.FAILED
    reason = WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_FAILED
    if error in {"codecraft_mode_disabled", "craft_ownership_mismatch", "craft_session_not_found"}:
        status = WorkerEphemeralCapabilityExecutionStatus.DENIED
        reason = WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_DENIED
    return _provider_result(
        request,
        status=status,
        reason_code=reason,
        error_code=error or result.verdict,
        craft_correlation=craft_id,
        trace_correlation=trace_correlation,
    )


def _resolve_ephemeral_reference(
    ctx: ToolWiringContext,
    craft_id: str,
) -> WorkerEphemeralCapabilityReference:
    store = get_ephemeral_registry_store(ctx)
    tools = store.for_craft(craft_id).list_tools()
    ephemeral_tool_id = tools[0] if tools else None
    return WorkerEphemeralCapabilityReference(
        craft_id=craft_id,
        ephemeral_tool_id=ephemeral_tool_id,
    )


def _provider_result(
    request: WorkerEphemeralCapabilityExecutionRequest,
    *,
    status: WorkerEphemeralCapabilityExecutionStatus,
    reason_code: WorkerEphemeralCapabilityExecutionReasonCode,
    error_code: str | None = None,
    ephemeral_capability: WorkerEphemeralCapabilityReference | None = None,
    craft_correlation: str | None = None,
    trace_correlation: str | None = None,
) -> WorkerEphemeralCapabilityExecutionResult:
    evidence_refs = request.evidence_refs + request.acquisition_decision.evidence_refs
    return WorkerEphemeralCapabilityExecutionResult(
        status=status,
        reason_code=reason_code,
        worker_instance_id=request.worker_instance_id,
        acquisition_decision_id=request.acquisition_decision.decision_id,
        need_id=request.need_id,
        evidence_refs=evidence_refs,
        executed_at=request.requested_at,
        execution_policy_version=EPHEMERAL_EXECUTION_POLICY_VERSION,
        ephemeral_capability=ephemeral_capability,
        craft_correlation=craft_correlation,
        trace_correlation=trace_correlation,
        error_code=error_code,
    )
