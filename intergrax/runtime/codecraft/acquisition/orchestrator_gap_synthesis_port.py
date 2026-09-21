# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production CodeCraftGapSynthesisPort over CodeCraftOrchestrator (UCA-6A-R)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.codecraft.contracts import CraftResult
from intergrax.contracts.codecraft.gap_synthesis import (
    CodeCraftGapSynthesisOutcome,
    CodeCraftGapSynthesisRequest,
    CodeCraftGapSynthesisResult,
)
from intergrax.runtime.codecraft.acquisition.gap_synthesis_craft_mapping import (
    artifact_reference_for_craft,
    gap_synthesis_outcome_for_craft_error,
    gap_synthesis_outcome_for_iterate_result,
)
from intergrax.runtime.codecraft.ephemeral_registry import get_ephemeral_registry_store
from intergrax.runtime.codecraft.orchestrator import (
    CodeCraftOrchestrator,
    resolve_codecraft_profile,
)
from intergrax.runtime.codecraft.ownership import (
    CodeCraftOwnershipError,
    resolve_codecraft_ownership,
)
from intergrax.tools.registry.wiring import ToolWiringContext

_OrchestratorFactory = Callable[[ToolWiringContext, str], CodeCraftOrchestrator]


def _default_orchestrator_factory(
    ctx: ToolWiringContext, run_id: str
) -> CodeCraftOrchestrator:
    return CodeCraftOrchestrator(ctx, run_id=run_id)


class CodeCraftOrchestratorGapSynthesisPort:
    """CodeCraft-owned gap synthesis — drives public orchestrator APIs only."""

    def __init__(
        self,
        wiring_context: ToolWiringContext,
        *,
        orchestrator_factory: _OrchestratorFactory | None = None,
    ) -> None:
        self._ctx = wiring_context
        self._orchestrator_factory = (
            orchestrator_factory or _default_orchestrator_factory
        )

    def synthesize_from_gap(
        self,
        request: CodeCraftGapSynthesisRequest,
    ) -> CodeCraftGapSynthesisResult:
        base = _result_shell(request)
        try:
            ownership = resolve_codecraft_ownership(self._ctx)
        except CodeCraftOwnershipError as exc:
            return base.model_copy(
                update={
                    "outcome": gap_synthesis_outcome_for_craft_error(exc.code),
                    "reason_detail": exc.code,
                },
            )

        tenant_id = ownership.tenant_id
        task_id = ownership.task_id
        craft_id = request.operation_id

        profile = resolve_codecraft_profile(self._ctx)
        if profile is None:
            return base.model_copy(
                update={
                    "outcome": CodeCraftGapSynthesisOutcome.UNAVAILABLE,
                    "reason_detail": "codecraft_profile_missing",
                },
            )

        orch = self._orchestrator_factory(self._ctx, request.operation_id)
        session, deny = orch.start(
            goal=request.synthesis_goal,
            task_id=task_id,
            tenant_id=tenant_id,
            constraints=request.synthesis_constraints,
            craft_id=craft_id,
        )
        if deny is not None:
            return _from_craft_result(base, deny, craft_id=craft_id)
        if session is None:
            return base.model_copy(
                update={
                    "outcome": CodeCraftGapSynthesisOutcome.UNAVAILABLE,
                    "reason_detail": "codecraft_start_failed",
                },
            )

        def dispose() -> None:
            orch.dispose(craft_id, tenant_id=tenant_id, task_id=task_id)

        last_result: CraftResult | None = None
        for _ in range(profile.max_iterations):
            _session, last_result = orch.iterate(
                craft_id=craft_id,
                task_id=task_id,
                tenant_id=tenant_id,
            )
            terminal = _terminal_from_iterate(
                base,
                ctx=self._ctx,
                result=last_result,
                craft_id=craft_id,
                orch=orch,
                tenant_id=tenant_id,
                task_id=task_id,
                dispose=dispose,
            )
            if terminal is not None:
                return terminal

        dispose()
        error = (
            last_result.error if last_result is not None else "max_iterations_exceeded"
        )
        return base.model_copy(
            update={
                "outcome": CodeCraftGapSynthesisOutcome.FAILED,
                "reason_detail": error or "max_iterations_exceeded",
                "codecraft_operation_correlation_id": craft_id,
            },
        )


def _result_shell(request: CodeCraftGapSynthesisRequest) -> CodeCraftGapSynthesisResult:
    return CodeCraftGapSynthesisResult(
        operation_id=request.operation_id,
        gap_id=request.gap_id,
        outcome=CodeCraftGapSynthesisOutcome.FAILED,
        correlation_id=request.correlation_id,
        causation_id=request.causation_id,
    )


def _from_craft_result(
    base: CodeCraftGapSynthesisResult,
    result: CraftResult,
    *,
    craft_id: str,
) -> CodeCraftGapSynthesisResult:
    error = result.error or (
        result.static_gate.message if not result.static_gate.passed else ""
    )
    outcome = (
        gap_synthesis_outcome_for_craft_error(error)
        if error
        else CodeCraftGapSynthesisOutcome.FAILED
    )
    detail = error or result.verdict
    return base.model_copy(
        update={
            "outcome": outcome,
            "reason_detail": detail,
            "codecraft_operation_correlation_id": craft_id,
        },
    )


def _terminal_from_iterate(
    base: CodeCraftGapSynthesisResult,
    *,
    ctx: ToolWiringContext,
    result: CraftResult,
    craft_id: str,
    orch: CodeCraftOrchestrator,
    tenant_id: str,
    task_id: str,
    dispose: Callable[[], None],
) -> CodeCraftGapSynthesisResult | None:
    outcome = gap_synthesis_outcome_for_iterate_result(result)
    if outcome is CodeCraftGapSynthesisOutcome.REQUIRES_HITL:
        return base.model_copy(
            update={
                "outcome": outcome,
                "reason_detail": result.error or "hitl_pending",
                "codecraft_operation_correlation_id": craft_id,
            },
        )

    if outcome is CodeCraftGapSynthesisOutcome.SUCCEEDED:
        promote_result = orch.promote(craft_id, tenant_id=tenant_id, task_id=task_id)
        if not promote_result.success:
            dispose()
            return base.model_copy(
                update={
                    "outcome": CodeCraftGapSynthesisOutcome.FAILED,
                    "reason_detail": promote_result.error or "promotion_failed",
                    "codecraft_operation_correlation_id": craft_id,
                },
            )
        if not _ephemeral_artifact_exists(ctx, craft_id):
            dispose()
            return base.model_copy(
                update={
                    "outcome": CodeCraftGapSynthesisOutcome.FAILED,
                    "reason_detail": "synthesis_artifact_missing",
                    "codecraft_operation_correlation_id": craft_id,
                },
            )
        return base.model_copy(
            update={
                "outcome": CodeCraftGapSynthesisOutcome.SUCCEEDED,
                "artifact_reference": artifact_reference_for_craft(craft_id),
                "codecraft_operation_correlation_id": craft_id,
            },
        )

    if result.verdict == "revise" or (result.success and result.verdict == "continue"):
        return None

    if outcome is CodeCraftGapSynthesisOutcome.BLOCKED:
        dispose()
        return base.model_copy(
            update={
                "outcome": outcome,
                "reason_detail": result.error or result.verdict,
                "codecraft_operation_correlation_id": craft_id,
            },
        )

    if outcome is CodeCraftGapSynthesisOutcome.UNAVAILABLE:
        dispose()
        return base.model_copy(
            update={
                "outcome": outcome,
                "reason_detail": result.error or "unavailable",
                "codecraft_operation_correlation_id": craft_id,
            },
        )

    dispose()
    error = result.error or (
        result.static_gate.message if not result.static_gate.passed else result.verdict
    )
    return base.model_copy(
        update={
            "outcome": CodeCraftGapSynthesisOutcome.FAILED,
            "reason_detail": error or "synthesis_failed",
            "codecraft_operation_correlation_id": craft_id,
        },
    )


def _ephemeral_artifact_exists(ctx: ToolWiringContext, craft_id: str) -> bool:
    tools = get_ephemeral_registry_store(ctx).for_craft(craft_id).list_tools()
    return bool(tools)


__all__ = ["CodeCraftOrchestratorGapSynthesisPort"]
