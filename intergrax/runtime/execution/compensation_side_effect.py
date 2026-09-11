# © Artur Czarnecki. All rights reserved.

"""ExecutionRuntime-admitted compensation tool side effects (U2 / EP-16)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.compensation_side_effect_execution import (
    CompensationSideEffectExecutionPort,
    CompensationSideEffectInput,
    CompensationSideEffectInvokeResult,
    CompensationToolInvokeSession,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    TaskId,
    require_active_execution_identity,
    validate_run_id,
)
from intergrax.runtime.execution.decision_lifecycle_host import (
    CanonicalDecisionLifecycleHost,
)
from intergrax.contracts.execution_lineage import ExecutionLineagePersistence
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionOptions,
)
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory
from intergrax.runtime.governance.active_execution_authority import (
    peek_active_execution_authority,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget


@dataclass(frozen=True, slots=True)
class _CompensationSideEffectDelegate:
    _tool_session: CompensationToolInvokeSession

    async def execute(self, work: CompensationSideEffectInput) -> CompensationSideEffectInvokeResult:
        active_authority = peek_active_execution_authority()
        if active_authority is not None and active_authority.is_unknown:
            return CompensationSideEffectInvokeResult(
                status="denied",
                error="execution_authority_unknown",
            )
        active_run_id, _active_attempt_id = require_active_execution_identity()
        if str(active_run_id) != work.run_id:
            return CompensationSideEffectInvokeResult(
                status="failed",
                error="compensation_execution_identity_run_mismatch",
            )
        return await self._tool_session.invoke(
            tenant_id=work.tenant_id,
            run_id=work.run_id,
            task_id=work.task_id,
            agent_id=work.agent_id,
            tool_id=work.compensation_tool_id,
            args=work.args,
            idempotency_key=work.idempotency_key,
        )


@dataclass(frozen=True, slots=True)
class RuntimeCompensationSideEffectExecution(CompensationSideEffectExecutionPort):
    """Admit compensation tool work through canonical root ExecutionRuntime."""

    _execution: Execution[CompensationSideEffectInput, CompensationSideEffectInvokeResult]
    _authority: ParentExecutionAuthority

    async def execute(self, work: CompensationSideEffectInput) -> CompensationSideEffectInvokeResult:
        options = RootExecutionOptions(
            authority=self._authority,
            tenant_id=work.tenant_id,
            run_id=validate_run_id(work.run_id),
            task_id=TaskId(work.task_id),
        )
        return await self._execution.execute(work, options=options)


def build_runtime_compensation_side_effect_execution(
    *,
    tool_session: CompensationToolInvokeSession,
    authority: ParentExecutionAuthority,
    ledger_factory: ExecutionBudgetLedgerFactory | None = None,
    run_budget: RunBudget | None = None,
    execution_lineage_persistence: ExecutionLineagePersistence | None = None,
) -> RuntimeCompensationSideEffectExecution:
    runtime = ExecutionRuntime[
        CompensationSideEffectInput,
        CompensationSideEffectInvokeResult,
    ](
        _CompensationSideEffectDelegate(tool_session),
        ledger_factory=ledger_factory,
        run_budget=run_budget,
        decision_lifecycle_host=CanonicalDecisionLifecycleHost(),
        execution_lineage_persistence=execution_lineage_persistence,
    )
    return RuntimeCompensationSideEffectExecution(
        _execution=Execution(runtime),
        _authority=authority,
    )
