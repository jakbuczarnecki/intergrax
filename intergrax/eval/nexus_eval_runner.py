# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Optional

from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.eval.eval_case import EvalCase
from intergrax.eval.eval_result import EvalResult
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.replay.metrics import ExecutionMetricsEngine
from intergrax.runtime.replay.models import ReconstructedRun
from intergrax.runtime.replay.replay_engine import ReplayEngine
from intergrax.runtime.task.task import TaskState
from intergrax.runtime.task.task_run_bridge import task_from_runtime_request
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


class NexusEvalRunner:
    """
    Evaluation runner via NexusLoop + AgentExecutionResult (Phase A.4).

    Complements legacy EvalRunner (RuntimeEngine-only path).
    """

    def __init__(
        self,
        task_runner: UnifiedTaskRunner,
        replay_engine: Optional[ReplayEngine] = None,
        metrics_engine: Optional[ExecutionMetricsEngine] = None,
    ) -> None:
        self._task_runner = task_runner
        self._replay_engine = replay_engine
        self._metrics_engine = metrics_engine

    @classmethod
    def from_nexus_loop(
        cls,
        nexus_loop: NexusLoop,
        *,
        replay_engine: Optional[ReplayEngine] = None,
        metrics_engine: Optional[ExecutionMetricsEngine] = None,
    ) -> "NexusEvalRunner":
        return cls(
            UnifiedTaskRunner(nexus_loop),
            replay_engine=replay_engine,
            metrics_engine=metrics_engine,
        )

    async def run_case(self, case: EvalCase) -> EvalResult:
        req = case.runtime_request
        tenant_id = req.tenant_id or "eval-tenant"
        user_id = req.user_id or "eval-user"
        capability = req.metadata.get("capability") if req.metadata else None

        try:
            result = await self._task_runner.run_runtime_request(
                req,
                tenant_id=tenant_id,
                user_id=user_id,
                capability=str(capability) if capability else None,
            )
        except Exception as exc:
            return EvalResult(
                case_id=case.case_id,
                success=False,
                final_answer="",
                total_tokens=0,
                total_cost=0.0,
                tool_calls_count=0,
                error=f"{type(exc).__name__}: {exc}",
            )

        execution = result.execution_result
        final_answer = result.answer or ""
        agent_id = result.agent_id or req.agent_id
        capability_id = (
            execution.structured_data.get("capability")
            if execution and execution.structured_data
            else None
        ) or (req.metadata or {}).get("capability")

        total_tokens = 0
        tool_calls_count = 0
        total_cost = float(execution.cost or 0.0) if execution else 0.0

        if (
            result.run_id
            and self._replay_engine is not None
            and self._metrics_engine is not None
        ):
            try:
                reconstructed: ReconstructedRun = self._replay_engine.reconstruct(
                    result.run_id
                )
                metrics = self._metrics_engine.compute(reconstructed)
                total_tokens = metrics.total_tokens
                tool_calls_count = metrics.total_tool_calls
                if not final_answer and reconstructed.final_answer:
                    final_answer = reconstructed.final_answer
            except Exception:
                pass

        if execution and execution.used_tools:
            tool_calls_count = max(tool_calls_count, len(execution.used_tools))

        success = (
            result.state == TaskState.COMPLETED
            and execution is not None
            and execution.status == AgentExecutionStatus.COMPLETED
            and final_answer == case.expected_output
        )

        error = None
        if not success:
            if execution and execution.errors:
                error = "; ".join(execution.errors)
            elif result.state != TaskState.COMPLETED:
                error = f"task_state={result.state.value}"
            else:
                error = "output_mismatch"

        _ = agent_id, capability_id  # reserved for future eval report fields

        return EvalResult(
            case_id=case.case_id,
            success=success,
            final_answer=final_answer,
            total_tokens=total_tokens,
            total_cost=total_cost,
            tool_calls_count=tool_calls_count,
            error=error,
        )

    async def run_cases(self, cases: List[EvalCase]) -> List[EvalResult]:
        results: List[EvalResult] = []
        for case in cases:
            results.append(await self.run_case(case))
        return results
