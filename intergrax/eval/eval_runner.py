# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from intergrax.eval.eval_case import EvalCase
from intergrax.eval.eval_result import EvalResult
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.replay.metrics import ExecutionMetricsEngine
from intergrax.runtime.replay.models import ReconstructedRun
from intergrax.runtime.replay.replay_engine import ReplayEngine


@runtime_checkable
class RuntimeRunnerProtocol(Protocol):
    async def run(self, request: RuntimeRequest) -> RuntimeAnswer: ...


class EvalRunner:

    def __init__(
        self,
        runtime_runner: RuntimeRunnerProtocol,
        replay_engine: ReplayEngine,
        metrics_engine: ExecutionMetricsEngine,
    ) -> None:
        self._runtime_runner = runtime_runner
        self._replay_engine = replay_engine
        self._metrics_engine = metrics_engine

    async def run_case(self, case: EvalCase) -> EvalResult:

        answer: RuntimeAnswer = await self._runtime_runner.run(
            case.runtime_request
        )

        if answer.run_id is None:
            return EvalResult(
                case_id=case.case_id,
                success=False,
                final_answer="",
                total_tokens=0,
                total_cost=0.0,
                tool_calls_count=0,
                error="Missing run_id in RuntimeAnswer",
            )

        reconstructed: ReconstructedRun = self._replay_engine.reconstruct(
            answer.run_id
        )

        metrics = self._metrics_engine.compute(reconstructed)

        final_answer: str = reconstructed.final_answer or ""
        total_tokens: int = metrics.total_tokens
        tool_calls_count: int = metrics.total_tool_calls

        total_cost: float = float(total_tokens)

        success: bool = final_answer == case.expected_output

        return EvalResult(
            case_id=case.case_id,
            success=success,
            final_answer=final_answer,
            total_tokens=total_tokens,
            total_cost=total_cost,
            tool_calls_count=tool_calls_count,
            error=None,
        )

    async def run_cases(self, cases: List[EvalCase]) -> List[EvalResult]:
        results: List[EvalResult] = []
        for case in cases:
            results.append(await self.run_case(case))
        return results
