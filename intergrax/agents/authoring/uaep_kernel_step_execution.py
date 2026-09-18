# © Artur Czarnecki. All rights reserved.

"""Internal UAEP ↔ HarnessKernel step outcome carrier (not a public platform contract)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_step import StepExecutionResult
from intergrax.contracts.step_execution import StepExecutionRecord


@dataclass(frozen=True, slots=True)
class UaepKernelStepExecution:
    """Typed bridge from HarnessKernel.execute_step to UAEPExecutor governance mapping."""

    step_result: StepExecutionResult
    kernel_record: StepExecutionRecord


@dataclass(frozen=True, slots=True)
class UaepExecutorStepOutcome:
    """UAEP executor loop holder: public step result plus optional kernel record."""

    step_result: StepExecutionResult
    kernel_record: StepExecutionRecord | None = None

    @classmethod
    def from_kernel_execution(cls, execution: UaepKernelStepExecution) -> UaepExecutorStepOutcome:
        return cls(step_result=execution.step_result, kernel_record=execution.kernel_record)
