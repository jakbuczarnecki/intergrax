# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Optional, Protocol
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.policies.policy_enforcer import PolicyAbortError, PolicyEnforcer
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind, RuntimePolicies
from intergrax.runtime.nexus.tracing.steps.step_failed import RuntimeStepFailedDiagV1
from intergrax.runtime.nexus.tracing.steps.step_finished import RuntimeStepFinishedDiagV1
from intergrax.runtime.nexus.tracing.steps.step_started import RuntimeStepStartedDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel

class RuntimeStep(Protocol):
    async def run(self, state: RuntimeState) -> None:
        ...

    def execution_kind(self) -> Optional[ExecutionKind]:
        """
        Execution category used by runtime policies (timeouts/retry/fallback).

        Return None for steps that are pure in-process logic and should not be
        policy-enforced (no external I/O, no side effects, no cost).
        """
        ...
        

class RuntimeStepRunner:

    @classmethod
    async def execute_pipeline(cls, steps: list[RuntimeStep], state: RuntimeState) -> None:
        for step in steps:
            step_name = step.__class__.__name__

            state.trace_event(
                component=TraceComponent.RUNTIME,
                step=step_name,
                message="Step started",
                level=TraceLevel.INFO,
                payload=RuntimeStepStartedDiagV1(step_name=step_name),
            )

            try:
                await cls._execute_step(step, state)
            except Exception as e:
                state.trace_event(
                    component=TraceComponent.PIPELINE,
                    step=step_name,
                    message="Step failed",
                    level=TraceLevel.ERROR,
                    payload=RuntimeStepFailedDiagV1(
                        step_name=step_name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        error_repr=repr(e),
                    ),
                )
                raise

            state.trace_event(
                component=TraceComponent.RUNTIME,
                step=step_name,
                message="Step finished",
                level=TraceLevel.INFO,
                payload=RuntimeStepFinishedDiagV1(step_name=step_name),
            )
    

    @classmethod
    async def _execute_step(cls, step: RuntimeStep, state: RuntimeState) -> None:
        """
        Central execution point for all runtime steps.

        Runtime policies (timeouts/retry/fallback) are enforced here.
        """

        # Obtain runtime policies from config (single source of truth)
        policies = state.context.config.runtime_policies
        if policies is None:
            policies = RuntimePolicies()

        enforcer = PolicyEnforcer(policies)

        async def _run_step():
            return await step.run(state)

        try:
            await enforcer.execute(
                kind=step.execution_kind(),
                op_name=step.__class__.__name__,
                fn=_run_step,
                state=state,
            )
        except PolicyAbortError:
            # For now, propagate upward; mapping to stop_reason comes in next step
            raise
