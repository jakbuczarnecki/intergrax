# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Protocol
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.tracing.steps.step_failed import RuntimeStepFailedDiagV1
from intergrax.runtime.drop_in_knowledge_mode.tracing.steps.step_finished import RuntimeStepFinishedDiagV1
from intergrax.runtime.drop_in_knowledge_mode.tracing.steps.step_started import RuntimeStepStartedDiagV1
from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import TraceLevel

class RuntimeStep(Protocol):
    async def run(self, state: RuntimeState) -> None:        
        ...

class RuntimeStepRunner:

    @classmethod
    async def execute_pipeline(cls, steps: list[RuntimeStep], state: RuntimeState) -> None:
        for step in steps:
            step_name = step.__class__.__name__

            state.trace_event(
                component="runtime",
                step=step_name,
                message="Step started",
                level=TraceLevel.INFO,
                payload=RuntimeStepStartedDiagV1(step_name=step_name),
            )

            try:
                await step.run(state)
            except Exception as e:
                state.trace_event(
                    component="pipeline",
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
                component="runtime",
                step=step_name,
                message="Step finished",
                level=TraceLevel.INFO,
                payload=RuntimeStepFinishedDiagV1(step_name=step_name),
            )