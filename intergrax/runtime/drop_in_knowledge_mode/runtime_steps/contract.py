# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Protocol
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState

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
                data={}
            )

            try:
                await step.run(state)
            except Exception as e:
                state.trace_event(
                    component="pipeline",
                    step=step_name,
                    message="Step failed",
                    data={"error": repr(e)},
                )
                raise

            state.trace_event(
                component="runtime",
                step=step_name,
                message="Step finished",
                data={}
            )