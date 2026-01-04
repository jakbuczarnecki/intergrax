# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.contract import RuntimeStep
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeAnswer

class RuntimePipeline(ABC):
    """
    Lightweight base class for pipeline runners.
    Shared implementation: execute_pipeline.
    """
    
    @abstractmethod
    async def run(self, state: RuntimeState) -> RuntimeAnswer:        
        ...


    async def execute_pipeline(self, steps: list[RuntimeStep], state: RuntimeState) -> None:
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
