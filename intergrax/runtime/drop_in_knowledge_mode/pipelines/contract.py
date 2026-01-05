# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeAnswer

class RuntimePipeline(ABC):
    """
    Lightweight base class for pipeline runners.
    Shared implementation: execute_pipeline.
    """
    
    async def run(self, state: RuntimeState) -> RuntimeAnswer:
        """
        Public entrypoint. Provides shared validation and invariants.
        Subclasses must implement _inner_run().
        """
        self._validate_state(state)

        runtime_answer = await self._inner_run(state)

        if runtime_answer is None:
            raise RuntimeError("Pipeline returned None RuntimeAnswer.")

        # Hard invariant: Persist step (or equivalent) must set state.runtime_answer
        if state.runtime_answer is None:
            raise RuntimeError("Persist step did not set state.runtime_answer.")

        # Prefer the actual object returned by inner_run, but enforce consistency with state
        # (Optional) If you want to hard-enforce object identity:
        # if runtime_answer is not state.runtime_answer: ...
        self._assert_valid_answer(runtime_answer)

        return runtime_answer

    
    @abstractmethod
    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        """
        Pipeline-specific execution. Must produce and return RuntimeAnswer.
        """
        raise NotImplementedError


    def _validate_state(self, state: RuntimeState) -> None:
        if state is None:
            raise ValueError("state is None.")
        if state.context is None:
            raise ValueError("state.context is None.")
        if state.request is None:
            raise ValueError("state.request is None.")
        if not state.run_id:
            raise ValueError("state.run_id is missing.")
        

    async def _execute_pipeline(self, steps: list[RuntimeStep], state: RuntimeState) -> None:
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

    def _assert_valid_answer(self, answer: RuntimeAnswer) -> None:
        assert answer is not None, "Pipeline returned None answer"
        assert answer.route is not None, "RuntimeAnswer has no route"
