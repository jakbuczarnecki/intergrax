# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations


from intergrax.runtime.drop_in_knowledge_mode.engine.pipelines.contract import RuntimePipeline
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeAnswer


# Runtime steps



class PlannerStaticPipeline(RuntimePipeline):
    """
    STATIC plan pipeline:
      - run base pipeline for planning (no RAG/web/tools/LLM)
      - EnginePlanner -> StepPlanner(STATIC) -> StepExecutor(registry)
      - runtime_answer MUST be produced by PersistAndBuildAnswerStep
    """

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        raise NotImplementedError(
            "StepPlanningStrategy.STATIC_LOOP is configured, but step planner is not implemented."
        ) 
