# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.pipelines.contract import RuntimePipeline
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeAnswer


class PlannerDynamicPipeline(RuntimePipeline):

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        raise NotImplementedError(
            "StepPlanningStrategy.DYNAMIC_LOOP is configured, but step planner is not implemented."
        ) 
    