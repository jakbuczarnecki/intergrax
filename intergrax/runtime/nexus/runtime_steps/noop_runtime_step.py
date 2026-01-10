# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep


class NoOpRuntimeStep(RuntimeStep):
    """
    No-op runtime step used for control-flow actions.

    This step intentionally performs no work and exists only to allow
    control-flow actions (e.g. ASK_CLARIFYING_QUESTION) to be executed
    by StepExecutor without raising "no handler" errors.

    Responsibilities:
    - act as a placeholder handler for non-executing, control-flow steps
    - allow StepExecutor to mark the step as OK and let PlanLoopController
      handle the corresponding stop reason (e.g. NEEDS_USER_INPUT)
    - avoid side effects on RuntimeState
    """

    async def run(self, state: RuntimeState) -> None:
        return