# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.runtime_step_handlers import RuntimeStep


class BuildBaseHistoryStep(RuntimeStep):
    """
    Build base history for the session (project/user/system seed history),
    using HistoryLayer.
    """

    async def run(self, state: RuntimeState) -> None:
        await state.context.history_layer.build_base_history(state)
