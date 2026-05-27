# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from legal.config.legal_agent_config import LegalAgentConfig
from legal.uaep.dynamic_steps import run_dynamic_pipeline_on_state
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer


class LegalDynamicPipeline(RuntimePipeline):

    def __init__(self, *, config: LegalAgentConfig) -> None:
        super().__init__()
        self._config = config

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        return await run_dynamic_pipeline_on_state(state, config=self._config)
