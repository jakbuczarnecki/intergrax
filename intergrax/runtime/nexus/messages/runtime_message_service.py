# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import StopReason


@dataclass
class RuntimeMessageConfig:
    """
    Configuration for system-level runtime messages returned to the user.
    """

    default_hitl_message: str = "Additional information is required to continue."


class RuntimeMessageService:
    """
    Service responsible for building user-facing runtime messages,
    such as HITL escalation prompts, abort messages, etc.
    """

    def __init__(self, config: Optional[RuntimeMessageConfig] = None) -> None:
        self._config = config or RuntimeMessageConfig()

    def build_message(
        self,
        *,
        stop_reason: StopReason,
        state: RuntimeState,
        error: Optional[Exception] = None,
    ) -> str:
        if stop_reason == StopReason.NEEDS_USER_INPUT:
            return self._build_hitl_message(state=state, error=error)

        return ""

    def _build_hitl_message(
        self,
        *,
        state: RuntimeState,
        error: Optional[Exception],
    ) -> str:
        # Future extension: inspect state, last step, error, etc.
        return self._config.default_hitl_message
