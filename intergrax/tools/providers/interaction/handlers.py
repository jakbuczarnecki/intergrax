# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.interaction.contracts import (
    InteractionGetLastInputInput,
    InteractionGetLastInputOutput,
    InteractionListSessionsInput,
    InteractionListSessionsOutput,
)
from intergrax.tools.providers.interaction.service import interaction_get_last_input, interaction_list_sessions


class InteractionListSessionsHandler(
    ServiceToolHandler[InteractionListSessionsInput, InteractionListSessionsOutput]
):
    _service = interaction_list_sessions


class InteractionGetLastInputHandler(
    ServiceToolHandler[InteractionGetLastInputInput, InteractionGetLastInputOutput]
):
    _service = interaction_get_last_input
