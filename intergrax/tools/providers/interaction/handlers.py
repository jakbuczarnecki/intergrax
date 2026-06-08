# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.interaction.contracts import (
    InteractionGetLastInputInput,
    InteractionGetLastInputOutput,
    InteractionGetSessionHistoryInput,
    InteractionGetSessionHistoryOutput,
    InteractionListSessionsInput,
    InteractionListSessionsOutput,
    InteractionPostReplyInput,
    InteractionPostReplyOutput,
)
from intergrax.tools.providers.interaction.service import (
    interaction_get_last_input,
    interaction_get_session_history,
    interaction_list_sessions,
    interaction_post_reply,
)


class InteractionListSessionsHandler(
    ServiceToolHandler[InteractionListSessionsInput, InteractionListSessionsOutput]
):
    _service = interaction_list_sessions


class InteractionGetLastInputHandler(
    ServiceToolHandler[InteractionGetLastInputInput, InteractionGetLastInputOutput]
):
    _service = interaction_get_last_input


class InteractionGetSessionHistoryHandler(
    ServiceToolHandler[InteractionGetSessionHistoryInput, InteractionGetSessionHistoryOutput]
):
    _service = interaction_get_session_history


class InteractionPostReplyHandler(ServiceToolHandler[InteractionPostReplyInput, InteractionPostReplyOutput]):
    _service = interaction_post_reply
