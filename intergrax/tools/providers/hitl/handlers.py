# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.hitl.contracts import (
    HitlGetDecisionInput,
    HitlGetDecisionOutput,
    HitlListPendingInput,
    HitlListPendingOutput,
    HitlSummarizeQueueInput,
    HitlSummarizeQueueOutput,
)
from intergrax.tools.providers.hitl.service import hitl_get_decision, hitl_list_pending, hitl_summarize_queue


class HitlListPendingHandler(ServiceToolHandler[HitlListPendingInput, HitlListPendingOutput]):
    _service = hitl_list_pending


class HitlGetDecisionHandler(ServiceToolHandler[HitlGetDecisionInput, HitlGetDecisionOutput]):
    _service = hitl_get_decision


class HitlSummarizeQueueHandler(ServiceToolHandler[HitlSummarizeQueueInput, HitlSummarizeQueueOutput]):
    _service = hitl_summarize_queue
