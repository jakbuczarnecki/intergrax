# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.replay.models import LLMCallInfo


def llm_call_info_from_adapter_response(
    response: LLMAdapterResponse,
    *,
    step_id: str,
    request_payload: object | None = None,
) -> LLMCallInfo:
    """Map a typed adapter response into replay ``LLMCallInfo`` fields."""
    usage = response.usage
    input_tokens = int(usage.input_tokens) if usage else 0
    output_tokens = int(usage.output_tokens) if usage else 0
    total_tokens = int(usage.total_tokens) if usage else input_tokens + output_tokens
    return LLMCallInfo(
        step_id=step_id,
        model=response.model or "",
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
        finish_reason=response.finish_reason.value,
        request_payload=request_payload,
        response_payload=response,
    )
