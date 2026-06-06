# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, List

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.nexus.tracing.adapters.core_llm_call_recorded import CoreLLMCallRecordedDiagV1
from intergrax.runtime.nexus.tracing.persistence_models import SerializedTraceEvent
from intergrax.runtime.replay.contracts.trace_event_dto import TraceEventDTO
from intergrax.runtime.replay.llm_call_mapper import llm_call_info_from_adapter_response

_CORE_LLM_CALL_SCHEMA = CoreLLMCallRecordedDiagV1.schema_id()
_CORE_LLM_RETURNED_SCHEMA = "intergrax.diag.engine.core_llm.adapter_returned"


def _parse_ts_utc(ts_utc: str) -> float:
    try:
        normalized = ts_utc.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return datetime.now(timezone.utc).timestamp()


def _payload_dict(event: SerializedTraceEvent) -> dict[str, Any]:
    return dict(event.payload or {})


def _llm_call_dto_from_recorded(
    event: SerializedTraceEvent,
    payload: dict[str, Any],
) -> TraceEventDTO:
    return TraceEventDTO(
        run_id=event.run_id,
        step_id=event.step,
        event_type="LLM_CALL",
        timestamp=_parse_ts_utc(event.ts_utc),
        payload={
            "model": payload.get("model", ""),
            "prompt_tokens": int(payload.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(payload.get("completion_tokens", 0) or 0),
            "total_tokens": int(payload.get("total_tokens", 0) or 0),
            "finish_reason": payload.get("finish_reason"),
        },
    )


def _llm_call_dto_from_returned(
    event: SerializedTraceEvent,
    payload: dict[str, Any],
) -> TraceEventDTO:
    input_tokens = int(payload.get("input_tokens", 0) or 0)
    output_tokens = int(payload.get("output_tokens", 0) or 0)
    return TraceEventDTO(
        run_id=event.run_id,
        step_id=event.step,
        event_type="LLM_CALL",
        timestamp=_parse_ts_utc(event.ts_utc),
        payload={
            "model": "",
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "finish_reason": payload.get("finish_reason"),
        },
    )


def serialized_trace_events_to_replay_dtos(
    events: Iterable[SerializedTraceEvent],
) -> List[TraceEventDTO]:
    """Map persisted Nexus trace events to replay ``TraceEventDTO`` rows."""
    out: List[TraceEventDTO] = []
    for event in events:
        schema_id = event.payload_schema_id or ""
        payload = _payload_dict(event)
        if schema_id == _CORE_LLM_CALL_SCHEMA:
            out.append(_llm_call_dto_from_recorded(event, payload))
            continue
        if schema_id == _CORE_LLM_RETURNED_SCHEMA:
            out.append(_llm_call_dto_from_returned(event, payload))
            continue
        if event.step == "core_llm" and event.component == "engine":
            if "finish_reason" in payload and "input_tokens" in payload:
                out.append(_llm_call_dto_from_returned(event, payload))
    return out


def llm_call_dto_from_adapter_response(
    response: LLMAdapterResponse,
    *,
    run_id: str,
    step_id: str,
    timestamp: float,
) -> TraceEventDTO:
    """Build a replay ``LLM_CALL`` DTO directly from a typed adapter response."""
    info = llm_call_info_from_adapter_response(response, step_id=step_id)
    return TraceEventDTO(
        run_id=run_id,
        step_id=step_id,
        event_type="LLM_CALL",
        timestamp=timestamp,
        payload={
            "model": info.model,
            "prompt_tokens": info.prompt_tokens,
            "completion_tokens": info.completion_tokens,
            "total_tokens": info.total_tokens,
            "finish_reason": info.finish_reason,
        },
    )
