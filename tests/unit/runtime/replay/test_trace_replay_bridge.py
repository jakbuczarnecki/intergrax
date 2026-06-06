# © Artur Czarnecki. All rights reserved.



from __future__ import annotations



import pytest



from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response

from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason

from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage

from intergrax.runtime.nexus.tracing.adapters.core_llm_call_recorded import CoreLLMCallRecordedDiagV1

from intergrax.runtime.nexus.tracing.persistence_models import SerializedTraceEvent

from intergrax.runtime.replay.trace_replay_bridge import (

    llm_call_dto_from_adapter_response,

    serialized_trace_events_to_replay_dtos,

)





@pytest.mark.gate

def test_serialized_trace_events_to_replay_dtos_from_call_recorded() -> None:

    event = SerializedTraceEvent(

        event_id="e1",

        run_id="run-1",

        seq=1,

        ts_utc="2026-06-06T12:00:00+00:00",

        level="info",

        component="engine",

        step="core_llm",

        message="recorded",

        payload_schema_id=CoreLLMCallRecordedDiagV1.schema_id(),

        payload_schema_version=1,

        payload={

            "model": "gpt-test",

            "provider": "openai",

            "prompt_tokens": 10,

            "completion_tokens": 5,

            "total_tokens": 15,

            "finish_reason": "stop",

            "response_id": "rid-1",

            "has_refusal": False,

            "has_tool_calls": False,

        },

        tags={},

        artifact_refs=[],

    )

    dtos = serialized_trace_events_to_replay_dtos([event])

    assert len(dtos) == 1

    dto = dtos[0]

    assert dto.event_type == "LLM_CALL"

    assert dto.run_id == "run-1"

    assert dto.step_id == "core_llm"

    assert dto.payload["model"] == "gpt-test"

    assert dto.payload["prompt_tokens"] == 10

    assert dto.payload["completion_tokens"] == 5

    assert dto.payload["finish_reason"] == "stop"





@pytest.mark.gate

def test_llm_call_dto_from_adapter_response() -> None:

    response = build_adapter_response(

        content="ok",

        finish_reason=LLMFinishReason.COMPLETED,

        usage=LLMTokenUsage.from_counts(input_tokens=4, output_tokens=6),

        model="claude-test",

        provider="claude",

    )

    dto = llm_call_dto_from_adapter_response(

        response,

        run_id="run-2",

        step_id="core_llm",

        timestamp=1_700_000_000.0,

    )

    assert dto.event_type == "LLM_CALL"

    assert dto.payload["model"] == "claude-test"

    assert dto.payload["total_tokens"] == 10

