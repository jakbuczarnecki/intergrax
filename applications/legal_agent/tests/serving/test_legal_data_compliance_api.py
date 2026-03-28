# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from legal_agent.serving.runtime_bridge import LegalApiV1RuntimeMapper
from intergrax.fastapi_core.context import RequestContext
from intergrax.runtime.nexus.policies.runtime_policies import DataCompliancePolicy
from intergrax.runtime.nexus.responses.response_schema import (
    RouteInfo,
    RuntimeAnswer,
    RuntimeStats,
    StopReason,
    ToolCallInfo,
)
from intergrax.runtime.nexus.tracing.tools.tool_invocation import ToolInvocationStartDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel

import pytest

pytestmark = pytest.mark.unit


def _ctx() -> RequestContext:
    return RequestContext(
        request_id="req-c",
        path="/v1/legal/chat",
        method="POST",
        tenant_id="t",
        user_id="u",
        auth=None,
    )


def _trace_with_secret_payload() -> TraceEvent:
    return TraceEvent(
        event_id="e1",
        run_id="r1",
        seq=0,
        ts_utc="2020-01-01T00:00:00+00:00",
        level=TraceLevel.INFO,
        component=TraceComponent.TOOLS,
        step="tools",
        message="start",
        payload=ToolInvocationStartDiagV1(
            tool_id="t1",
            step_id="s1",
            side_effects=False,
            input_payload={"secret": "client-data"},
        ),
    )


def test_api_trace_export_none_never_serializes_trace() -> None:
    answer = RuntimeAnswer(
        answer="x",
        stop_reason=StopReason.COMPLETED,
        run_id="r1",
        route=RouteInfo(strategy="legal_agent"),
        stats=RuntimeStats(),
        trace_events=[_trace_with_secret_payload()],
    )
    out = LegalApiV1RuntimeMapper().to_legal_chat_response(
        answer,
        http_context=_ctx(),
        include_trace=True,
        data_compliance=DataCompliancePolicy(api_trace_export="none"),
    )
    assert out.trace_events is None


def test_api_trace_export_redacted_strips_payload_secrets() -> None:
    answer = RuntimeAnswer(
        answer="x",
        stop_reason=StopReason.COMPLETED,
        run_id="r1",
        route=RouteInfo(strategy="legal_agent"),
        stats=RuntimeStats(),
        trace_events=[_trace_with_secret_payload()],
    )
    out = LegalApiV1RuntimeMapper().to_legal_chat_response(
        answer,
        http_context=_ctx(),
        include_trace=True,
        data_compliance=DataCompliancePolicy(api_trace_export="redacted"),
    )
    assert out.trace_events is not None
    payload = out.trace_events[0].get("payload") or {}
    assert payload.get("input_payload") == {"_redacted": True}


def test_api_trace_export_full_keeps_raw_payload() -> None:
    answer = RuntimeAnswer(
        answer="x",
        stop_reason=StopReason.COMPLETED,
        run_id="r1",
        route=RouteInfo(strategy="legal_agent"),
        stats=RuntimeStats(),
        trace_events=[_trace_with_secret_payload()],
    )
    out = LegalApiV1RuntimeMapper().to_legal_chat_response(
        answer,
        http_context=_ctx(),
        include_trace=True,
        data_compliance=DataCompliancePolicy(api_trace_export="full"),
    )
    assert out.trace_events is not None
    payload = out.trace_events[0].get("payload") or {}
    assert payload.get("input_payload") == {"secret": "client-data"}


def test_redact_tool_calls_in_api() -> None:
    answer = RuntimeAnswer(
        answer="x",
        stop_reason=StopReason.COMPLETED,
        run_id="r1",
        route=RouteInfo(strategy="legal_agent"),
        stats=RuntimeStats(),
        tool_calls=[
            ToolCallInfo(tool_name="lookup", arguments={"pin": "1234"}, success=True),
        ],
    )
    out = LegalApiV1RuntimeMapper().to_legal_chat_response(
        answer,
        http_context=_ctx(),
        include_trace=False,
        data_compliance=DataCompliancePolicy(redact_tool_calls_in_api=True),
    )
    assert out.tool_calls[0]["arguments"] == {"_redacted": True}

    out2 = LegalApiV1RuntimeMapper().to_legal_chat_response(
        answer,
        http_context=_ctx(),
        include_trace=False,
        data_compliance=DataCompliancePolicy(redact_tool_calls_in_api=False),
    )
    assert out2.tool_calls[0]["arguments"] == {"pin": "1234"}
