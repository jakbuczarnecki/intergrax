from __future__ import annotations

from intergrax.runtime.observability.modality_metrics import aggregate_modality_metrics_from_trace_events
from intergrax.runtime.nexus.tracing.persistence_models import SerializedTraceEvent
from intergrax.runtime.observability.modality_counters import (
    MODALITY_INVOCATION_COUNTERS_KEY,
    ModalityInvocationCounters,
)
from intergrax.runtime.observability.modality_tool_trace import (
    consume_modality_metrics_for_tool,
    modality_metrics_from_extras,
)
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from pydantic import BaseModel


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    ok: bool = True


def _noop(_ctx: ToolWiringContext, _payload: _In) -> _Out:
    return _Out(ok=True)


class _EchoHandler(ServiceToolHandler[_In, _Out]):
    _service = _noop


def test_consume_modality_metrics_clears_counters() -> None:
    ctx = ToolWiringContext(
        extras={
            MODALITY_INVOCATION_COUNTERS_KEY: ModalityInvocationCounters(
                inference_ms=50,
                vision_detections=2,
            )
        }
    )
    registry = ToolRegistry()
    contract = ToolContract(
        tool_id="vision.detect",
        name="vision.detect",
        description="detect",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
    )
    registry.register(contract, _EchoHandler(ctx))
    metrics = consume_modality_metrics_for_tool(registry, "vision.detect")
    assert metrics is not None
    assert metrics.inference_ms == 50
    assert metrics.vision_detections == 2
    assert MODALITY_INVOCATION_COUNTERS_KEY not in ctx.extras


def test_export_aggregates_tool_invocation_end_metrics() -> None:
    events = [
        SerializedTraceEvent(
            event_id="e1",
            run_id="run-1",
            seq=1,
            ts_utc="2020-01-01T00:00:00Z",
            level="info",
            component="tools",
            step="tool_invocation_end",
            message="done",
            payload_schema_id=None,
            payload_schema_version=None,
            payload={"modality_metrics": {"inference_ms": 10, "vision_detections": 1}},
            tags={},
            artifact_refs=[],
        ),
        SerializedTraceEvent(
            event_id="e2",
            run_id="run-1",
            seq=2,
            ts_utc="2020-01-01T00:00:01Z",
            level="info",
            component="tools",
            step="tool_invocation_end",
            message="done",
            payload_schema_id=None,
            payload_schema_version=None,
            payload={"modality_metrics": {"inference_ms": 20, "vision_detections": 2}},
            tags={},
            artifact_refs=[],
        ),
    ]
    aggregated = aggregate_modality_metrics_from_trace_events(events)
    assert aggregated.inference_ms == 30
    assert aggregated.vision_detections == 3


def test_modality_metrics_from_extras_empty_returns_none() -> None:
    assert modality_metrics_from_extras(ToolWiringContext()) is None
