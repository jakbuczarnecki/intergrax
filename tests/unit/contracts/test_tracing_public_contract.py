# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import dataclasses
import inspect
import typing
from dataclasses import dataclass

import pytest

from intergrax.contracts.tracing import (
    DiagnosticPayload,
    ToolCallTrace,
    TraceArtifactRef,
    TraceComponent,
    TraceEvent,
    TraceLevel,
    TraceObject,
    TraceValue,
    validate_trace_value,
)
from intergrax.contracts.tracing import events as tracing_events_module
from intergrax.runtime.nexus.tracing.trace_models import (
    ArtifactRef,
    DiagnosticPayload as RuntimeDiagnosticPayload,
    ToolCallTrace as RuntimeToolCallTrace,
    TraceEvent as RuntimeTraceEvent,
)

pytestmark = pytest.mark.unit


def _public_field_annotations(module: object, type_names: tuple[str, ...]) -> dict[str, object]:
    hints: dict[str, object] = {}
    for name in type_names:
        cls = getattr(module, name)
        for field in dataclasses.fields(cls):
            hints[f"{name}.{field.name}"] = field.type
    return hints


def test_trace_event_is_single_canonical_class() -> None:
    assert TraceEvent is RuntimeTraceEvent


def test_nexus_artifact_ref_aliases_contract() -> None:
    assert ArtifactRef is TraceArtifactRef


def test_diagnostic_payload_reexport_matches_contract() -> None:
    assert DiagnosticPayload is RuntimeDiagnosticPayload


def test_tool_call_trace_reexport_is_canonical() -> None:
    assert ToolCallTrace is RuntimeToolCallTrace


def test_public_trace_models_do_not_use_any_in_fields() -> None:
    field_types = _public_field_annotations(
        tracing_events_module,
        ("TraceEvent", "ToolCallTrace", "TraceArtifactRef"),
    )
    for label, annotation in field_types.items():
        assert typing.Any not in typing.get_args(annotation) and annotation is not typing.Any, label


def test_diagnostic_payload_to_dict_is_typed_trace_object() -> None:
    hints = typing.get_type_hints(DiagnosticPayload.to_dict)
    assert hints["return"] == TraceObject
    assert typing.Any not in typing.get_args(hints["return"])


def test_trace_value_accepts_nested_json_safe_data() -> None:
    value: TraceValue = validate_trace_value(
        {"a": [1, {"b": True}, None], "c": "ok"},
        field_name="sample",
    )
    assert value == {"a": [1, {"b": True}, None], "c": "ok"}


def test_trace_value_rejects_arbitrary_objects() -> None:
    class _Payload:
        pass

    with pytest.raises(ValueError, match="JSON-compatible"):
        validate_trace_value(_Payload(), field_name="sample")

    with pytest.raises(ValueError, match="keys must be strings"):
        validate_trace_value({1: "bad-key"}, field_name="sample")


def test_trace_event_tags_normalized_and_immutable_copy() -> None:
    external: dict[str, str] = {"correlation_id": "c-1"}
    event = TraceEvent(
        event_id="e1",
        run_id="r1",
        seq=1,
        ts_utc="2026-01-01T00:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.RUNTIME,
        step="step",
        message="msg",
        tags=external,
    )
    external["correlation_id"] = "mutated"
    assert event.tags["correlation_id"] == "c-1"


def test_trace_event_to_dict_stable_shape() -> None:
    @dataclass(frozen=True)
    class _SampleDiag(DiagnosticPayload):
        flag: bool

        @classmethod
        def schema_id(cls) -> str:
            return "test.sample.v1"

        def to_dict(self) -> TraceObject:
            return {"flag": self.flag}

        def redact(self) -> _SampleDiag:
            return self

    event = TraceEvent(
        event_id="e1",
        run_id="r1",
        seq=2,
        ts_utc="2026-01-01T00:00:00Z",
        level=TraceLevel.WARNING,
        component=TraceComponent.TOOLS,
        step="tool.run",
        message="done",
        payload=_SampleDiag(flag=True),
        tags={"task_id": "t1"},
        artifact_refs=(TraceArtifactRef(artifact_id="a1", kind="json", size_bytes=3),),
    )
    exported = event.to_dict()
    assert set(exported.keys()) == {
        "event_id",
        "run_id",
        "seq",
        "ts_utc",
        "level",
        "component",
        "step",
        "message",
        "payload_schema_id",
        "payload_schema_version",
        "payload",
        "tags",
        "artifact_refs",
    }
    assert exported["payload_schema_id"] == "test.sample.v1"
    assert exported["payload"] == {"flag": True}
    assert exported["artifact_refs"] == [
        {"artifact_id": "a1", "kind": "json", "size_bytes": 3},
    ]


def test_tool_call_trace_arguments_and_raw_trace_are_json_safe() -> None:
    trace = ToolCallTrace(
        tool_name="search",
        arguments={"q": "pay", "limit": 2},
        output_preview="ok",
        success=True,
        error_message=None,
        raw_trace={"provider": "lab"},
    )
    assert trace.arguments == {"q": "pay", "limit": 2}
    assert trace.raw_trace == {"provider": "lab"}


def test_erl_qual_004_tracing_imports_public_contract_only() -> None:
    import platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution.result as result_mod
    import platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.tracing.port as port_mod

    assert result_mod.TraceEvent is TraceEvent
    assert port_mod.TraceEvent is TraceEvent
    assert "intergrax.runtime.nexus.tracing" not in inspect.getsource(result_mod)
    assert "intergrax.runtime.nexus.tracing" not in inspect.getsource(port_mod)


def test_only_one_trace_event_and_diagnostic_payload_definition() -> None:
    import intergrax.contracts.tracing.events as contract_events
    import intergrax.contracts.tracing.diagnostics as contract_diagnostics

    assert contract_events.TraceEvent.__module__ == "intergrax.contracts.tracing.events"
    assert contract_diagnostics.DiagnosticPayload.__module__ == "intergrax.contracts.tracing.diagnostics"
