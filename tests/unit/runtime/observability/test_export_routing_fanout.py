# © Artur Czarnecki. All rights reserved.

"""OBS-ROUTING-0 — problem signal export routing/fanout contract tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    observability_attribute_key,
)
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    InMemoryObservabilityExporter,
    ObservabilityExportEnvelope,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    try_export_observability_envelope,
)
from intergrax.runtime.observability.export_routing import (
    FanoutObservabilityExporter,
    ObservabilityExportRoute,
    route_matches_envelope,
)
from intergrax.runtime.observability.problem_export import envelope_from_problem_signal
from intergrax.runtime.observability.problem_signal import PlatformProblemSignal

pytestmark = pytest.mark.unit

_EXPORT_ROUTING_PATH = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "observability"
    / "export_routing.py"
)

_FORBIDDEN_EXPORT_ROUTING_IMPORTS = frozenset(
    {
        "ObservabilityEmitter",
        "RuntimeEventBus",
        "sentry_sdk",
        "elasticsearch",
        "opentelemetry",
        "httpx",
        "local_workspace_application",
    }
)


def _module_import_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[-1])
            for alias in node.names:
                names.add(alias.name)
    return names


def _problem_signal_envelope() -> ObservabilityExportEnvelope:
    return ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.PROBLEM_SIGNAL,
        problem_kind="lkw.retrieve_failed",
        problem_severity="error",
        problem_error_code="LKW_RETRIEVE_FAILED",
    )


def _route(
    *,
    route_id: str,
    exporter: InMemoryObservabilityExporter,
    enabled: bool = True,
    record_kinds: tuple[ExportRecordKind, ...] = (),
    problem_kinds: tuple[str, ...] = (),
    problem_severities: tuple[str, ...] = (),
    problem_error_codes: tuple[str, ...] = (),
) -> ObservabilityExportRoute:
    return ObservabilityExportRoute(
        route_id=route_id,
        exporter=exporter,
        enabled=enabled,
        record_kinds=record_kinds,
        problem_kinds=problem_kinds,
        problem_severities=problem_severities,
        problem_error_codes=problem_error_codes,
    )


class _FailingObservabilityExporter:
    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        raise RuntimeError("simulated exporter failure")


@pytest.mark.asyncio
async def test_matching_problem_signal_route_receives_envelope() -> None:
    inner = InMemoryObservabilityExporter()
    fanout = FanoutObservabilityExporter(
        [
            _route(
                route_id="problem-errors",
                exporter=inner,
                record_kinds=(ExportRecordKind.PROBLEM_SIGNAL,),
                problem_kinds=("lkw.retrieve_failed",),
                problem_severities=("error",),
                problem_error_codes=("LKW_RETRIEVE_FAILED",),
            )
        ]
    )

    envelope = _problem_signal_envelope()
    await fanout.export(envelope)

    assert len(inner.envelopes) == 1
    assert inner.envelopes[0] == envelope
    assert fanout.last_result is not None
    assert fanout.last_result.exported_count == 1
    assert fanout.last_result.selected_count == 1


@pytest.mark.asyncio
async def test_disabled_route_is_skipped() -> None:
    inner = InMemoryObservabilityExporter()
    fanout = FanoutObservabilityExporter(
        [_route(route_id="disabled-route", exporter=inner, enabled=False)]
    )

    await fanout.export(_problem_signal_envelope())

    assert inner.envelopes == []
    assert fanout.last_result is not None
    assert fanout.last_result.skipped_count == 1
    assert fanout.last_result.deliveries[0].reason == "route_disabled"


@pytest.mark.asyncio
async def test_record_kind_filter_skips_non_matching_route() -> None:
    inner = InMemoryObservabilityExporter()
    fanout = FanoutObservabilityExporter(
        [
            _route(
                route_id="problem-only",
                exporter=inner,
                record_kinds=(ExportRecordKind.PROBLEM_SIGNAL,),
            )
        ]
    )

    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT)
    await fanout.export(envelope)

    assert inner.envelopes == []
    assert fanout.last_result is not None
    assert fanout.last_result.deliveries[0].reason == "record_kind_not_matched"


@pytest.mark.parametrize(
    ("filter_name", "filter_value", "envelope_value", "expected_reason"),
    [
        ("problem_kinds", ("lkw.retrieve_failed",), "other.kind", "problem_kind_not_matched"),
        ("problem_severities", ("error",), "warning", "problem_severity_not_matched"),
        ("problem_error_codes", ("LKW_RETRIEVE_FAILED",), "OTHER_CODE", "problem_error_code_not_matched"),
    ],
)
@pytest.mark.asyncio
async def test_problem_filters_skip_non_matching_routes(
    filter_name: str,
    filter_value: tuple[str, ...],
    envelope_value: str,
    expected_reason: str,
) -> None:
    inner = InMemoryObservabilityExporter()
    route_kwargs = {
        "route_id": "filtered-route",
        "exporter": inner,
        filter_name: filter_value,
    }
    fanout = FanoutObservabilityExporter([_route(**route_kwargs)])

    envelope = _problem_signal_envelope().model_copy(
        update={
            "problem_kind": envelope_value if filter_name == "problem_kinds" else "lkw.retrieve_failed",
            "problem_severity": envelope_value if filter_name == "problem_severities" else "error",
            "problem_error_code": envelope_value
            if filter_name == "problem_error_codes"
            else "LKW_RETRIEVE_FAILED",
        }
    )
    await fanout.export(envelope)

    assert inner.envelopes == []
    assert fanout.last_result is not None
    assert fanout.last_result.deliveries[0].reason == expected_reason


@pytest.mark.asyncio
async def test_empty_filters_match_all() -> None:
    inner = InMemoryObservabilityExporter()
    fanout = FanoutObservabilityExporter([_route(route_id="match-all", exporter=inner)])

    await fanout.export(_problem_signal_envelope())

    assert len(inner.envelopes) == 1
    matched, reason = route_matches_envelope(fanout._routes[0], _problem_signal_envelope())
    assert matched is True
    assert reason == ""


@pytest.mark.asyncio
async def test_fanout_continues_after_one_exporter_fails() -> None:
    failing = _FailingObservabilityExporter()
    inner = InMemoryObservabilityExporter()
    fanout = FanoutObservabilityExporter(
        [
            ObservabilityExportRoute(route_id="failing-route", exporter=failing),
            _route(route_id="memory-route", exporter=inner),
        ]
    )

    envelope = _problem_signal_envelope()
    await fanout.export(envelope)

    assert len(inner.envelopes) == 1
    assert fanout.last_result is not None
    assert fanout.last_result.failed_count == 1
    assert fanout.last_result.exported_count == 1
    assert fanout.last_result.deliveries[0].reason == "exporter_failed"
    assert fanout.last_result.deliveries[1].reason == "exported"


def test_export_routing_module_avoids_emitter_vendor_and_application_imports() -> None:
    imports = _module_import_names(_EXPORT_ROUTING_PATH)
    assert _FORBIDDEN_EXPORT_ROUTING_IMPORTS.isdisjoint(imports)


class _LkwProblemAttributes(ApplicationObservabilityAttributes):
    namespace: str = "lkw"
    operation: str = "local.workspace.pipeline"
    pipeline_stage: str = "search"
    source_count: int = 2


@pytest.mark.asyncio
async def test_fanout_integrates_with_try_export_observability_envelope() -> None:
    inner = InMemoryObservabilityExporter()
    fanout = FanoutObservabilityExporter(
        [_route(route_id="policy-safe-route", exporter=inner)]
    )

    signal = PlatformProblemSignal(
        problem_kind="lkw.retrieve_failed",
        severity="error",
        error_code="LKW_RETRIEVE_FAILED",
        application_attributes=_LkwProblemAttributes(),
    )
    envelope = envelope_from_problem_signal(signal)

    result = await try_export_observability_envelope(
        envelope,
        exporter=fanout,
        policy=ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is True
    assert len(inner.envelopes) == 1
    downstream = inner.envelopes[0]
    assert downstream.application_attributes is None
    sanitized = downstream.sanitized_application_attributes
    assert sanitized is not None
    assert sanitized.attributes[observability_attribute_key("lkw", "pipeline_stage")] == "search"
    assert sanitized.attributes[observability_attribute_key("lkw", "source_count")] == 2
