# © Artur Czarnecki. All rights reserved.

"""OBS-PROBLEM-3 — problem emission boundary contract (code-side guardrails)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.observability.problem_reporter import (
    ProblemReportContext,
    ProblemReporter,
    build_problem_signal,
    report_problem,
)
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_SOURCE_LAYER_APPLICATION,
    PlatformProblemSignal,
)

pytestmark = pytest.mark.unit

_PROBLEM_REPORTER_PATH = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "observability"
    / "problem_reporter.py"
)

_FORBIDDEN_PROBLEM_REPORTER_IMPORTS = frozenset(
    {
        "ObservabilityEmitter",
        "RuntimeEvent",
        "RuntimeEventBus",
        "sentry_sdk",
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


def test_build_problem_signal_does_not_require_runtime_event() -> None:
    context = ProblemReportContext(
        run_id="run-fake-003",
        task_id="task-fake-003",
        correlation_id="corr-fake-003",
    )
    signal = build_problem_signal(
        context=context,
        problem_kind="lkw.retrieve_failed",
        severity="error",
        error_code="LKW_RETRIEVE_FAILED",
        source_layer=PROBLEM_SOURCE_LAYER_APPLICATION,
        source_component="boundary_contract_test",
    )

    assert isinstance(signal, PlatformProblemSignal)
    assert signal.problem_kind == "lkw.retrieve_failed"
    assert signal.run_id == "run-fake-003"
    assert signal.correlation_id == "corr-fake-003"


@pytest.mark.asyncio
async def test_problem_reporter_reports_without_runtime_event_dependency() -> None:
    from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy

    reporter = ProblemReporter(
        context=ProblemReportContext(
            run_id="run-fake-004",
            task_id="task-fake-004",
            correlation_id="corr-fake-004",
        ),
        policy=ObservabilityExportPolicy(enabled=True),
    )

    result = await reporter.report(
        problem_kind="lkw.retrieve_failed",
        error_code="LKW_RETRIEVE_FAILED",
        source_layer=PROBLEM_SOURCE_LAYER_APPLICATION,
        source_component="boundary_contract_test",
    )

    assert result.exported is True
    assert result.envelope is not None
    assert result.envelope.problem_kind == "lkw.retrieve_failed"


def test_report_problem_signature_has_no_runtime_event_or_exception_params() -> None:
    params = report_problem.__code__.co_varnames[: report_problem.__code__.co_argcount]
    forbidden = {"runtime_event", "exception", "exc", "trace_event"}
    assert forbidden.isdisjoint(params)


def test_problem_reporter_module_avoids_emitter_runtime_event_and_vendor_sdk() -> None:
    imports = _module_import_names(_PROBLEM_REPORTER_PATH)
    assert _FORBIDDEN_PROBLEM_REPORTER_IMPORTS.isdisjoint(imports)
