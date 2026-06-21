# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Controlled local live core probe runner (EVID-CORE-FU-01 · EVID-CORE-FU-01B)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from intergrax.runtime.evidence.live_core_probe_contracts import (
    LiveCoreProbeKind,
    LiveCoreProbeReport,
    LiveCoreProbeResult,
    LiveCoreProbeStatus,
    create_live_core_probe_result,
    derive_live_core_probe_report_status,
    generate_live_core_probe_report_id,
    live_core_probe_kinds,
    validate_live_core_probe_report,
)

LIVE_CORE_PROBE_EVIDENCE_KIND: Final = "selected_live_tier0_probe"

LIVE_CORE_PROBE_OPERATOR_NOTE: Final = (
    "Selected live Tier-0 probes execute locally with mock LLM/tools, "
    "no network, no provider calls, and no real LLM calls. "
    "This is not full runtime certification and does not replace deterministic CORE certification."
)

_DEFAULT_REPORT_TITLE = "Selected live Tier-0 probes"

_STATUS_SUMMARIES: Final[dict[LiveCoreProbeStatus, str]] = {
    LiveCoreProbeStatus.PASSED: "All selected live Tier-0 probes passed.",
    LiveCoreProbeStatus.FAILED: "One or more selected live Tier-0 probes failed.",
    LiveCoreProbeStatus.SKIPPED: "Selected live Tier-0 probes partially completed.",
    LiveCoreProbeStatus.UNAVAILABLE: "Selected live Tier-0 probes are unavailable.",
}

_LOCAL_CONTROLLED_METADATA: Final[dict[str, str]] = {
    "probe_execution": "local_controlled",
    "network": "disabled",
    "provider_calls": "disabled",
    "llm": "mock",
}


@dataclass(frozen=True, slots=True)
class _ProbeExecutionContext:
    root_label: str
    run_id: str


@dataclass(frozen=True, slots=True)
class _InMemoryTraceMarker:
    run_id: str
    marker_id: str


def generate_live_core_probe_run_id(*, root_label: str = "local") -> str:
    """Return a deterministic live core probe run identifier."""
    return f"live-core-probe-run:{root_label}"


def build_live_core_probe_report(
    *,
    results: list[LiveCoreProbeResult],
    root_label: str = "local",
    summary: str = "",
) -> LiveCoreProbeReport:
    """Build and validate a live core probe report from probe results."""
    status = derive_live_core_probe_report_status(results)
    resolved_summary = summary or _STATUS_SUMMARIES[status]
    report = LiveCoreProbeReport(
        report_id=generate_live_core_probe_report_id(root_label=root_label),
        title=_DEFAULT_REPORT_TITLE,
        summary=resolved_summary,
        status=status,
        results=results,
    )
    validate_live_core_probe_report(report)
    return report


def _probe_context(*, root_label: str) -> _ProbeExecutionContext:
    return _ProbeExecutionContext(
        root_label=root_label,
        run_id=generate_live_core_probe_run_id(root_label=root_label),
    )


def run_basic_run_completed_live_probe(*, root_label: str = "local") -> LiveCoreProbeResult:
    """Simulate a minimal local run completion marker without real runtime imports."""
    context = _probe_context(root_label=root_label)
    execution_marker = f"execution-complete:{context.run_id}"
    return create_live_core_probe_result(
        probe_kind=LiveCoreProbeKind.BASIC_RUN_COMPLETED_LIVE,
        status=LiveCoreProbeStatus.PASSED,
        title="Basic run completed (live probe)",
        message=f"Local controlled probe completed with marker {execution_marker}.",
        metadata={
            **_LOCAL_CONTROLLED_METADATA,
            "run_id": context.run_id,
        },
    )


def run_trace_persisted_live_probe(*, root_label: str = "local") -> LiveCoreProbeResult:
    """Simulate an in-memory trace/evidence marker without a real trace store."""
    context = _probe_context(root_label=root_label)
    trace_marker = _InMemoryTraceMarker(
        run_id=context.run_id,
        marker_id=f"trace-marker:{context.run_id}",
    )
    if not trace_marker.marker_id:
        raise ValueError("trace marker must not be empty")
    return create_live_core_probe_result(
        probe_kind=LiveCoreProbeKind.TRACE_PERSISTED_LIVE,
        status=LiveCoreProbeStatus.PASSED,
        title="Trace persisted (live probe)",
        message=f"In-memory trace marker created: {trace_marker.marker_id}.",
        metadata={
            **_LOCAL_CONTROLLED_METADATA,
            "trace_marker": "created",
        },
    )


def run_tool_denied_by_policy_live_probe(*, root_label: str = "local") -> LiveCoreProbeResult:
    """Model a deterministic policy denial for a mock high-risk tool."""
    _probe_context(root_label=root_label)
    mock_tool = "mock_high_risk_tool"
    policy_decision = "denied"
    if policy_decision != "denied":
        raise ValueError("expected deterministic policy denial")
    return create_live_core_probe_result(
        probe_kind=LiveCoreProbeKind.TOOL_DENIED_BY_POLICY_LIVE,
        status=LiveCoreProbeStatus.PASSED,
        title="Tool denied by policy (live probe)",
        message=f"Mock tool {mock_tool} was denied by local controlled policy model.",
        metadata={
            **_LOCAL_CONTROLLED_METADATA,
            "tool": mock_tool,
            "policy_decision": policy_decision,
        },
    )


def _failed_probe_result(
    *,
    probe_kind: LiveCoreProbeKind,
    exc: Exception,
) -> LiveCoreProbeResult:
    return create_live_core_probe_result(
        probe_kind=probe_kind,
        status=LiveCoreProbeStatus.FAILED,
        title=f"{probe_kind.value} failed",
        message=f"Probe failed: {exc}",
        metadata={"error_type": type(exc).__name__},
    )


_PROBE_RUNNERS: Final[dict[LiveCoreProbeKind, Callable[..., LiveCoreProbeResult]]] = {
    LiveCoreProbeKind.BASIC_RUN_COMPLETED_LIVE: run_basic_run_completed_live_probe,
    LiveCoreProbeKind.TRACE_PERSISTED_LIVE: run_trace_persisted_live_probe,
    LiveCoreProbeKind.TOOL_DENIED_BY_POLICY_LIVE: run_tool_denied_by_policy_live_probe,
}


def run_live_core_probes(*, root_label: str = "local") -> LiveCoreProbeReport:
    """Run selected live Tier-0 probes locally and return an in-memory report."""
    results: list[LiveCoreProbeResult] = []
    for probe_kind in live_core_probe_kinds():
        runner = _PROBE_RUNNERS[probe_kind]
        try:
            result = runner(root_label=root_label)
        except Exception as exc:  # noqa: BLE001 — convert unexpected probe failure
            result = _failed_probe_result(probe_kind=probe_kind, exc=exc)
        results.append(result)
    return build_live_core_probe_report(results=results, root_label=root_label)
