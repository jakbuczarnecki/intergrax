# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import json

from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata, RunSummary
from intergrax.tools.providers.harness.contracts import (
    HarnessCompareRunsInput,
    HarnessCompareRunsOutput,
    HarnessExportRunBundleInput,
    HarnessExportRunBundleOutput,
    HarnessGetRunCostInput,
    HarnessGetRunCostOutput,
    HarnessGetRunEventsInput,
    HarnessGetRunEventsOutput,
    HarnessGetRunInput,
    HarnessGetRunOutput,
    HarnessListRunsInput,
    HarnessListRunsOutput,
    HarnessRunComparisonOutput,
    HarnessRunEventOutput,
    HarnessRunMetadataOutput,
    HarnessRunSummaryOutput,
)
from intergrax.tools.registry.runtime_bindings import RunTraceReaderBinding
from intergrax.tools.registry.wiring import ToolWiringContext

HARNESS_GET_RUN_TOOL_ID = "harness.get_run"
HARNESS_LIST_RUNS_TOOL_ID = "harness.list_runs"
HARNESS_GET_RUN_COST_TOOL_ID = "harness.get_run_cost"
HARNESS_GET_RUN_EVENTS_TOOL_ID = "harness.get_run_events"
HARNESS_COMPARE_RUNS_TOOL_ID = "harness.compare_runs"
HARNESS_EXPORT_RUN_BUNDLE_TOOL_ID = "harness.export_run_bundle"


def _require_trace_reader(ctx: ToolWiringContext) -> RunTraceReaderBinding:
    reader = ctx.trace_reader
    if reader is None:
        raise RuntimeError("trace_reader_not_configured")
    return reader


def _read_persisted(reader: RunTraceReaderBinding, run_id: str, tenant_id: str) -> PersistedRun:
    persisted = reader.read_run(run_id, tenant_id)
    if not isinstance(persisted, PersistedRun):
        raise RuntimeError("trace_reader_invalid_result")
    return persisted


def _metadata_output(metadata: RunMetadata) -> HarnessRunMetadataOutput:
    error_type = ""
    error_message = ""
    if metadata.error is not None:
        error_type = metadata.error.error_type.value
        error_message = metadata.error.message
    return HarnessRunMetadataOutput(
        run_id=metadata.run_id,
        session_id=metadata.session_id,
        user_id=metadata.user_id,
        tenant_id=metadata.tenant_id,
        started_at_utc=metadata.started_at_utc,
        duration_ms=metadata.stats.duration_ms,
        llm_usage=dict(metadata.stats.llm_usage),
        error_type=error_type,
        error_message=error_message,
    )


def harness_get_run(ctx: ToolWiringContext, params: HarnessGetRunInput) -> HarnessGetRunOutput:
    persisted = _read_persisted(
        _require_trace_reader(ctx),
        params.run_id.strip(),
        params.tenant_id.strip(),
    )
    events = [dict(item) for item in persisted.events]
    return HarnessGetRunOutput(
        metadata=_metadata_output(persisted.metadata),
        events=events,
        event_count=len(events),
    )


def harness_list_runs(ctx: ToolWiringContext, params: HarnessListRunsInput) -> HarnessListRunsOutput:
    raw = _require_trace_reader(ctx).list_runs(params.tenant_id.strip(), limit=params.limit)
    runs: list[HarnessRunSummaryOutput] = []
    for item in raw:
        if not isinstance(item, RunSummary):
            continue
        runs.append(
            HarnessRunSummaryOutput(
                run_id=item.run_id,
                tenant_id=item.tenant_id,
                user_id=item.user_id,
                session_id=item.session_id,
                started_at_utc=item.started_at_utc,
                duration_ms=item.duration_ms,
                event_count=item.event_count,
            )
        )
    return HarnessListRunsOutput(runs=runs, total=len(runs))


def harness_get_run_cost(ctx: ToolWiringContext, params: HarnessGetRunCostInput) -> HarnessGetRunCostOutput:
    persisted = _read_persisted(
        _require_trace_reader(ctx),
        params.run_id.strip(),
        params.tenant_id.strip(),
    )
    meta_out = _metadata_output(persisted.metadata)
    return HarnessGetRunCostOutput(
        run_id=meta_out.run_id,
        tenant_id=meta_out.tenant_id,
        duration_ms=meta_out.duration_ms,
        llm_usage=meta_out.llm_usage,
    )


def _event_output(raw: dict[str, object]) -> HarnessRunEventOutput:
    payload = raw.get("payload")
    return HarnessRunEventOutput(
        event_id=str(raw.get("event_id", "")),
        step=str(raw.get("step", "")),
        level=str(raw.get("level", "")),
        message=str(raw.get("message", "")),
        ts_utc=str(raw.get("ts_utc", "")),
        payload=dict(payload) if isinstance(payload, dict) else {},
    )


def harness_get_run_events(ctx: ToolWiringContext, params: HarnessGetRunEventsInput) -> HarnessGetRunEventsOutput:
    persisted = _read_persisted(
        _require_trace_reader(ctx),
        params.run_id.strip(),
        params.tenant_id.strip(),
    )
    step_filter = params.step.strip()
    level_filter = params.level.strip().upper()
    filtered: list[HarnessRunEventOutput] = []
    for item in persisted.events:
        event = dict(item)
        if step_filter and str(event.get("step", "")) != step_filter:
            continue
        if level_filter and str(event.get("level", "")).upper() != level_filter:
            continue
        filtered.append(_event_output(event))
        if len(filtered) >= params.limit:
            break
    return HarnessGetRunEventsOutput(
        run_id=params.run_id.strip(),
        events=filtered,
        total=len(filtered),
    )


def _comparison_output(persisted: PersistedRun) -> HarnessRunComparisonOutput:
    meta = _metadata_output(persisted.metadata)
    return HarnessRunComparisonOutput(
        run_id=meta.run_id,
        duration_ms=meta.duration_ms,
        event_count=len(persisted.events),
        llm_usage=dict(meta.llm_usage),
        error_type=meta.error_type,
    )


def harness_compare_runs(ctx: ToolWiringContext, params: HarnessCompareRunsInput) -> HarnessCompareRunsOutput:
    reader = _require_trace_reader(ctx)
    baseline = _read_persisted(reader, params.baseline_run_id.strip(), params.tenant_id.strip())
    candidate = _read_persisted(reader, params.candidate_run_id.strip(), params.tenant_id.strip())
    baseline_out = _comparison_output(baseline)
    candidate_out = _comparison_output(candidate)
    return HarnessCompareRunsOutput(
        baseline=baseline_out,
        candidate=candidate_out,
        duration_delta_ms=candidate_out.duration_ms - baseline_out.duration_ms,
        event_count_delta=candidate_out.event_count - baseline_out.event_count,
    )


def harness_export_run_bundle(
    ctx: ToolWiringContext,
    params: HarnessExportRunBundleInput,
) -> HarnessExportRunBundleOutput:
    persisted = _read_persisted(
        _require_trace_reader(ctx),
        params.run_id.strip(),
        params.tenant_id.strip(),
    )
    events = [dict(item) for item in persisted.events[: params.max_events]]
    payload = {
        "metadata": _metadata_output(persisted.metadata).model_dump(),
        "events": events,
        "truncated": len(persisted.events) > len(events),
    }
    return HarnessExportRunBundleOutput(
        run_id=params.run_id.strip(),
        bundle_json=json.dumps(payload, indent=2),
        event_count=len(events),
    )
