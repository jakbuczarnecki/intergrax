# © Artur Czarnecki. All rights reserved.

"""Build qualification analysis.json from canonical runs.json."""

from __future__ import annotations

from typing import Any

from testing_support.decision_e2e.local_qualification_session.classification_adapter import (
    ClassificationParseError,
    classification_from_persisted_dict,
    failure_view_from_classification,
)
from testing_support.decision_e2e.local_qualification_session.contracts import QualificationSpec
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)


def build_qualification_analysis(
    runs_payload: dict[str, object],
    spec: QualificationSpec,
) -> dict[str, object]:
    runs = runs_payload.get("runs")
    per_run: list[dict[str, object]] = []
    alignment_statuses: list[str] = []
    platform_failures = 0
    if isinstance(runs, list):
        for item in runs:
            if not isinstance(item, dict):
                continue
            run_id = item.get("run_id")
            trace_events = item.get("trace_events")
            events: tuple[dict[str, object], ...] = ()
            if isinstance(trace_events, list):
                events = tuple(
                    dict(event) for event in trace_events if isinstance(event, dict)
                )
            readback = read_typed_alignment_events(events)
            alignment_statuses.append(readback.status.value)
            failure_row: dict[str, Any] = {
                "run_id": run_id if isinstance(run_id, str) else "",
                "alignment_readback_status": readback.status.value,
            }
            run_result = item.get("run_result")
            if isinstance(run_result, dict):
                classification = run_result.get("classification")
                if isinstance(classification, dict):
                    try:
                        typed = classification_from_persisted_dict(classification)
                        view = failure_view_from_classification(typed)
                        failure_row["category"] = view.category.value
                        failure_row["reason"] = view.reason.value
                        failure_row["boundary"] = view.boundary.value
                        failure_row["owner"] = view.owner.value
                        if view.is_platform_failure:
                            platform_failures += 1
                    except ClassificationParseError:
                        failure_row["classification_parse_error"] = True
                final_state = run_result.get("terminal_outcome")
                if isinstance(final_state, str):
                    failure_row["final_state"] = final_state
                validation_errors = run_result.get("validation_error_categories")
                if isinstance(validation_errors, list):
                    failure_row["validation_error"] = "|".join(
                        str(entry) for entry in validation_errors
                    )
            per_run.append(failure_row)
    return {
        "qualification_session_schema_version": "qualification_analysis.v1",
        "planned_runs": spec.run_count,
        "typed_alignment_readback_statuses": alignment_statuses,
        "platform_failure_count": platform_failures,
        "max_evaluator_attempt_index": spec.max_evaluator_attempt_index,
        "runs": per_run,
    }
