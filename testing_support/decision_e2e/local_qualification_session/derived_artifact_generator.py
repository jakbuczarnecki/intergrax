# © Artur Czarnecki. All rights reserved.

"""Derived CSV/JSON artifacts from analysis and typed trace readback."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from testing_support.decision_e2e.local_qualification_session.alignment_revision_evidence import (
    infer_alignment_revision_evidence,
)
from testing_support.decision_e2e.local_qualification_session.atomic_io import (
    atomic_write_json,
    atomic_write_text,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationExperimentIdentity,
    QualificationRuntimeIdentity,
    QualificationSpec,
)
from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_evidence import (
    _parse_attempt_events,
)
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)


class DerivedArtifactGenerationError(RuntimeError):
    """Derived artifact could not be produced from observed inputs."""


class LocalModelProfileError(DerivedArtifactGenerationError):
    """local_model_profile.json requires experiment identity and probe result."""


R4R1_CONTEXT_WINDOW = 32768


def _runs_list(runs_payload: dict[str, object]) -> list[dict[str, object]]:
    runs = runs_payload.get("runs")
    if not isinstance(runs, list):
        return []
    return [dict(item) for item in runs if isinstance(item, dict)]


def _csv_line(rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buffer.getvalue()


def build_failure_cohort_rows(
    analysis: dict[str, object],
    runs_payload: dict[str, object],
) -> list[dict[str, str]]:
    analysis_runs = analysis.get("runs")
    by_run_id: dict[str, dict[str, object]] = {}
    if isinstance(analysis_runs, list):
        for item in analysis_runs:
            if isinstance(item, dict):
                run_id = item.get("run_id")
                if isinstance(run_id, str):
                    by_run_id[run_id] = item
    rows: list[dict[str, str]] = []
    for item in _runs_list(runs_payload):
        run_id = str(item.get("run_id", ""))
        source = by_run_id.get(run_id, {})
        rows.append(
            {
                "run_id": run_id,
                "category": str(source.get("category", "")),
                "reason": str(source.get("reason", "")),
                "boundary": str(source.get("boundary", "")),
                "owner": str(source.get("owner", "")),
                "validation_error": str(source.get("validation_error", "")),
                "final_state": str(source.get("final_state", "")),
            }
        )
    return rows


def build_revision_effectiveness_rows(
    runs_payload: dict[str, object],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in _runs_list(runs_payload):
        run_id = str(item.get("run_id", ""))
        trace_events = item.get("trace_events")
        events: tuple[dict[str, object], ...] = ()
        if isinstance(trace_events, list):
            events = tuple(dict(event) for event in trace_events if isinstance(event, dict))
        readback = read_typed_alignment_events(events)
        revision_flags = infer_alignment_revision_evidence(
            readback.events,
            _parse_attempt_events(events),
        )
        event = readback.events[-1] if readback.events else None
        rows.append(
            {
                "run_id": run_id,
                "alignment_direction": event.alignment_direction.value if event else "",
                "revision_attempted": str(revision_flags.revision_attempted).lower(),
                "typed_context_present": str(revision_flags.typed_context_present).lower(),
                "revision_repaired": str(revision_flags.revision_repaired).lower(),
                "revision_exhausted": str(
                    revision_flags.revision_attempted and not revision_flags.revision_repaired
                ).lower(),
                "success_after_revision": str(revision_flags.revision_repaired).lower(),
            }
        )
    return rows


def build_alignment_direction_rows(
    runs_payload: dict[str, object],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in _runs_list(runs_payload):
        run_id = str(item.get("run_id", ""))
        trace_events = item.get("trace_events")
        events: tuple[dict[str, object], ...] = ()
        if isinstance(trace_events, list):
            events = tuple(dict(event) for event in trace_events if isinstance(event, dict))
        readback = read_typed_alignment_events(events)
        event = readback.events[-1] if readback.events else None
        direction = event.alignment_direction if event else None
        mismatch = (
            event.alignment_status is AlignmentStatus.MISMATCH if event else False
        )
        rows.append(
            {
                "run_id": run_id,
                "forward_mismatch": str(
                    direction is AlignmentDirection.FORWARD and mismatch
                ).lower(),
                "reverse_mismatch": str(
                    direction is AlignmentDirection.REVERSE and mismatch
                ).lower(),
                "correctable": str(event.correctable if event else False).lower(),
                "direction": direction.value if direction else "",
                "final_state": _terminal_outcome(item),
            }
        )
    return rows


def _terminal_outcome(run_item: dict[str, object]) -> str:
    run_result = run_item.get("run_result")
    if isinstance(run_result, dict):
        terminal = run_result.get("terminal_outcome")
        if isinstance(terminal, str):
            return terminal
    return ""


def build_local_model_profile(
    identity: QualificationExperimentIdentity,
    observed: QualificationRuntimeIdentity,
    *,
    temperature: float,
    context_window: int = R4R1_CONTEXT_WINDOW,
) -> dict[str, Any]:
    if not identity.model_name:
        raise LocalModelProfileError("experiment identity missing model_name")
    if observed.model_name is None or observed.model_digest is None:
        raise LocalModelProfileError("probe result missing model identity")
    if observed.runtime_version is None:
        raise LocalModelProfileError("probe result missing runtime_version")
    return {
        "provider": identity.provider_kind,
        "runtime_version": observed.runtime_version.normalized(),
        "model": observed.model_name,
        "digest": observed.model_digest,
        "temperature": temperature,
        "context_window": context_window,
    }


def generate_derived_artifacts(
    session_dir: Path,
    *,
    spec: QualificationSpec,
    analysis: dict[str, object],
    runs_payload: dict[str, object],
    observed: QualificationRuntimeIdentity | None,
    temperature: float,
    overwrite: bool = True,
) -> None:
    failure_rows = build_failure_cohort_rows(analysis, runs_payload)
    revision_rows = build_revision_effectiveness_rows(runs_payload)
    alignment_rows = build_alignment_direction_rows(runs_payload)

    failure_fields = (
        "run_id",
        "category",
        "reason",
        "boundary",
        "owner",
        "validation_error",
        "final_state",
    )
    revision_fields = (
        "run_id",
        "alignment_direction",
        "revision_attempted",
        "typed_context_present",
        "revision_repaired",
        "revision_exhausted",
        "success_after_revision",
    )
    alignment_fields = (
        "run_id",
        "forward_mismatch",
        "reverse_mismatch",
        "correctable",
        "direction",
        "final_state",
    )

    targets: list[tuple[str, str]] = [
        ("failure_cohort.csv", _csv_line(failure_rows, failure_fields)),
        ("revision_effectiveness.csv", _csv_line(revision_rows, revision_fields)),
        ("alignment_direction.csv", _csv_line(alignment_rows, alignment_fields)),
    ]

    for name, body in targets:
        path = session_dir / name
        if path.is_file() and not overwrite:
            continue
        atomic_write_text(path, body)

    profile_path = session_dir / "local_model_profile.json"
    if observed is None:
        if not profile_path.is_file():
            raise LocalModelProfileError("observed runtime identity required for model profile")
        return
    if profile_path.is_file() and not overwrite:
        return
    profile = build_local_model_profile(
        spec.experiment_identity,
        observed,
        temperature=temperature,
    )
    atomic_write_json(profile_path, profile)


def parse_runs_payload(session_dir: Path) -> dict[str, object]:
    runs_path = session_dir / "runs.json"
    payload = json.loads(runs_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DerivedArtifactGenerationError("runs.json must be an object")
    return payload
