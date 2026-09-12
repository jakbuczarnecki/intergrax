# © Artur Czarnecki. All rights reserved.

"""Qualification artifact emission for DS-E2E-15J-L1.R4.R4."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from testing_support.decision_e2e.controlled_alignment.evidence import (
    build_attempt_timeline_rows,
)
from testing_support.decision_e2e.controlled_alignment.runner import (
    ControlledAlignmentRunResult,
    run_result_to_run_record,
)
from testing_support.decision_e2e.controlled_alignment.source_freeze import (
    TASK_ID,
    verify_controlled_alignment_source_freeze,
)
from testing_support.decision_e2e.local_qualification_session.atomic_io import (
    atomic_write_text,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeStatus,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    COMPLETION_ALIGNMENT_TRACE_SCHEMA,
    CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
)


class QualificationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class ControlledAlignmentArtifactResult:
    output_dir: Path
    status: QualificationStatus
    source_freeze_status: SourceFreezeStatus


def default_artifact_dir(repo_root: Path) -> Path:
    return repo_root / ".artifacts" / "qualification" / TASK_ID


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _csv_from_rows(rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buffer.getvalue()


def _trace_manifest(trace_events: tuple[dict[str, object], ...]) -> dict[str, object]:
    schema_ids: dict[str, int] = {}
    for event in trace_events:
        schema = event.get("payload_schema_id")
        if isinstance(schema, str):
            schema_ids[schema] = schema_ids.get(schema, 0) + 1
    return {
        "task_id": TASK_ID,
        "event_count": len(trace_events),
        "schema_counts": schema_ids,
        "required_schemas": {
            "alignment": COMPLETION_ALIGNMENT_TRACE_SCHEMA,
            "model_attempt": CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
        },
    }


def write_qualification_artifacts(
    repo_root: Path,
    run_result: ControlledAlignmentRunResult,
    *,
    output_dir: Path | None = None,
) -> ControlledAlignmentArtifactResult:
    target = output_dir or default_artifact_dir(repo_root)
    target.mkdir(parents=True, exist_ok=True)

    source_freeze = verify_controlled_alignment_source_freeze(repo_root)
    run_record = run_result_to_run_record(run_result)
    runs_payload = {"task_id": TASK_ID, "runs": [run_record]}
    repair = run_result.repair_evidence
    pre = repair.alignment.pre_repair
    post = repair.alignment.post_repair

    summary = {
        "task_id": TASK_ID,
        "status": QualificationStatus.PASS.value
        if repair.repair_status.value == "PASS"
        and source_freeze.status is SourceFreezeStatus.PASS
        else QualificationStatus.FAIL.value,
        "SOURCE_FREEZE_STATUS": source_freeze.status.value,
        "controlled_stimulus": run_result.scenario.to_dict(),
        "alignment_result": {
            "direction": (
                repair.alignment.correction_direction.value
                if repair.alignment.correction_direction
                else None
            ),
            "correctable": repair.alignment.correctable,
            "pre_repair_status": pre.alignment_status.value if pre else None,
            "post_repair_status": post.alignment_status.value if post else None,
        },
        "revision_result": {
            "revision_context_valid": repair.revision_context_valid,
            "attempt_indices": list(repair.attempt_indices),
        },
        "repair_result": repair.repair_status.value,
    }

    alignment_rows = [
        {
            "run_id": str(run_record.get("run_id", "")),
            "direction": summary["alignment_result"]["direction"] or "",
            "correctable": str(summary["alignment_result"]["correctable"]).lower(),
            "pre_status": summary["alignment_result"]["pre_repair_status"] or "",
            "post_status": summary["alignment_result"]["post_repair_status"] or "",
        }
    ]
    revision_rows = [
        {
            "run_id": str(run_record.get("run_id", "")),
            "revision_context_valid": str(repair.revision_context_valid).lower(),
            "repair_status": repair.repair_status.value,
            "attempt_0": str(0 in repair.attempt_indices).lower(),
            "attempt_1": str(1 in repair.attempt_indices).lower(),
        }
    ]
    timeline_rows = build_attempt_timeline_rows(run_result.trace_events)
    timeline_fields = (
        "run_id",
        "node_id",
        "attempt_index",
        "alignment_status",
        "alignment_direction",
        "correctable",
    )

    report_md = "\n".join(
        [
            f"# {TASK_ID}",
            "",
            f"- STATUS: `{summary['status']}`",
            f"- SOURCE_FREEZE_STATUS: `{summary['SOURCE_FREEZE_STATUS']}`",
            f"- REPAIR: `{summary['repair_result']}`",
            "",
            "## Controlled stimulus",
            "",
            f"- scenario: `{run_result.scenario.scenario_id}`",
            f"- expected direction: `{run_result.scenario.expected_direction.value}`",
            "",
            "## Trace evidence",
            "",
            f"- alignment events: `{len([e for e in run_result.trace_events if e.get('payload_schema_id') == COMPLETION_ALIGNMENT_TRACE_SCHEMA])}`",
            f"- model attempts: `{len(repair.attempt_indices)}`",
            "",
        ]
    )
    final_report = report_md + "\n## Repair acceptance\n\n" + (
        "PASS — bounded reverse alignment correction observed."
        if summary["status"] == QualificationStatus.PASS.value
        else "FAIL — repair chain incomplete."
    )

    files: list[tuple[str, str]] = [
        ("runs.json", json.dumps(runs_payload, indent=2) + "\n"),
        ("summary.json", json.dumps(summary, indent=2) + "\n"),
        ("report.md", report_md),
        (
            "alignment_direction.csv",
            _csv_from_rows(
                alignment_rows,
                ("run_id", "direction", "correctable", "pre_status", "post_status"),
            ),
        ),
        (
            "revision_effectiveness.csv",
            _csv_from_rows(
                revision_rows,
                (
                    "run_id",
                    "revision_context_valid",
                    "repair_status",
                    "attempt_0",
                    "attempt_1",
                ),
            ),
        ),
        ("attempt_timeline.csv", _csv_from_rows(timeline_rows, timeline_fields)),
        (
            "trace_manifest.json",
            json.dumps(_trace_manifest(run_result.trace_events), indent=2) + "\n",
        ),
        ("final-report.md", final_report),
    ]

    manifest_lines: list[str] = []
    for name, body in files:
        path = target / name
        atomic_write_text(path, body)
        manifest_lines.append(f"{name} sha256:{_sha256_text(body)}")

    manifest_body = "\n".join(manifest_lines) + "\n"
    atomic_write_text(target / "artifact-manifest.txt", manifest_body)

    status = (
        QualificationStatus.PASS
        if summary["status"] == QualificationStatus.PASS.value
        else QualificationStatus.FAIL
    )
    return ControlledAlignmentArtifactResult(
        output_dir=target,
        status=status,
        source_freeze_status=source_freeze.status,
    )


__all__ = [
    "ControlledAlignmentArtifactResult",
    "QualificationStatus",
    "default_artifact_dir",
    "write_qualification_artifacts",
]
