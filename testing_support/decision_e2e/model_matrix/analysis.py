# © Artur Czarnecki. All rights reserved.

"""Multi-model alignment qualification analysis (DS-E2E-15J-L1.R6)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
)

from testing_support.decision_e2e.local_ai_incident_qualification import resolve_repository_head_sha
from testing_support.decision_e2e.local_qualification_session.alignment_revision_evidence import (
    infer_alignment_revision_evidence,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_analysis import (
    run_behavioral_analysis,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_evidence import (
    _parse_attempt_events,
    extract_run_evidence,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeReport,
    SourceFreezeStatus,
)
from testing_support.decision_e2e.local_qualification_session.contracts import SafetyGateOutcome
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)
from testing_support.decision_e2e.model_matrix.profiles import ModelQualificationProfile
from testing_support.decision_e2e.model_matrix.qualification_plan import (
    ModelAvailability,
    ProfileCohortPlan,
    qualification_artifact_root,
    summary_session_dir,
)
from testing_support.decision_e2e.model_matrix.registry import iter_qualification_profiles
from testing_support.decision_e2e.model_matrix.source_freeze import (
    TASK_ID,
    verify_model_matrix_source_freeze,
)
from testing_support.decision_e2e.natural_alignment.analysis import (
    COHORT_TASK_ID as R4R5_TASK_ID,
    CONTROLLED_PROOF_TASK_ID,
)

ANALYSIS_TASK_ID = "DS-E2E-15J-L1.R6.ANALYSIS"


class FifteenKBEffectR6(StrEnum):
    MODEL_PROVEN = "MODEL_PROVEN"
    NOT_TRIGGERED = "NOT_TRIGGERED"
    FAIL = "FAIL"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class PerModelMetrics:
    profile_key: str
    model_id: str
    provider: str
    availability: ModelAvailability
    total_runs: int
    evaluable_runs: int
    alignment_events: int
    match_rate: float
    reverse_mismatch_rate: float
    forward_mismatch_rate: float
    revision_attempted: int
    typed_context_delivered: int
    repair_count: int
    repair_rate: float
    third_pass: int
    reconciliation_leak: SafetyGateOutcome
    model_overcommit_count: int
    fifteen_kb_effect: FifteenKBEffectR6


@dataclass(frozen=True, slots=True)
class MultiModelQualificationResult:
    repo_source_freeze: SourceFreezeReport
    per_model: tuple[PerModelMetrics, ...]
    output_dir: Path
    matrix_status: str


def _load_runs(session_dir: Path) -> list[dict[str, object]]:
    path = session_dir / "runs.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    return [dict(item) for item in runs if isinstance(item, dict)]


def _count_alignment_events(runs: list[dict[str, object]]) -> int:
    total = 0
    for item in runs:
        trace_events = item.get("trace_events")
        if not isinstance(trace_events, list):
            continue
        events = tuple(dict(event) for event in trace_events if isinstance(event, dict))
        readback = read_typed_alignment_events(events)
        total += len(readback.events)
    return total


def _csv_bytes(rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buffer.getvalue()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output_dir: Path, files: tuple[str, ...]) -> None:
    lines = [f"{name} sha256:{_sha256_file(output_dir / name)}" for name in sorted(files)]
    body = "\n".join(lines) + ("\n" if lines else "")
    self_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    manifest_body = body + f"artifact-manifest.txt sha256:{self_digest}\n"
    (output_dir / "artifact-manifest.txt").write_text(manifest_body, encoding="utf-8")


def classify_fifteen_kb_effect(
    *,
    availability: ModelAvailability,
    model_overcommit_count: int,
    revision_attempted: int,
    typed_context_delivered: int,
    repair_count: int,
    total_runs: int,
    third_pass: int,
    reconciliation_leak: SafetyGateOutcome,
    alignment_events: int,
) -> FifteenKBEffectR6:
    if availability is ModelAvailability.MODEL_UNAVAILABLE:
        return FifteenKBEffectR6.MODEL_UNAVAILABLE
    if third_pass > 0 or reconciliation_leak is SafetyGateOutcome.FAIL:
        return FifteenKBEffectR6.FAIL
    if total_runs < 1 or alignment_events < 1:
        return FifteenKBEffectR6.INCONCLUSIVE
    repair_rate = repair_count / total_runs if total_runs else 0.0
    if model_overcommit_count > 0:
        if revision_attempted == 0 or typed_context_delivered == 0:
            return FifteenKBEffectR6.FAIL
        if repair_rate > 0:
            return FifteenKBEffectR6.MODEL_PROVEN
        return FifteenKBEffectR6.FAIL
    return FifteenKBEffectR6.NOT_TRIGGERED


def analyze_profile_session(
    *,
    repo_root: Path,
    profile: ModelQualificationProfile,
    session_dir: Path,
    availability: ModelAvailability,
) -> PerModelMetrics:
    runs = _load_runs(session_dir)
    alignment_events = _count_alignment_events(runs)
    if not runs or availability is ModelAvailability.MODEL_UNAVAILABLE:
        return PerModelMetrics(
            profile_key=profile.profile_key,
            model_id=profile.model_id,
            provider=profile.provider,
            availability=availability,
            total_runs=len(runs),
            evaluable_runs=0,
            alignment_events=alignment_events,
            match_rate=0.0,
            reverse_mismatch_rate=0.0,
            forward_mismatch_rate=0.0,
            revision_attempted=0,
            typed_context_delivered=0,
            repair_count=0,
            repair_rate=0.0,
            third_pass=0,
            reconciliation_leak=SafetyGateOutcome.PASS,
            model_overcommit_count=0,
            fifteen_kb_effect=classify_fifteen_kb_effect(
                availability=availability,
                model_overcommit_count=0,
                revision_attempted=0,
                typed_context_delivered=0,
                repair_count=0,
                total_runs=len(runs),
                third_pass=0,
                reconciliation_leak=SafetyGateOutcome.PASS,
                alignment_events=alignment_events,
            ),
        )

    bridge_dir = session_dir / ".analysis-r6-bridge"
    behavioral = run_behavioral_analysis(
        repo_root=repo_root,
        session_dir=session_dir,
        output_dir=bridge_dir,
    )
    total = behavioral.alignment.total_runs
    evaluable = behavioral.alignment.evaluable_runs
    repair_count = behavioral.revision.revision_repaired
    repair_rate = repair_count / total if total else 0.0
    match_rate = 0.0
    if alignment_events:
        mismatch = behavioral.alignment.alignment_mismatch_count
        match_rate = max(0.0, 1.0 - (mismatch / alignment_events))
    reverse_rate = behavioral.alignment.reverse_count / total if total else 0.0
    forward_rate = behavioral.alignment.forward_count / total if total else 0.0
    effect = classify_fifteen_kb_effect(
        availability=availability,
        model_overcommit_count=behavioral.alignment.reverse_count,
        revision_attempted=behavioral.revision.revision_attempted,
        typed_context_delivered=behavioral.revision.typed_context_present,
        repair_count=repair_count,
        total_runs=total,
        third_pass=behavioral.third_pass_count,
        reconciliation_leak=behavioral.reconciliation_leak,
        alignment_events=alignment_events,
    )
    return PerModelMetrics(
        profile_key=profile.profile_key,
        model_id=profile.model_id,
        provider=profile.provider,
        availability=availability,
        total_runs=total,
        evaluable_runs=evaluable,
        alignment_events=alignment_events,
        match_rate=match_rate,
        reverse_mismatch_rate=reverse_rate,
        forward_mismatch_rate=forward_rate,
        revision_attempted=behavioral.revision.revision_attempted,
        typed_context_delivered=behavioral.revision.typed_context_present,
        repair_count=repair_count,
        repair_rate=repair_rate,
        third_pass=behavioral.third_pass_count,
        reconciliation_leak=behavioral.reconciliation_leak,
        model_overcommit_count=behavioral.alignment.reverse_count,
        fifteen_kb_effect=effect,
    )


def _metrics_comparison_row(metrics: PerModelMetrics) -> dict[str, str]:
    return {
        "profile_key": metrics.profile_key,
        "model_id": metrics.model_id,
        "total_runs": str(metrics.total_runs),
        "evaluable_runs": str(metrics.evaluable_runs),
        "alignment_events": str(metrics.alignment_events),
        "match_rate": f"{metrics.match_rate:.4f}",
        "reverse_mismatch_rate": f"{metrics.reverse_mismatch_rate:.4f}",
        "forward_mismatch_rate": f"{metrics.forward_mismatch_rate:.4f}",
        "revision_attempted": str(metrics.revision_attempted),
        "typed_context_delivered": str(metrics.typed_context_delivered),
        "repair_count": str(metrics.repair_count),
        "repair_rate": f"{metrics.repair_rate:.4f}",
        "third_pass": str(metrics.third_pass),
        "reconciliation_leak": metrics.reconciliation_leak.value,
        "15K_B_EFFECT": metrics.fifteen_kb_effect.value,
        "availability": metrics.availability.value,
    }


def _collect_csv_rows(
    plans: tuple[ProfileCohortPlan, ...],
    per_model: tuple[PerModelMetrics, ...],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    comparison = [_metrics_comparison_row(m) for m in per_model]
    alignment_dist: list[dict[str, str]] = []
    revision_rows: list[dict[str, str]] = []
    attempt_rows: list[dict[str, str]] = []

    for plan in plans:
        session_dir = plan.session_dir
        runs = _load_runs(session_dir)
        for item in sorted(runs, key=lambda row: str(row.get("run_id", ""))):
            run_id = str(item.get("run_id", ""))
            evidence = extract_run_evidence(item)
            trace_events = item.get("trace_events")
            events: tuple[dict[str, object], ...] = ()
            if isinstance(trace_events, list):
                events = tuple(dict(event) for event in trace_events if isinstance(event, dict))
            readback = read_typed_alignment_events(events)
            for event in readback.events:
                if event.alignment_status is AlignmentStatus.MATCH:
                    bucket = "MATCH"
                elif event.alignment_direction is AlignmentDirection.REVERSE:
                    bucket = "MODEL_OVERCOMMIT"
                elif event.alignment_direction is AlignmentDirection.FORWARD:
                    bucket = "MODEL_UNDERCOMMIT"
                else:
                    bucket = "UNKNOWN"
                alignment_dist.append(
                    {
                        "profile_key": plan.profile.profile_key,
                        "run_id": run_id,
                        "bucket": bucket,
                        "alignment_status": event.alignment_status.value,
                    }
                )
            flags = infer_alignment_revision_evidence(
                readback.events,
                _parse_attempt_events(events),
            )
            revision_rows.append(
                {
                    "profile_key": plan.profile.profile_key,
                    "run_id": run_id,
                    "natural_overcommit_repair": str(flags.natural_overcommit_repair).lower(),
                    "revision_attempted": str(flags.revision_attempted).lower(),
                    "typed_context_present": str(flags.typed_context_present).lower(),
                    "revision_repaired": str(flags.revision_repaired).lower(),
                }
            )
            for attempt in evidence.attempt_events:
                attempt_rows.append(
                    {
                        "profile_key": plan.profile.profile_key,
                        "run_id": run_id,
                        "attempt_index": str(attempt.attempt_index),
                        "max_iterations": str(attempt.max_iterations),
                        "node_id": attempt.node_id,
                    }
                )
    return comparison, alignment_dist, revision_rows, attempt_rows


def run_multi_model_qualification_analysis(
    *,
    repo_root: Path,
    plans: tuple[ProfileCohortPlan, ...] | None = None,
) -> MultiModelQualificationResult:
    repo_freeze = verify_model_matrix_source_freeze(repo_root)
    output_dir = summary_session_dir(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if plans is None:
        artifact_root = qualification_artifact_root(repo_root)
        built: list[ProfileCohortPlan] = []
        for profile in iter_qualification_profiles():
            session_dir = artifact_root / profile.artifact_dir_name()
            availability = (
                ModelAvailability.AVAILABLE
                if (session_dir / "runs.json").is_file()
                else ModelAvailability.MODEL_UNAVAILABLE
            )
            built.append(
                ProfileCohortPlan(
                    profile=profile,
                    session_dir=session_dir,
                    run_count=0,
                    availability=availability,
                )
            )
        plans = tuple(built)

    per_model = tuple(
        analyze_profile_session(
            repo_root=repo_root,
            profile=plan.profile,
            session_dir=plan.session_dir,
            availability=plan.availability,
        )
        for plan in plans
    )

    comparison, alignment_dist, revision_rows, attempt_rows = _collect_csv_rows(
        plans,
        per_model,
    )

    analyzed_at = datetime.now(tz=UTC).isoformat()
    head_sha = resolve_repository_head_sha(repo_root)

    runs_snapshot: dict[str, object] = {
        "task_id": TASK_ID,
        "analyzed_at": analyzed_at,
        "profiles": [
            {
                "profile_key": plan.profile.profile_key,
                "model_id": plan.profile.model_id,
                "session_dir": str(plan.session_dir.relative_to(repo_root)),
                "run_count": len(_load_runs(plan.session_dir)),
                "availability": plan.availability.value,
            }
            for plan in plans
        ],
    }
    (output_dir / "runs.json").write_text(
        json.dumps(runs_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_payload = {
        "task_id": TASK_ID,
        "analyzed_at": analyzed_at,
        "SOURCE_FREEZE_STATUS": repo_freeze.status.value,
        "MODEL_MATRIX": {
            "profiles": [m.profile_key for m in per_model],
            "per_model_15K_B_EFFECT": {m.profile_key: m.fifteen_kb_effect.value for m in per_model},
        },
        "matrix_status": (
            "PASS"
            if repo_freeze.status is SourceFreezeStatus.PASS
            and not any(m.fifteen_kb_effect is FifteenKBEffectR6.FAIL for m in per_model)
            else "FAIL"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    analysis_payload = {
        "task": ANALYSIS_TASK_ID,
        "cohort_task": TASK_ID,
        "analyzed_at": analyzed_at,
        "SOURCE_FREEZE_STATUS": repo_freeze.status.value,
        "per_model": [
            {
                "profile_key": m.profile_key,
                "model_id": m.model_id,
                "15K_B_EFFECT": m.fifteen_kb_effect.value,
                "model_overcommit_count": m.model_overcommit_count,
                "repair_count": m.repair_count,
                "repair_rate": m.repair_rate,
                "revision_attempted": m.revision_attempted,
                "typed_context_delivered": m.typed_context_delivered,
                "alignment_events": m.alignment_events,
                "third_pass": m.third_pass,
                "reconciliation_leak": m.reconciliation_leak.value,
            }
            for m in per_model
        ],
        "controlled_vs_natural_vs_multi_model": {
            "R4.R4 Controlled": "PROVEN",
            "R4.R5 Qwen Natural": "NOT_PROVEN",
            "R6 Multi-model": summary_payload["matrix_status"],
        },
        "repository_head_sha": head_sha,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (output_dir / "model_comparison.csv").write_text(
        _csv_bytes(
            comparison,
            (
                "profile_key",
                "model_id",
                "total_runs",
                "evaluable_runs",
                "alignment_events",
                "match_rate",
                "reverse_mismatch_rate",
                "forward_mismatch_rate",
                "revision_attempted",
                "typed_context_delivered",
                "repair_count",
                "repair_rate",
                "third_pass",
                "reconciliation_leak",
                "15K_B_EFFECT",
                "availability",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "alignment_distribution.csv").write_text(
        _csv_bytes(
            alignment_dist,
            ("profile_key", "run_id", "bucket", "alignment_status"),
        ),
        encoding="utf-8",
    )
    (output_dir / "revision_effectiveness.csv").write_text(
        _csv_bytes(
            revision_rows,
            (
                "profile_key",
                "run_id",
                "natural_overcommit_repair",
                "revision_attempted",
                "typed_context_present",
                "revision_repaired",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "attempt_timeline.csv").write_text(
        _csv_bytes(
            attempt_rows,
            ("profile_key", "run_id", "attempt_index", "max_iterations", "node_id"),
        ),
        encoding="utf-8",
    )

    profile_json = {
        "task_id": TASK_ID,
        "profiles": [
            {
                "profile_key": plan.profile.profile_key,
                "provider": plan.profile.provider,
                "model": plan.profile.model_id,
                "digest": plan.profile.digest or None,
                "temperature": plan.profile.temperature,
                "expected_behavior_class": plan.profile.expected_behavior_class,
            }
            for plan in plans
        ],
    }
    (output_dir / "local_model_profile.json").write_text(
        json.dumps(profile_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    final_report = _build_final_report(
        repo_freeze=repo_freeze,
        per_model=per_model,
        head_sha=head_sha,
        matrix_status=str(summary_payload["matrix_status"]),
    )
    (output_dir / "final-report.md").write_text(final_report, encoding="utf-8")

    artifact_names = (
        "runs.json",
        "summary.json",
        "analysis.json",
        "model_comparison.csv",
        "alignment_distribution.csv",
        "revision_effectiveness.csv",
        "attempt_timeline.csv",
        "local_model_profile.json",
        "final-report.md",
    )
    _write_manifest(
        output_dir,
        tuple(name for name in artifact_names if (output_dir / name).is_file()),
    )

    return MultiModelQualificationResult(
        repo_source_freeze=repo_freeze,
        per_model=per_model,
        output_dir=output_dir,
        matrix_status=str(summary_payload["matrix_status"]),
    )


def _build_final_report(
    *,
    repo_freeze: SourceFreezeReport,
    per_model: tuple[PerModelMetrics, ...],
    head_sha: str,
    matrix_status: str,
) -> str:
    lines = [
        f"# {TASK_ID} — multi-model alignment reliability qualification",
        "",
        "## STATUS",
        f"- matrix_status: `{matrix_status}`",
        "",
        "## TASK",
        f"- `{TASK_ID}` Multi-Model Alignment Reliability Qualification",
        "",
        "## SOURCE_FREEZE",
        f"- SOURCE_FREEZE_STATUS: `{repo_freeze.status.value}`",
        "",
        "## MODEL_MATRIX",
    ]
    for metrics in per_model:
        lines.append(
            f"- `{metrics.profile_key}` (`{metrics.model_id}`): "
            f"availability=`{metrics.availability.value}`, "
            f"15K_B_EFFECT=`{metrics.fifteen_kb_effect.value}`"
        )
    lines.extend(
        [
            "",
            "## PER_MODEL_RESULTS",
            "",
            "| profile | runs | overcommit | repair | effect |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for m in per_model:
        lines.append(
            f"| {m.profile_key} | {m.total_runs} | {m.model_overcommit_count} | "
            f"{m.repair_count} | {m.fifteen_kb_effect.value} |"
        )
    lines.extend(
        [
            "",
            "## ALIGNMENT_DISTRIBUTION",
            "- See `alignment_distribution.csv` (MATCH / MODEL_OVERCOMMIT / MODEL_UNDERCOMMIT / UNKNOWN).",
            "",
            "## REVISION_RESULTS",
            "- See `revision_effectiveness.csv`.",
            "",
            "## REPAIR_RESULTS",
            "- Per-model `repair_rate` in `model_comparison.csv`.",
            "",
            "## CONTROLLED_VS_NATURAL_VS_MULTI_MODEL",
            "",
            "| Evidence | Status |",
            "| --- | --- |",
            f"| {CONTROLLED_PROOF_TASK_ID} Controlled | PROVEN |",
            f"| {R4R5_TASK_ID} Qwen Natural | NOT_PROVEN |",
            f"| R6 Multi-model | {matrix_status} |",
            "",
            "## SAFETY_GATES",
        ]
    )
    for m in per_model:
        lines.append(
            f"- `{m.profile_key}`: third_pass={m.third_pass}, "
            f"reconciliation_leak={m.reconciliation_leak.value}"
        )
    lines.extend(
        [
            "",
            "## REGRESSIONS",
            "- Run mandatory regression matrix (15I, 15K-B, O1, O1.R1, O2, QI1, QI2, R4.R4, R4.R5).",
            "",
            "## COMMIT SHA",
            f"- `{head_sha}`",
            "",
            "## NEXT TASK",
            "- Extend model registry or re-run cohorts when hardware permits unavailable profiles.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ANALYSIS_TASK_ID",
    "FifteenKBEffectR6",
    "MultiModelQualificationResult",
    "PerModelMetrics",
    "analyze_profile_session",
    "classify_fifteen_kb_effect",
    "run_multi_model_qualification_analysis",
]
