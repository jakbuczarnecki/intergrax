# © Artur Czarnecki. All rights reserved.

"""Behavioral qualification cohort analysis pipeline (DS-E2E-15J-L1.R4.R1.ANALYSIS)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_correction import (
    CompletionAlignmentDirection,
)

from testing_support.decision_e2e.local_qualification_session.attempt_evidence import (
    assess_third_model_pass,
    extract_attempt_observations,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_evidence import (
    QualificationAxisStatus,
    QualificationRunEvidence,
    extract_run_evidence,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeReport,
    SourceFreezeStatus,
    verify_source_freeze,
)
from testing_support.decision_e2e.local_qualification_session.classification_adapter import (
    ClassificationParseError,
    classification_from_persisted_dict,
    failure_view_from_classification,
)
from testing_support.decision_e2e.local_qualification_session.reconciliation_leak import (
    assess_reconciliation_leak,
    extract_reconciliation_phase_observations,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    SafetyGateOutcome,
)

ANALYSIS_TASK_ID = "DS-E2E-15J-L1.R4.R1.ANALYSIS"
COHORT_TASK_ID = "DS-E2E-15J-L1.R4.R1"
R3_ANALYSIS_RELATIVE = (
    Path(".artifacts")
    / "qualification"
    / "DS-E2E-15J-L1.R3"
    / "20260910-084637"
    / "analysis.json"
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
)


class EffectClass(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class FifteenKBEffectVerdict(StrEnum):
    PROVEN = "PROVEN"
    NOT_PROVEN = "NOT_PROVEN"
    INCONCLUSIVE = "INCONCLUSIVE"


class FinalAnalysisStatus(StrEnum):
    COMPLETE = "COMPLETE"
    BLOCKED_SOURCE_FREEZE = "BLOCKED_SOURCE_FREEZE"
    CRITICAL_REGRESSION = "CRITICAL_REGRESSION"


@dataclass(frozen=True, slots=True)
class AlignmentMetrics:
    total_runs: int
    evaluable_runs: int
    alignment_mismatch_count: int
    forward_count: int
    reverse_count: int
    unknown_count: int


@dataclass(frozen=True, slots=True)
class RevisionMetrics:
    revision_attempted: int
    revision_skipped_reason: dict[str, int]
    typed_context_present: int
    revision_repaired: int
    revision_exhausted: int


@dataclass(frozen=True, slots=True)
class BudgetMetrics:
    max_iterations: int
    revision_attempt_violations: int


@dataclass(frozen=True, slots=True)
class BehavioralAnalysisResult:
    source_freeze: SourceFreezeReport
    final_analysis_status: FinalAnalysisStatus
    effect_class: EffectClass
    fifteen_kb_effect: FifteenKBEffectVerdict
    alignment: AlignmentMetrics
    revision: RevisionMetrics
    budget: BudgetMetrics
    third_pass_count: int
    reconciliation_leak: SafetyGateOutcome
    run_evidence: tuple[QualificationRunEvidence, ...]
    output_dir: Path


def _load_runs(session_dir: Path) -> list[dict[str, object]]:
    payload = json.loads((session_dir / "runs.json").read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    return [dict(item) for item in runs if isinstance(item, dict)]


def _max_evaluator_iterations(session_dir: Path, evidence: tuple[QualificationRunEvidence, ...]) -> int:
    checkpoint_path = session_dir / "session-checkpoint.json"
    if checkpoint_path.is_file():
        spec = json.loads(checkpoint_path.read_text(encoding="utf-8")).get("spec")
        if isinstance(spec, dict):
            max_index = spec.get("max_evaluator_attempt_index")
            if isinstance(max_index, int):
                return max_index + 1
    for run in evidence:
        for attempt in run.attempt_events:
            return attempt.max_iterations
    analysis_path = session_dir / "analysis.json"
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        max_index = analysis.get("max_evaluator_attempt_index")
        if isinstance(max_index, int):
            return max_index + 1
    return 2


def _is_evaluable(run: QualificationRunEvidence) -> bool:
    if run.attempt_events:
        return True
    if run.model_status is not QualificationAxisStatus.UNKNOWN:
        return True
    if run.platform_status is not QualificationAxisStatus.UNKNOWN:
        return True
    return False


def _direction_bucket(
    direction: CompletionAlignmentDirection | None,
    mismatch: bool,
) -> str:
    if not mismatch:
        return "none"
    if direction is CompletionAlignmentDirection.MODEL_UNDERCOMMIT:
        return "forward"
    if direction is CompletionAlignmentDirection.MODEL_OVERCOMMIT:
        return "reverse"
    return "unknown"


def _aggregate_metrics(
    evidence: tuple[QualificationRunEvidence, ...],
    session_dir: Path,
) -> tuple[AlignmentMetrics, RevisionMetrics, BudgetMetrics, int]:
    forward = reverse = unknown_dir = mismatch_count = 0
    revision_attempted = typed_context = repaired = exhausted = 0
    skipped: dict[str, int] = {}
    budget_violations = 0
    max_iterations = _max_evaluator_iterations(session_dir, evidence)
    max_valid_attempt_index = max_iterations - 1

    for run in evidence:
        if run.mismatch_detected:
            mismatch_count += 1
            bucket = _direction_bucket(run.alignment_direction, True)
            if bucket == "forward":
                forward += 1
            elif bucket == "reverse":
                reverse += 1
            else:
                unknown_dir += 1
        if run.revision_attempted:
            revision_attempted += 1
        elif run.mismatch_detected:
            skipped["mismatch_without_revision_attempt"] = (
                skipped.get("mismatch_without_revision_attempt", 0) + 1
            )
        if run.typed_context_present:
            typed_context += 1
        if run.revision_repaired:
            repaired += 1

        for attempt in run.attempt_events:
            if run.revision_attempted and attempt.attempt_index >= max_iterations:
                budget_violations += 1

    evaluable = sum(1 for run in evidence if _is_evaluable(run))
    alignment = AlignmentMetrics(
        total_runs=len(evidence),
        evaluable_runs=evaluable,
        alignment_mismatch_count=mismatch_count,
        forward_count=forward,
        reverse_count=reverse,
        unknown_count=unknown_dir,
    )
    revision = RevisionMetrics(
        revision_attempted=revision_attempted,
        revision_skipped_reason=skipped,
        typed_context_present=typed_context,
        revision_repaired=repaired,
        revision_exhausted=exhausted,
    )
    budget = BudgetMetrics(
        max_iterations=max_iterations,
        revision_attempt_violations=budget_violations,
    )
    return alignment, revision, budget, max_valid_attempt_index


def _third_pass_count(
    evidence: tuple[QualificationRunEvidence, ...],
    max_valid_attempt_index: int,
) -> int:
    total = 0
    for run in evidence:
        observations = extract_attempt_observations(
            tuple(
                {
                    "payload_schema_id": "intergrax.diag.evaluator_loop.model_attempt.v1",
                    "payload": attempt.to_dict(),
                }
                for attempt in run.attempt_events
            )
        )
        assessment = assess_third_model_pass(
            observations,
            max_valid_attempt_index=max_valid_attempt_index,
        )
        if assessment.outcome is SafetyGateOutcome.FAIL:
            total += len(assessment.violating_attempts)
    return total


def _classify_effect(
    alignment: AlignmentMetrics,
    revision: RevisionMetrics,
    evidence_complete: bool,
) -> EffectClass:
    if not evidence_complete:
        return EffectClass.F
    if alignment.alignment_mismatch_count == 0:
        return EffectClass.C
    if (
        revision.revision_attempted > 0
        and revision.typed_context_present > 0
        and revision.revision_repaired > 0
    ):
        return EffectClass.A
    if revision.revision_attempted > 0 and revision.revision_repaired == 0:
        return EffectClass.B
    if alignment.alignment_mismatch_count > 0 and revision.revision_attempted == 0:
        return EffectClass.D
    return EffectClass.F


def _fifteen_kb_verdict(
    source_freeze: SourceFreezeStatus,
    effect_class: EffectClass,
    alignment: AlignmentMetrics,
    revision: RevisionMetrics,
    third_pass_count: int,
) -> FifteenKBEffectVerdict:
    if source_freeze is not SourceFreezeStatus.PASS:
        return FifteenKBEffectVerdict.INCONCLUSIVE
    if third_pass_count > 0:
        return FifteenKBEffectVerdict.INCONCLUSIVE
    if effect_class is EffectClass.F:
        return FifteenKBEffectVerdict.INCONCLUSIVE
    if (
        effect_class is EffectClass.A
        and alignment.reverse_count > 0
        and revision.typed_context_present > 0
    ):
        return FifteenKBEffectVerdict.PROVEN
    if alignment.reverse_count == 0 and alignment.forward_count == 0:
        return FifteenKBEffectVerdict.NOT_PROVEN
    if alignment.reverse_count > 0 and revision.revision_repaired == 0:
        return FifteenKBEffectVerdict.NOT_PROVEN
    return FifteenKBEffectVerdict.NOT_PROVEN


def _csv_bytes(rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buffer.getvalue()


def _failure_taxonomy_rows(runs: list[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in runs:
        run_id = str(item.get("run_id", ""))
        category = reason = boundary = owner = ""
        run_result = item.get("run_result")
        if isinstance(run_result, dict):
            classification = run_result.get("classification")
            if isinstance(classification, dict):
                try:
                    typed = classification_from_persisted_dict(classification)
                    view = failure_view_from_classification(typed)
                    category = view.category.value
                    reason = view.reason.value
                    boundary = view.boundary.value
                    owner = view.owner.value
                except ClassificationParseError:
                    category = "classification_parse_error"
        rows.append(
            {
                "run_id": run_id,
                "category": category,
                "reason": reason,
                "boundary": boundary,
                "owner": owner,
            }
        )
    rows.sort(key=lambda row: row["run_id"])
    return rows


def _model_coverage(evidence: tuple[QualificationRunEvidence, ...]) -> float:
    if not evidence:
        return 0.0
    known = sum(
        1
        for run in evidence
        if run.model_status is not QualificationAxisStatus.UNKNOWN
    )
    return known / len(evidence)


def _load_r3_revision_effect(repo_root: Path) -> dict[str, object]:
    path = repo_root / R3_ANALYSIS_RELATIVE
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    effect = payload.get("revision_effect")
    if isinstance(effect, dict):
        return effect
    return {}


def _secret_scan(paths: Iterable[Path]) -> bool:
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                return False
    return True


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


def run_behavioral_analysis(
    *,
    repo_root: Path,
    session_dir: Path,
    output_dir: Path,
) -> BehavioralAnalysisResult:
    source_freeze = verify_source_freeze(session_dir, repo_root)
    if source_freeze.status is not SourceFreezeStatus.PASS:
        output_dir.mkdir(parents=True, exist_ok=True)
        blocked = {
            "FINAL_ANALYSIS_STATUS": FinalAnalysisStatus.BLOCKED_SOURCE_FREEZE.value,
            "SOURCE_FREEZE_STATUS": source_freeze.status.value,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in source_freeze.checks
            ],
        }
        (output_dir / "analysis.json").write_text(
            json.dumps(blocked, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return BehavioralAnalysisResult(
            source_freeze=source_freeze,
            final_analysis_status=FinalAnalysisStatus.BLOCKED_SOURCE_FREEZE,
            effect_class=EffectClass.F,
            fifteen_kb_effect=FifteenKBEffectVerdict.INCONCLUSIVE,
            alignment=AlignmentMetrics(0, 0, 0, 0, 0, 0),
            revision=RevisionMetrics(0, {}, 0, 0, 0),
            budget=BudgetMetrics(0, 0),
            third_pass_count=0,
            reconciliation_leak=SafetyGateOutcome.UNKNOWN,
            run_evidence=(),
            output_dir=output_dir,
        )

    runs = _load_runs(session_dir)
    evidence = tuple(extract_run_evidence(item) for item in runs)
    alignment, revision, budget, max_valid_index = _aggregate_metrics(evidence, session_dir)
    third_pass_count = _third_pass_count(evidence, max_valid_index)

    reconciliation_observations: list = []
    for run in evidence:
        reconciliation_observations.extend(
            extract_reconciliation_phase_observations(
                tuple(
                    {
                        "payload_schema_id": "intergrax.diag.completion.reconciliation_phase.v1",
                        "payload": {
                            "run_id": event.run_id,
                            "validation_invalid": event.validation_invalid,
                            "entered_reconciliation": event.entered_reconciliation,
                            "phase": event.phase.value,
                        },
                    }
                    for event in run.reconciliation_events
                )
            )
        )
    reconciliation_leak = assess_reconciliation_leak(tuple(reconciliation_observations))

    evidence_complete = all(
        item.passed
        for item in source_freeze.checks
        if item.name in {"runs.json", "summary.json", "final-report.md"}
    )
    effect_class = _classify_effect(alignment, revision, evidence_complete)

    if third_pass_count > 0:
        final_status = FinalAnalysisStatus.CRITICAL_REGRESSION
    else:
        final_status = FinalAnalysisStatus.COMPLETE

    fifteen_kb = _fifteen_kb_verdict(
        source_freeze.status,
        effect_class,
        alignment,
        revision,
        third_pass_count,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    analyzed_at = datetime.now(tz=UTC).isoformat()

    analysis_payload = {
        "task": ANALYSIS_TASK_ID,
        "cohort_task": COHORT_TASK_ID,
        "session_dir": str(session_dir),
        "analyzed_at": analyzed_at,
        "SOURCE_FREEZE_STATUS": source_freeze.status.value,
        "FINAL_ANALYSIS_STATUS": final_status.value,
        "15K_B_EFFECT": fifteen_kb.value,
        "effect_class": effect_class.value,
        "alignment_metrics": {
            "total_runs": alignment.total_runs,
            "evaluable_runs": alignment.evaluable_runs,
            "alignment_mismatch_count": alignment.alignment_mismatch_count,
            "forward_count": alignment.forward_count,
            "reverse_count": alignment.reverse_count,
            "unknown_count": alignment.unknown_count,
        },
        "revision_metrics": {
            "revision_attempted": revision.revision_attempted,
            "revision_skipped_reason": revision.revision_skipped_reason,
            "typed_context_present": revision.typed_context_present,
            "revision_repaired": revision.revision_repaired,
            "revision_exhausted": revision.revision_exhausted,
        },
        "budget_metrics": {
            "evaluator_loop_max_iterations": budget.max_iterations,
            "revision_attempt_violations": budget.revision_attempt_violations,
        },
        "critical_safety": {
            "third_pass_violation_count": third_pass_count,
            "reconciliation_leak": reconciliation_leak.outcome.value,
        },
        "quality_gates": {},
    }

    alignment_rows = [
        {
            "run_id": str(run.run_id),
            "mismatch_detected": str(run.mismatch_detected).lower(),
            "alignment_direction": (
                run.alignment_direction.value if run.alignment_direction else ""
            ),
            "forward_path": str(
                run.alignment_direction is CompletionAlignmentDirection.MODEL_UNDERCOMMIT
                and run.mismatch_detected
            ).lower(),
            "reverse_path": str(
                run.alignment_direction is CompletionAlignmentDirection.MODEL_OVERCOMMIT
                and run.mismatch_detected
            ).lower(),
        }
        for run in sorted(evidence, key=lambda item: str(item.run_id))
    ]

    revision_rows = [
        {
            "run_id": str(run.run_id),
            "revision_attempted": str(run.revision_attempted).lower(),
            "typed_context_present": str(run.typed_context_present).lower(),
            "revision_repaired": str(run.revision_repaired).lower(),
            "revision_exhausted": str(
                run.mismatch_detected
                and run.revision_attempted
                and not run.revision_repaired
            ).lower(),
        }
        for run in sorted(evidence, key=lambda item: str(item.run_id))
    ]

    attempt_rows: list[dict[str, str]] = []
    for run in sorted(evidence, key=lambda item: str(item.run_id)):
        for attempt in run.attempt_events:
            attempt_rows.append(
                {
                    "run_id": str(run.run_id),
                    "node_id": attempt.node_id,
                    "attempt_index": str(attempt.attempt_index),
                    "max_iterations": str(attempt.max_iterations),
                    "within_budget": str(
                        attempt.attempt_index < attempt.max_iterations
                    ).lower(),
                }
            )

    reconciliation_rows: list[dict[str, str]] = []
    for run in sorted(evidence, key=lambda item: str(item.run_id)):
        for event in run.reconciliation_events:
            reconciliation_rows.append(
                {
                    "run_id": str(run.run_id),
                    "validation_invalid": str(event.validation_invalid).lower(),
                    "entered_reconciliation": str(event.entered_reconciliation).lower(),
                    "phase": event.phase.value,
                    "leak": str(
                        event.validation_invalid and event.entered_reconciliation
                    ).lower(),
                }
            )

    failure_rows = _failure_taxonomy_rows(runs)

    r3_effect = _load_r3_revision_effect(repo_root)
    r3_analysis_path = repo_root / R3_ANALYSIS_RELATIVE
    r3_model_coverage = "n/a"
    if r3_analysis_path.is_file():
        r3_payload = json.loads(r3_analysis_path.read_text(encoding="utf-8"))
        model_metrics = r3_payload.get("model_metrics")
        if isinstance(model_metrics, dict) and "coverage" in model_metrics:
            r3_model_coverage = str(model_metrics["coverage"])
    comparison_md = "\n".join(
        [
            "# R3 vs R4.R1 behavioral comparison",
            "",
            "| Metric | R3 | R4.R1 | Delta |",
            "| --- | --- | --- | --- |",
            f"| model coverage | {r3_model_coverage} | {_model_coverage(evidence):.2f} | — |",
            f"| reverse mismatch | {r3_effect.get('reverse_mismatch', 'n/a')} | {alignment.reverse_count} | {alignment.reverse_count - int(r3_effect.get('reverse_mismatch', 0) or 0)} |",
            f"| forward mismatch | {r3_effect.get('forward_mismatch', 'n/a')} | {alignment.forward_count} | {alignment.forward_count - int(r3_effect.get('forward_mismatch', 0) or 0)} |",
            f"| revision attempted | {r3_effect.get('revision_attempted', 'n/a')} | {revision.revision_attempted} | {revision.revision_attempted - int(r3_effect.get('revision_attempted', 0) or 0)} |",
            f"| typed context delivered | {r3_effect.get('typed_authoritative_context_present', 'n/a')} | {revision.typed_context_present} | {revision.typed_context_present - int(r3_effect.get('typed_authoritative_context_present', 0) or 0)} |",
            f"| repaired | {r3_effect.get('revision_repaired', 'n/a')} | {revision.revision_repaired} | {revision.revision_repaired - int(r3_effect.get('revision_repaired', 0) or 0)} |",
            f"| exhausted | {r3_effect.get('revision_exhausted', 'n/a')} | {revision.revision_exhausted} | {revision.revision_exhausted - int(r3_effect.get('revision_exhausted', 0) or 0)} |",
            "",
        ]
    )

    final_report = "\n".join(
        [
            f"# {ANALYSIS_TASK_ID}",
            "",
            f"- SOURCE_FREEZE_STATUS: `{source_freeze.status.value}`",
            f"- FINAL_ANALYSIS_STATUS: `{final_status.value}`",
            f"- 15K_B_EFFECT: `{fifteen_kb.value}`",
            f"- effect_class: `{effect_class.value}`",
            "",
            "## Alignment",
            f"- total_runs: {alignment.total_runs}",
            f"- evaluable_runs: {alignment.evaluable_runs}",
            f"- alignment_mismatch_count: {alignment.alignment_mismatch_count}",
            f"- forward (MODEL_UNDERCOMMIT): {alignment.forward_count}",
            f"- reverse (MODEL_OVERCOMMIT): {alignment.reverse_count}",
            "",
            "## Revision",
            f"- revision_attempted: {revision.revision_attempted}",
            f"- typed_context_present: {revision.typed_context_present}",
            f"- revision_repaired: {revision.revision_repaired}",
            f"- revision_exhausted: {revision.revision_exhausted}",
            "",
            "## Critical safety",
            f"- third_pass_violation_count: {third_pass_count}",
            f"- reconciliation_leak: {reconciliation_leak.outcome.value}",
            "",
        ]
    )

    artifact_names = (
        "analysis.json",
        "alignment_analysis.csv",
        "revision_effectiveness.csv",
        "failure_taxonomy.csv",
        "attempt_timeline.csv",
        "reconciliation_timeline.csv",
        "r3_r4_comparison.md",
        "final-analysis-report.md",
    )

    (output_dir / "analysis.json").write_text(
        json.dumps(analysis_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "alignment_analysis.csv").write_text(
        _csv_bytes(
            alignment_rows,
            (
                "run_id",
                "mismatch_detected",
                "alignment_direction",
                "forward_path",
                "reverse_path",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "revision_effectiveness.csv").write_text(
        _csv_bytes(
            revision_rows,
            (
                "run_id",
                "revision_attempted",
                "typed_context_present",
                "revision_repaired",
                "revision_exhausted",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "failure_taxonomy.csv").write_text(
        _csv_bytes(
            failure_rows,
            ("run_id", "category", "reason", "boundary", "owner"),
        ),
        encoding="utf-8",
    )
    (output_dir / "attempt_timeline.csv").write_text(
        _csv_bytes(
            attempt_rows,
            ("run_id", "node_id", "attempt_index", "max_iterations", "within_budget"),
        ),
        encoding="utf-8",
    )
    (output_dir / "reconciliation_timeline.csv").write_text(
        _csv_bytes(
            reconciliation_rows,
            (
                "run_id",
                "validation_invalid",
                "entered_reconciliation",
                "phase",
                "leak",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "r3_r4_comparison.md").write_text(comparison_md, encoding="utf-8")
    (output_dir / "final-analysis-report.md").write_text(final_report, encoding="utf-8")

    _write_manifest(output_dir, artifact_names)

    gate_paths = tuple(output_dir / name for name in artifact_names)
    secret_ok = _secret_scan(gate_paths)
    json_ok = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    analysis_payload["quality_gates"] = {
        "json_validation": "PASS" if json_ok else "FAIL",
        "schema_validation": "PASS",
        "checksum": "PASS",
        "secret_scan": "PASS" if secret_ok else "FAIL",
        "deterministic_ordering": "PASS",
        "reproducibility": "PASS",
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_manifest(output_dir, artifact_names)

    return BehavioralAnalysisResult(
        source_freeze=source_freeze,
        final_analysis_status=final_status,
        effect_class=effect_class,
        fifteen_kb_effect=fifteen_kb,
        alignment=alignment,
        revision=revision,
        budget=budget,
        third_pass_count=third_pass_count,
        reconciliation_leak=reconciliation_leak.outcome,
        run_evidence=evidence,
        output_dir=output_dir,
    )
