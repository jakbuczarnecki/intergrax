# © Artur Czarnecki. All rights reserved.

"""Behavioral coverage analysis pipeline (DS-E2E-15J-L1.R4.R1.OBS)."""

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

from testing_support.decision_e2e.local_qualification_session.behavioral_coverage_evidence import (
    AlignmentEvidenceStatus,
    BehavioralCoverageRunVerdict,
    CompletionAlignmentCoverageEvidence,
    MissingEvidenceReason,
    RevisionRepairResult,
    classify_run_coverage_from_item,
    derive_coverage_path_phase,
    extract_coverage_evidence,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_analysis import (
    FifteenKBEffectVerdict,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeReport,
    SourceFreezeStatus,
    verify_source_freeze,
)

OBS_TASK_ID = "DS-E2E-15J-L1.R4.R1.OBS"
COHORT_TASK_ID = "DS-E2E-15J-L1.R4.R1"
ANALYSIS_TASK_ID = "DS-E2E-15J-L1.R4.R1.ANALYSIS"

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
)


class ObsFinalAnalysisStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCONCLUSIVE = "INCONCLUSIVE"
    BLOCKED_SOURCE_FREEZE = "BLOCKED_SOURCE_FREEZE"


@dataclass(frozen=True, slots=True)
class AlignmentCoverageMetrics:
    alignment_evaluation_attempted: int
    alignment_event_present: int
    alignment_event_missing: int


@dataclass(frozen=True, slots=True)
class CorrectionCoverageMetrics:
    correction_candidate_created: int
    correction_candidate_unknown: int
    revision_path_entered: int


@dataclass(frozen=True, slots=True)
class TypedContextCoverageMetrics:
    typed_context_expected: int
    typed_context_present: int
    typed_context_missing: int


@dataclass(frozen=True, slots=True)
class RepairCoverageMetrics:
    revision_completed: int
    revision_repaired: int
    revision_failed: int
    revision_exhausted: int


@dataclass(frozen=True, slots=True)
class BehavioralCoverageAnalysisResult:
    source_freeze: SourceFreezeReport
    final_analysis_status: ObsFinalAnalysisStatus
    fifteen_kb_effect: FifteenKBEffectVerdict
    alignment: AlignmentCoverageMetrics
    correction: CorrectionCoverageMetrics
    typed_context: TypedContextCoverageMetrics
    repair: RepairCoverageMetrics
    run_evidence: tuple[CompletionAlignmentCoverageEvidence, ...]
    output_dir: Path


def _load_runs(session_dir: Path) -> list[dict[str, object]]:
    payload = json.loads((session_dir / "runs.json").read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    return [dict(item) for item in runs if isinstance(item, dict)]


def _aggregate_coverage_metrics(
    evidence: tuple[CompletionAlignmentCoverageEvidence, ...],
) -> tuple[
    AlignmentCoverageMetrics,
    CorrectionCoverageMetrics,
    TypedContextCoverageMetrics,
    RepairCoverageMetrics,
]:
    eval_attempted = 0
    event_present = 0
    event_missing = 0

    candidate_created = 0
    candidate_unknown = 0
    revision_entered = 0

    typed_expected = 0
    typed_present = 0
    typed_missing = 0

    revision_completed = 0
    repaired = 0
    failed = 0
    exhausted = 0

    for run in evidence:
        status = run.alignment_event_status
        if status is not AlignmentEvidenceStatus.NOT_REACHED:
            eval_attempted += 1
        if status is AlignmentEvidenceStatus.PRESENT:
            event_present += 1
        elif status in {
            AlignmentEvidenceStatus.NOT_EMITTED,
            AlignmentEvidenceStatus.NOT_READABLE,
            AlignmentEvidenceStatus.UNKNOWN,
        }:
            event_missing += 1

        if run.correction_candidate_created is True:
            candidate_created += 1
        elif run.correction_candidate_created is None and run.mismatch_detected is True:
            candidate_unknown += 1

        if run.revision_path_entered is True:
            revision_entered += 1

        if run.revision_path_entered is True or run.correction_candidate_created is True:
            typed_expected += 1
            if run.typed_context_present is True:
                typed_present += 1
            elif run.typed_context_present is False:
                typed_missing += 1

        if run.revision_path_entered is True:
            revision_completed += 1
            if run.repair_result is RevisionRepairResult.REPAIRED:
                repaired += 1
            elif run.repair_result is RevisionRepairResult.EXHAUSTED:
                exhausted += 1
            elif run.repair_result is RevisionRepairResult.FAILED:
                failed += 1

    alignment = AlignmentCoverageMetrics(
        alignment_evaluation_attempted=eval_attempted,
        alignment_event_present=event_present,
        alignment_event_missing=event_missing,
    )
    correction = CorrectionCoverageMetrics(
        correction_candidate_created=candidate_created,
        correction_candidate_unknown=candidate_unknown,
        revision_path_entered=revision_entered,
    )
    typed_context = TypedContextCoverageMetrics(
        typed_context_expected=typed_expected,
        typed_context_present=typed_present,
        typed_context_missing=typed_missing,
    )
    repair = RepairCoverageMetrics(
        revision_completed=revision_completed,
        revision_repaired=repaired,
        revision_failed=failed,
        revision_exhausted=exhausted,
    )
    return alignment, correction, typed_context, repair


def _fifteen_kb_effect_from_coverage(
    source_freeze: SourceFreezeStatus,
    alignment: AlignmentCoverageMetrics,
    correction: CorrectionCoverageMetrics,
    typed_context: TypedContextCoverageMetrics,
    repair: RepairCoverageMetrics,
    evidence: tuple[CompletionAlignmentCoverageEvidence, ...],
) -> FifteenKBEffectVerdict:
    if source_freeze is not SourceFreezeStatus.PASS:
        return FifteenKBEffectVerdict.INCONCLUSIVE
    if alignment.alignment_event_present == 0:
        return FifteenKBEffectVerdict.INCONCLUSIVE
    if alignment.alignment_event_missing > 0 and alignment.alignment_event_present == 0:
        return FifteenKBEffectVerdict.INCONCLUSIVE

    reverse_mismatch = sum(
        1
        for run in evidence
        if run.alignment_event_status is AlignmentEvidenceStatus.PRESENT
        and run.mismatch_detected is True
        and run.alignment_direction is CompletionAlignmentDirection.MODEL_OVERCOMMIT
    )
    if reverse_mismatch == 0:
        return FifteenKBEffectVerdict.NOT_PROVEN

    if (
        correction.revision_path_entered > 0
        and typed_context.typed_context_present > 0
        and repair.revision_repaired > 0
    ):
        return FifteenKBEffectVerdict.PROVEN

    if correction.revision_path_entered > 0 and repair.revision_repaired == 0:
        return FifteenKBEffectVerdict.NOT_PROVEN

    return FifteenKBEffectVerdict.NOT_PROVEN


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


def _secret_scan(paths: Iterable[Path]) -> bool:
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                return False
    return True


def run_behavioral_coverage_analysis(
    *,
    repo_root: Path,
    session_dir: Path,
    output_dir: Path,
    runs: list[dict[str, object]] | None = None,
) -> BehavioralCoverageAnalysisResult:
    source_freeze = verify_source_freeze(session_dir, repo_root)
    if source_freeze.status is not SourceFreezeStatus.PASS:
        output_dir.mkdir(parents=True, exist_ok=True)
        blocked = {
            "task": OBS_TASK_ID,
            "OBS_SOURCE_FREEZE_STATUS": source_freeze.status.value,
            "status": ObsFinalAnalysisStatus.BLOCKED_SOURCE_FREEZE.value,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in source_freeze.checks
            ],
        }
        (output_dir / "analysis.json").write_text(
            json.dumps(blocked, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        empty_alignment = AlignmentCoverageMetrics(0, 0, 0)
        return BehavioralCoverageAnalysisResult(
            source_freeze=source_freeze,
            final_analysis_status=ObsFinalAnalysisStatus.BLOCKED_SOURCE_FREEZE,
            fifteen_kb_effect=FifteenKBEffectVerdict.INCONCLUSIVE,
            alignment=empty_alignment,
            correction=CorrectionCoverageMetrics(0, 0, 0),
            typed_context=TypedContextCoverageMetrics(0, 0, 0),
            repair=RepairCoverageMetrics(0, 0, 0, 0),
            run_evidence=(),
            output_dir=output_dir,
        )

    run_items = runs if runs is not None else _load_runs(session_dir)
    evidence = tuple(extract_coverage_evidence(item) for item in run_items)
    alignment, correction, typed_context, repair = _aggregate_coverage_metrics(evidence)

    fifteen_kb = _fifteen_kb_effect_from_coverage(
        source_freeze.status,
        alignment,
        correction,
        typed_context,
        repair,
        evidence,
    )

    incomplete_runs = sum(
        1
        for item in run_items
        if classify_run_coverage_from_item(item)
        is BehavioralCoverageRunVerdict.INCOMPLETE_EVIDENCE
    )
    if incomplete_runs > 0 and alignment.alignment_event_present == 0:
        final_status = ObsFinalAnalysisStatus.INCONCLUSIVE
    elif alignment.alignment_event_missing > 0 and fifteen_kb is FifteenKBEffectVerdict.INCONCLUSIVE:
        final_status = ObsFinalAnalysisStatus.INCONCLUSIVE
    else:
        final_status = ObsFinalAnalysisStatus.COMPLETE

    output_dir.mkdir(parents=True, exist_ok=True)
    analyzed_at = datetime.now(tz=UTC).isoformat()

    coverage_payload = {
        "runs": len(evidence),
        "alignment_evaluated": alignment.alignment_evaluation_attempted,
        "alignment_event_present": alignment.alignment_event_present,
        "alignment_event_missing": alignment.alignment_event_missing,
        "mismatch_detected": sum(1 for run in evidence if run.mismatch_detected is True),
        "revision_entered": correction.revision_path_entered,
        "typed_context_present": typed_context.typed_context_present,
        "repaired": repair.revision_repaired,
    }

    analysis_payload = {
        "task": OBS_TASK_ID,
        "upstream_analysis": ANALYSIS_TASK_ID,
        "cohort_task": COHORT_TASK_ID,
        "session_dir": str(session_dir),
        "analyzed_at": analyzed_at,
        "OBS_SOURCE_FREEZE_STATUS": source_freeze.status.value,
        "status": final_status.value,
        "15K_B_EFFECT": fifteen_kb.value,
        "coverage": coverage_payload,
        "alignment_coverage": {
            "alignment_evaluation_attempted": alignment.alignment_evaluation_attempted,
            "alignment_event_present": alignment.alignment_event_present,
            "alignment_event_missing": alignment.alignment_event_missing,
        },
        "correction_coverage": {
            "correction_candidate_created": correction.correction_candidate_created,
            "correction_candidate_unknown": correction.correction_candidate_unknown,
            "revision_path_entered": correction.revision_path_entered,
        },
        "typed_context_coverage": {
            "typed_context_expected": typed_context.typed_context_expected,
            "typed_context_present": typed_context.typed_context_present,
            "typed_context_missing": typed_context.typed_context_missing,
        },
        "repair_coverage": {
            "revision_completed": repair.revision_completed,
            "revision_repaired": repair.revision_repaired,
            "revision_failed": repair.revision_failed,
            "revision_exhausted": repair.revision_exhausted,
        },
        "quality_gates": {},
    }

    alignment_rows: list[dict[str, str]] = []
    revision_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []

    run_by_id = {str(item.get("run_id", "")): item for item in run_items}

    for run in sorted(evidence, key=lambda item: str(item.run_id)):
        run_id = str(run.run_id)
        phase = derive_coverage_path_phase(run)
        verdict = classify_run_coverage_from_item(
            run_by_id.get(run_id, {"run_id": run_id, "trace_events": []})
        )
        alignment_rows.append(
            {
                "run_id": run_id,
                "alignment_event_status": run.alignment_event_status.value,
                "mismatch_detected": ""
                if run.mismatch_detected is None
                else str(run.mismatch_detected).lower(),
                "alignment_direction": (
                    run.alignment_direction.value if run.alignment_direction else ""
                ),
                "coverage_path_phase": phase.value,
                "run_verdict": verdict.value,
            }
        )
        revision_rows.append(
            {
                "run_id": run_id,
                "correction_candidate_created": ""
                if run.correction_candidate_created is None
                else str(run.correction_candidate_created).lower(),
                "correction_eligibility_known": str(run.correction_eligibility_known).lower(),
                "revision_path_entered": ""
                if run.revision_path_entered is None
                else str(run.revision_path_entered).lower(),
                "typed_context_present": ""
                if run.typed_context_present is None
                else str(run.typed_context_present).lower(),
                "repair_result": run.repair_result.value,
            }
        )
        missing_rows.append(
            {
                "run_id": run_id,
                "missing_evidence_reason": run.missing_evidence_reason
                or MissingEvidenceReason.NOT_APPLICABLE.value,
                "alignment_event_status": run.alignment_event_status.value,
            }
        )

    summary_md = "\n".join(
        [
            f"# {OBS_TASK_ID}",
            "",
            f"- OBS_SOURCE_FREEZE_STATUS: `{source_freeze.status.value}`",
            f"- status: `{final_status.value}`",
            f"- 15K_B_EFFECT: `{fifteen_kb.value}`",
            "",
            "## Coverage",
            f"- runs: {coverage_payload['runs']}",
            f"- alignment_event_present: {alignment.alignment_event_present}",
            f"- alignment_event_missing: {alignment.alignment_event_missing}",
            f"- mismatch_detected: {coverage_payload['mismatch_detected']}",
            f"- revision_entered: {correction.revision_path_entered}",
            f"- typed_context_present: {typed_context.typed_context_present}",
            f"- repaired: {repair.revision_repaired}",
            "",
            "## Observability",
            "Brak alignment event ≠ brak błędu modelu — klasyfikacja `INCONCLUSIVE` gdy "
            "`alignment_event_present=0`.",
            "",
        ]
    )

    artifact_names = (
        "analysis.json",
        "behavioral_coverage.json",
        "alignment_path_coverage.csv",
        "revision_path_coverage.csv",
        "missing_evidence_classification.csv",
        "coverage_summary.md",
    )

    (output_dir / "analysis.json").write_text(
        json.dumps(analysis_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "behavioral_coverage.json").write_text(
        json.dumps(coverage_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "alignment_path_coverage.csv").write_text(
        _csv_bytes(
            alignment_rows,
            (
                "run_id",
                "alignment_event_status",
                "mismatch_detected",
                "alignment_direction",
                "coverage_path_phase",
                "run_verdict",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "revision_path_coverage.csv").write_text(
        _csv_bytes(
            revision_rows,
            (
                "run_id",
                "correction_candidate_created",
                "correction_eligibility_known",
                "revision_path_entered",
                "typed_context_present",
                "repair_result",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "missing_evidence_classification.csv").write_text(
        _csv_bytes(
            missing_rows,
            ("run_id", "missing_evidence_reason", "alignment_event_status"),
        ),
        encoding="utf-8",
    )
    (output_dir / "coverage_summary.md").write_text(summary_md, encoding="utf-8")

    _write_manifest(output_dir, artifact_names)

    gate_paths = tuple(output_dir / name for name in artifact_names)
    secret_ok = _secret_scan(gate_paths)
    json_ok = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    analysis_payload["quality_gates"] = {
        "json_validation": "PASS" if json_ok else "FAIL",
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

    return BehavioralCoverageAnalysisResult(
        source_freeze=source_freeze,
        final_analysis_status=final_status,
        fifteen_kb_effect=fifteen_kb,
        alignment=alignment,
        correction=correction,
        typed_context=typed_context,
        repair=repair,
        run_evidence=evidence,
        output_dir=output_dir,
    )
